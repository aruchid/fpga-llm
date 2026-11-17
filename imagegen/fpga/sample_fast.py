"""快速端到端：CLIP(numpy 定点) → UNet(torch CUDA) ×4 步 → VAE(torch CUDA)。

用与 ComfyUI 相同的噪声和 sigma 序列，逐步保存 latent 便于对拍。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import fx
import fx_torch as ft
import sd_clip
import sd_unet
import sd_unet_t
import sd_vae
import sd_vae_t

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"
SIGMAS = [14.614640235900879, 2.918307065963745, 0.9323579668998718, 0.029167160391807556, 0.0]
TIMESTEPS = [999, 666, 333, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", default="out/noise_seed42.npy")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="fast")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--no-decode", action="store_true", help="只采样存 latent，解码另起进程")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    t_all = time.perf_counter()

    # ComfyUI 若在跑，先让它把模型卸掉，否则我们连 safetensors 都打不开
    try:
        import json as _json
        import urllib.request as _ur
        req = _ur.Request("http://127.0.0.1:8188/free",
                          data=_json.dumps({"unload_models": True, "free_memory": True}).encode(),
                          headers={"Content-Type": "application/json"})
        _ur.urlopen(req, timeout=30).read()
        time.sleep(2)
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 文本编码（numpy 定点，7.5s）
    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode(args.prompt)
    cw = sd_clip.ClipWeights(CKPT)
    # 文本编码器用浮点参考实现：它与 ComfyUI 的 context 完全一致（cos=1.0），
    # 定点版在这个编码器上量化损失过大（cos≈0.19）。按"不能纯 int8 就保持原样"处理。
    ctx_np = sd_clip.text_encoder_float(cw, ids)
    ctx = ft.to_fx(ctx_np.astype(np.float64))
    cw._c.clear()
    del ctx_np
    gc.collect()
    print(f"[fast] prompt tokens={sum(1 for t in ids if t != sd_clip.EOT)} ctx ready", flush=True)

    # --- 噪声
    noise = np.load(args.noise) if Path(args.noise).exists() else None
    if noise is None:
        rng = np.random.default_rng(args.seed)
        noise = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
    sig = SIGMAS[: args.steps + 1]
    ts = TIMESTEPS[: args.steps]
    x = torch.from_numpy(np.transpose(noise[0], (1, 2, 0)).astype(np.float64) * np.sqrt(1.0 + sig[0] ** 2)).cuda()

    uw = sd_unet.UNetWeights(CKPT, cache=False)
    tw = sd_unet_t.TWeights(uw)
    xq = ft.to_fx(x.cpu().numpy())

    for i, t in enumerate(ts):
        t0 = time.perf_counter()
        eps = sd_unet_t.forward(tw, xq, float(t), ctx)
        s, s_next = sig[i], sig[i + 1]
        x_f = xq.double() / 65536.0
        x0 = x_f - s * (eps.double() / 65536.0)
        x_f = (x_f + (s_next - s) * ((x_f - x0) / s)) if s > 0 else x0
        xq = ft.to_fx(x_f.cpu().numpy())
        z = xq.cpu().numpy().astype(np.float64) / 65536.0
        np.save(out / f"latent_{args.tag}_s{i}.npy", np.transpose(z, (2, 0, 1)))
        print(f"  step {i} t={t:4d} sigma {s:.4f}->{s_next:.4f}  {time.perf_counter()-t0:5.1f}s  "
              f"std={z.std():.3f} range[{z.min():.2f},{z.max():.2f}]", flush=True)

    # --- 解码
    np.save(out / f"latent_{args.tag}_final.npy", np.transpose(z, (2, 0, 1)))
    print(f"[fast] latent -> {out / f'latent_{args.tag}_final.npy'}", flush=True)
    if args.no_decode:
        return 0

    t0 = time.perf_counter()
    img = sd_vae_t.decoder(sd_vae_t.TWeights(sd_vae.Weights(CKPT)),
                           np.transpose(z, (2, 0, 1))[None], 1.0)
    print(f"[fast] VAE {time.perf_counter()-t0:.1f}s", flush=True)
    path = out / f"img_{args.tag}_seed{args.seed}.png"
    Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
    print(json.dumps({"image": str(path), "total_seconds": round(time.perf_counter() - t_all, 1)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
