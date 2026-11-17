"""INT8 版 SD-Turbo 端到端（torch，不依赖 ComfyUI）：CLIP(fp) → UNet(INT8)×4 步(v-pred) → VAE。

权重量化：per-channel int8；激活：per-row int8；矩阵乘：int8×int8→int32（_int_mm）。
这正好是 32 路矩阵单元要做的运算，之后只需把 _int_mm 换成引擎即可对齐。

用法: python sample_int8.py [--fp] [--steps 4] [--seed 42]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sd_clip
import sd_unet
import sd_unet_q8
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
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--fp", action="store_true", help="用浮点 UNet 做对照")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="q8")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(exist_ok=True)
    t_all = time.perf_counter()

    # --- 文本编码（浮点参考实现，与 ComfyUI 一致）
    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode(args.prompt)
    cw = sd_clip.ClipWeights(CKPT)
    ctx_np = sd_clip.text_encoder_float(cw, ids)
    cw._c.clear()
    ctx = torch.from_numpy(ctx_np).to("cuda")
    print(f"[int8] ctx ready {ctx.shape}", flush=True)

    # --- 噪声
    noise = np.load(out / f"noise_seed{args.seed}.npy") if (out / f"noise_seed{args.seed}.npy").exists() else None
    if noise is None:
        import torch as th
        g = th.Generator().manual_seed(args.seed)
        noise = th.randn((1, 4, 64, 64), generator=g).numpy()
        np.save(out / f"noise_seed{args.seed}.npy", noise)
    sig = SIGMAS[: args.steps + 1]
    ts = TIMESTEPS[: args.steps]
    x = torch.from_numpy(np.transpose(noise[0], (1, 2, 0)).astype(np.float32) * np.float32(np.sqrt(1 + sig[0] ** 2))).to("cuda")

    qw = None if args.fp else sd_unet_q8.W(CKPT)
    fw = sd_unet.UNetWeights(CKPT, cache=False) if args.fp else None

    for i, t in enumerate(ts):
        t0 = time.perf_counter()
        if args.fp:
            eps = torch.from_numpy(sd_unet.forward(fw, x.cpu().numpy(), float(t), ctx_np)).to("cuda")
        else:
            eps = sd_unet_q8.forward(qw, x, float(t), ctx)
        s, s_next = sig[i], sig[i + 1]
        # v-prediction：x0 = x/(s²+1) - v·s/√(s²+1)
        den = x / (s * s + 1.0) - eps * (s / np.sqrt(s * s + 1.0))
        d = (x - den) / s
        x = x + d * (s_next - s)
        z = np.transpose(x.cpu().numpy(), (2, 0, 1))
        np.save(out / f"latent_{args.tag}_s{i}.npy", z)
        print(f"  step {i} t={t:4d} {s:.4f}->{s_next:.4f}  {time.perf_counter()-t0:5.1f}s  "
              f"std={z.std():.3f}  range[{z.min():.1f},{z.max():.1f}]", flush=True)

    t0 = time.perf_counter()
    np.save(out / (f"latent_{args.tag}_final.npy"), z)
    print("[int8] latent saved:", out / (f"latent_{args.tag}_final.npy"), flush=True)
    # 释放 UNet 的 int8 权重，否则 512² 的 VAE 解码会显存不足
    qw = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[int8] VRAM used {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)
    t0 = time.perf_counter()
    img = sd_vae_t.decoder(sd_vae_t.TWeights(sd_vae.Weights(CKPT)), z[None], 1.0)
    print(f"[int8] VAE {time.perf_counter()-t0:.1f}s", flush=True)
    path = out / f"img_{args.tag}_seed{args.seed}.png"
    Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
    print(json.dumps({"image": str(path), "total": round(time.perf_counter() - t_all, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
