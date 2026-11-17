"""端到端采样：CLIP → UNet×4 步 Euler → VAE 解码。

sigma 序列直接用 ComfyUI 的 normal 调度（4 步）：
    14.6146, 2.9183, 0.93236, 0.029167, 0
时间步：999, 666, 333, 0（v-prediction）

用法:
    python sample.py --mode float --prompt "..." --seed 42
    python sample.py --mode fx                 # 全部算子走 32 路引擎
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import fx
import mvm32
import mvm32_cuda
import sd_clip
import sd_unet
import sd_vae

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
SIGMAS = np.array([14.614640235900879, 2.918307065963745, 0.9323579668998718, 0.029167160391807556, 0.0])
TIMESTEPS = [999.0, 666.0, 333.0, 0.0]
DEFAULT_PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def latent_noise(seed: int, shape=(64, 64, 4), source: str = "torch") -> np.ndarray:
    if source == "torch":
        import torch
        g = torch.Generator().manual_seed(seed)
        z = torch.randn((1, 4, shape[0], shape[1]), generator=g).numpy()
        return np.transpose(z[0], (1, 2, 0))
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def euler_float(z, ctx, wt_unet, sigmas=SIGMAS, timesteps=TIMESTEPS, v_pred=False, log=print):
    x = z * np.sqrt(1.0 + sigmas[0] ** 2)
    for i, t in enumerate(timesteps):
        s, s_next = sigmas[i], sigmas[i + 1]
        t0 = time.perf_counter()
        out = sd_unet.forward(wt_unet, x, t, ctx)
        if v_pred:
            c_skip = 1.0 / (s * s + 1.0)
            c_out = -s / np.sqrt(s * s + 1.0)
            x0 = c_skip * x + c_out * out
        else:
            x0 = x - s * out
        d = (x - x0) / s
        x = x + d * (s_next - s)
        log(f"  step {i} t={t:.0f} sigma={s:.4f}->{s_next:.4f} {time.perf_counter()-t0:.1f}s")
    return x


def euler_fx(z, ctx_q, wt_unet, engine, sigmas=SIGMAS, timesteps=TIMESTEPS, v_pred=False, log=print):
    import sd_unet_fx
    x = fx.to_fx(z.astype(np.float64) * np.sqrt(1.0 + sigmas[0] ** 2))
    for i, t in enumerate(timesteps):
        s, s_next = sigmas[i], sigmas[i + 1]
        t0 = time.perf_counter()
        out = sd_unet_fx.forward(wt_unet, x, t, ctx_q, engine)
        out_f = fx.from_fx(out)
        x_f = fx.from_fx(x)
        if v_pred:
            c_skip = 1.0 / (s * s + 1.0)
            c_out = -s / np.sqrt(s * s + 1.0)
            x0 = c_skip * x_f + c_out * out_f
        else:
            x0 = x_f - s * out_f
        d = (x_f - x0) / s
        x = fx.to_fx(x_f + d * (s_next - s))
        log(f"  step {i} t={t:.0f} sigma={s:.4f}->{s_next:.4f} {time.perf_counter()-t0:.1f}s")
    return x


def make_engine(backend: str):
    if backend == "numpy":
        return mvm32.MVM32()
    if backend == "cuda":
        return mvm32_cuda.MVM32CUDA()
    return mvm32_cuda.HybridEngine()
def schedule(steps: int):
    """复刻 ComfyUI normal 调度：timesteps = linspace(999, 0, steps)，sigma 用 linear beta 表。"""
    betas = np.linspace(np.sqrt(0.00085), np.sqrt(0.012), 1000, dtype=np.float64) ** 2
    ac = np.cumprod(1.0 - betas)

    def sig(t):
        return float(np.sqrt((1.0 - ac[int(round(t))]) / ac[int(round(t))]))

    ts = np.linspace(999.0, 0.0, steps)
    ts_int = [float(round(t)) for t in ts]
    sg = np.array([sig(t) for t in ts_int] + [0.0])
    return sg, ts_int


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="float", choices=["float", "fx"])
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", default="torch", choices=["torch", "numpy"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-decode", action="store_true", help="只采样，保存 latent，稍后单独解码")
    ap.add_argument("--vpred", action="store_true", help="按 v-prediction 解释模型输出（默认 eps）")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "cuda", "hybrid"], help="矩阵乘后端")
    ap.add_argument("--steps", type=int, default=4, help="采样步数（ComfyUI normal 调度: linspace(999,0,steps)）")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode(args.prompt)
    cw = sd_clip.ClipWeights(CKPT)
    uw = sd_unet.UNetWeights(CKPT, cache=False)

    if args.mode == "float":
        ctx = sd_clip.text_encoder_float(cw, ids)
        z = latent_noise(args.seed, source=args.noise)
        print(f"[sample] float  ctx {ctx.shape} noise {z.shape} std={z.std():.3f}")
        cw._c.clear()
        import gc; gc.collect()
        z = euler_float(z, ctx, uw, *schedule(args.steps), v_pred=args.vpred)
        extra = {}
    else:
        eng = make_engine(args.backend)
        ctx_q = sd_clip.text_encoder_fx(cw, ids, eng)
        z = latent_noise(args.seed, source=args.noise)
        print(f"[sample] fx  ctx {ctx_q.shape}")
        cw._c.clear()              # CLIP 权重用完立刻放掉，UNet 阶段不需要
        import gc; gc.collect()
        z = euler_fx(z, ctx_q, uw, eng, *schedule(args.steps), v_pred=args.vpred)
        st = eng.stats
        extra = {"macs_g": round(st.macs / 1e9, 1), "cycles_g": round(st.cycles / 1e9, 2),
                 "hw_seconds": round(eng.cycles_to_seconds(st.cycles), 1)}

    lat = fx.from_fx(z) if args.mode == "fx" else z
    lat_path = out_dir / f"latent_{args.mode}{args.tag}_seed{args.seed}.npy"
    np.save(lat_path, np.transpose(lat, (2, 0, 1)))
    print(f"[sample] latent -> {lat_path}  range [{lat.min():.2f},{lat.max():.2f}]")

    if args.no_decode:
        print("[sample] skip decode")
        return 0

    if args.mode == "float":
        img = sd_vae.decoder_float(sd_vae.Weights(CKPT), np.transpose(z, (2, 0, 1))[None], 1.0)
    else:
        img = sd_vae.decoder_fx(sd_vae.Weights(CKPT), np.transpose(lat, (2, 0, 1))[None], eng, 1.0)

    name = f"fpga_{args.mode}{args.tag}_seed{args.seed}.png"
    Image.fromarray((img * 255).round().astype(np.uint8)).save(out_dir / name)
    wall = time.perf_counter() - t_start
    print(json.dumps({"image": str(out_dir / name), "mode": args.mode, "wall_seconds": round(wall, 1),
                      **extra}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
