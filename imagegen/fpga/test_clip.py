"""CLIP 文本编码器验证：分词 + 浮点/定点 context 对比。"""

import argparse
import time

import numpy as np

import fx
import mvm32
import sd_clip

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def cosine(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--layer", type=int, default=-2, help="-1=last, -2=penultimate（ComfyUI 默认）")
    args = ap.parse_args()

    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode(args.prompt)
    n_pad = sum(1 for t in ids if t == sd_clip.EOT)
    print(f"[clip] tokens={len(ids)} eot_count={n_pad} first={ids[:12]} last={ids[-4:]}")
    print(f"[clip] 回读: {tok.decoder[ids[1]]} ... {tok.decoder[ids[len(ids)-n_pad-1]]}")

    wt = sd_clip.ClipWeights(CKPT)
    t0 = time.perf_counter()
    ctx_f32 = sd_clip.text_encoder_float(wt, ids, layer_idx=args.layer)
    print(f"[clip] float {time.perf_counter()-t0:.1f}s  shape {ctx_f32.shape} "
          f"mean {ctx_f32.mean():+.3f} std {ctx_f32.std():.3f}")

    eng = mvm32.MVM32()
    t0 = time.perf_counter()
    ctx_fx = sd_clip.text_encoder_fx(wt, ids, eng, layer_idx=args.layer)
    dt = time.perf_counter() - t0
    ctx_fx_f = fx.from_fx(ctx_fx)
    print(f"[clip] fx    {dt:.1f}s  cos={cosine(ctx_f32, ctx_fx_f):.6f} "
          f"max|d|={np.max(np.abs(ctx_f32-ctx_fx_f)):.4f} std={ctx_fx_f.std():.3f}")
    st = eng.stats
    print(f"[clip] 引擎: MACs={st.macs/1e9:.2f}G cycles={st.cycles/1e6:.1f}M "
          f"({eng.cycles_to_seconds(st.cycles):.2f}s @115MHz) 饱和={st.saturations}")

    np.save("out/clip_ctx_float.npy", ctx_f32)
    np.save("out/clip_ctx_fx.npy", ctx_fx_f.astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
