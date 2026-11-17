"""把采样好的 latent 解码成图（独立进程，避免和 UNet 抢内存）。

用法: python decode.py --latent out/latent_fx_v1_seed42.npy --mode fx
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

import fx
import mvm32
import sd_vae

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--mode", default="fx", choices=["fx", "float", "torch"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    z = np.load(args.latent)
    print(f"[decode] latent {z.shape} range [{z.min():.2f},{z.max():.2f}]", flush=True)

    wt = sd_vae.Weights(CKPT)
    t0 = time.perf_counter()
    if args.mode == "torch":
        import sd_vae_t
        img = sd_vae_t.decoder(sd_vae_t.TWeights(wt), z, 1.0)
        print(f"[decode] torch {time.perf_counter()-t0:.1f}s", flush=True)
    elif args.mode == "fx":
        eng = mvm32.MVM32()
        img = sd_vae.decoder_fx(wt, z[None], eng, 1.0)
        st = eng.stats
        print(f"[decode] fx {time.perf_counter()-t0:.1f}s  MACs {st.macs/1e9:.0f}G  "
              f"硬件估算 {eng.cycles_to_seconds(st.cycles):.0f}s @{eng.fmax_mhz:g}MHz", flush=True)
    else:
        img = sd_vae.decoder_float(wt, z[None], 1.0)
        print(f"[decode] float {time.perf_counter()-t0:.1f}s", flush=True)

    name = Path(args.latent).stem.replace("latent_", f"img_{args.mode}{args.tag}_")
    path = out_dir / (name + ".png")
    Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
    print("[decode] ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
