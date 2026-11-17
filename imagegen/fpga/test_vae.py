"""VAE 解码一致性/质量闸门。

- 浮点版 vs ComfyUI 的 float 解码（验证网络实现是否正确）
- 定点版（32 路引擎）vs 浮点版（量化损失到底多大）

用法:
    python test_vae.py --mode float          # 只跑浮点对照
    python test_vae.py --mode fx             # 只跑定点
    python test_vae.py --mode both
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

import sd_vae

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 0 else 10 * np.log10(1.0 / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["float", "fx", "both"])
    ap.add_argument("--latent", default="out/vae_latent.npy")
    ap.add_argument("--ref", default="out/vae_ref_float.png")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    z = np.load(args.latent)
    z = z[0] if z.ndim == 4 else z
    print(f"[vae] latent {z.shape}")

    ref = np.asarray(Image.open(args.ref).convert("RGB"), dtype=np.float32) / 255.0
    wt = sd_vae.Weights(CKPT)

    if args.mode in ("float", "both"):
        t0 = time.perf_counter()
        small = sd_vae.decoder_float(wt, z[:, :16, :16][None], 1.0)
        print(f"[vae] shape 自检 (16x16 latent -> {small.shape}) {time.perf_counter()-t0:.1f}s")

        for sf in (1.0, 0.18215):
            t0 = time.perf_counter()
            img = sd_vae.decoder_float(wt, z[None], sf)
            dt = time.perf_counter() - t0
            p = psnr(img, ref)
            Image.fromarray((img * 255).round().astype(np.uint8)).save(
                out_dir / f"vae_float_sf{sf:g}.png")
            print(f"[vae] float sf={sf:<7g} {dt:5.1f}s  PSNR vs ComfyUI {p:5.2f} dB")

    if args.mode in ("fx", "both"):
        import mvm32
        eng = mvm32.MVM32()
        t0 = time.perf_counter()
        img = sd_vae.decoder_fx(wt, z[None], eng, 1.0)
        dt = time.perf_counter() - t0
        Image.fromarray((img * 255).round().astype(np.uint8)).save(out_dir / "vae_fx.png")
        p_ref = psnr(img, ref)
        print(f"[vae] fx  {dt:5.1f}s  PSNR vs ComfyUI {p_ref:5.2f} dB")
        st = eng.stats
        print(f"[vae] 引擎: calls={st.calls} MACs={st.macs/1e9:.1f}G tiles={st.tiles} "
              f"cycles={st.cycles/1e9:.2f}G ({eng.cycles_to_seconds(st.cycles):.1f}s @{eng.fmax_mhz:g}MHz)"
              f" 饱和={st.saturations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
