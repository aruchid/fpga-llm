"""VAE 解码器（INT8）内部耗时分解：pad / im2col / 量化 / GEMM / 反量化。"""

import sys
import time
import argparse
from pathlib import Path

import torch

sys.path.insert(0, r"E:\FPGALLM\imagegen\fpga")

MODEL = r"E:\FPGALLM\imagegen\models\sd-turbo-diffusers"
KEYS = ("pad", "im2col", "quant", "gemm", "dequant")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", action="store_true")
    args = ap.parse_args()

    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(torch.float32)

    import quant_unet
    if args.tile:
        import mvm_tiled
        quant_unet.TILE[0] = mvm_tiled.MVM32Tiled()
    quant_unet.quantize_unet(pipe.vae.decoder)
    torch.cuda.empty_cache()

    lat = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float32)
    with torch.no_grad():
        pipe.vae.decode(lat / 0.18215)          # 预热
    torch.cuda.synchronize()

    for k in KEYS:
        quant_unet.PROFILE[k] = 0.0
    quant_unet.PROFILE["on"] = True

    t0 = time.perf_counter()
    with torch.no_grad():
        pipe.vae.decode(lat / 0.18215)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    quant_unet.PROFILE["on"] = False

    P = quant_unet.PROFILE
    print(f"\n=== VAE 解码（INT8, 512×512）总 {total:.2f} s ===")
    for k in KEYS:
        print(f"  {k:<9} {P[k]:6.2f} s  {P[k] / total * 100:5.1f}%")
    other = total - sum(P[k] for k in KEYS)
    print(f"  {'其他':<9} {other:6.2f} s  {other / total * 100:5.1f}%  (GroupNorm/SiLU/注意力/调度)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
