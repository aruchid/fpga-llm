"""热态下的一次完整生成：算子级耗时分解（pad / im2col / 量化 / GEMM / 反量化）。"""

import sys
import time

import torch

sys.path.insert(0, r"E:\FPGALLM\imagegen\fpga")

MODEL = r"E:\FPGALLM\imagegen\models\sd-turbo-diffusers"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"
KEYS = ("pad", "im2col", "quant", "gemm", "dequant")


def main() -> int:
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(torch.float32)

    import quant_unet
    quant_unet.quantize_unet(pipe.unet)
    quant_unet.quantize_unet(pipe.vae.decoder)
    import mvm_tiled
    quant_unet.TILE[0] = mvm_tiled.MVM32Tiled()
    torch.cuda.empty_cache()

    g = lambda: torch.Generator("cuda").manual_seed(42)
    for _ in range(2):                       # 预热
        pipe(PROMPT, num_inference_steps=4, guidance_scale=0.0, generator=g())
    torch.cuda.synchronize()

    for k in KEYS:
        quant_unet.PROFILE[k] = 0.0
    quant_unet.PROFILE["on"] = True

    t0 = time.perf_counter()
    pipe(PROMPT, num_inference_steps=4, guidance_scale=0.0, generator=g())
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    quant_unet.PROFILE["on"] = False

    P = quant_unet.PROFILE
    print(f"\n=== 完整生成（热态）总 {total:.2f} s ===")
    for k in KEYS:
        print(f"  {k:<9} {P[k]:5.2f} s  {P[k] / total * 100:5.1f}%")
    other = total - sum(P[k] for k in KEYS)
    print(f"  {'其他':<9} {other:5.2f} s  {other / total * 100:5.1f}%  (GroupNorm/SiLU/注意力/调度)")

    st = quant_unet.TILE[0].stats
    print(f"\n  矩阵乘本体吞吐：{st.macs/1e9/max(P['gemm'],1e-9):.0f} GMAC/s")
    print(f"  tile {st.tiles/1e6:.1f}M / 周期 {st.cycles/1e9:.2f}G / 等效 {quant_unet.TILE[0].hw_seconds():.0f} s @115MHz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
