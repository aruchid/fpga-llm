"""手工计时剖析 INT8 管线的耗时构成（不依赖 CUPTI）。

把 mm8（矩阵乘本体）与 q_rows（激活量化）包一层计时器，其余（im2col、反量化、
归一化、激活函数、kernel 启动）算作「其余」。
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"E:\FPGALLM\imagegen")
sys.path.insert(0, str(ROOT / "fpga"))

MODEL = str(ROOT / "models" / "sd-turbo-diffusers")
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"

ACC = {"gemm": 0.0, "quant": 0.0, "calls_gemm": 0, "calls_quant": 0}


def wrap(mod, name, key):
    orig = getattr(mod, name)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        ACC[key] += time.perf_counter() - t0
        ACC["calls_" + key] += 1
        return out

    setattr(mod, name, timed)


def main() -> int:
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(torch.float32)

    import quant_unet
    wrap(quant_unet, "mm8", "gemm")
    wrap(quant_unet, "q_rows", "quant")
    quant_unet.quantize_unet(pipe.unet)
    pipe.unet.to("cuda").float()

    ctx = pipe.encode_prompt(PROMPT, device="cuda", num_images_per_prompt=1,
                             do_classifier_free_guidance=False)[0].float()
    lat = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float32)
    t = torch.tensor(999.0, device="cuda")

    with torch.no_grad():
        pipe.unet(lat, t, encoder_hidden_states=ctx)      # 预热
    torch.cuda.synchronize()

    for k in ACC:
        ACC[k] = 0 if "calls" not in k else 0

    t0 = time.perf_counter()
    with torch.no_grad():
        pipe.unet(lat, t, encoder_hidden_states=ctx)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    gemm, quant = ACC["gemm"], ACC["quant"]
    rest = total - gemm - quant
    print(f"\n=== INT8 UNet 单步 ===  总 {total:.2f} s")
    print(f"  mm8 矩阵乘      {gemm:6.2f} s  {gemm/total*100:5.1f}%   ({ACC['calls_gemm']} 次)")
    print(f"  激活量化 q_rows  {quant:6.2f} s  {quant/total*100:5.1f}%   ({ACC['calls_quant']} 次)")
    print(f"  其余（im2col/反量化/归一化/调度） {rest:6.2f} s  {rest/total*100:5.1f}%")

    # 纯矩阵乘的理论下限：IMMA 实测约 8.6 TMAC/s，单步 339 GMAC
    print(f"\n  参考：单步 339.1 GMAC，按 IMMA 8.6 TMAC/s 理论需 {339.1/8600*1000:.0f} ms")
    print(f"  实测 mm8 吞吐 {339.1/gemm:.1f} GMAC/s（{8.6/(339.1/gemm):.0f}× 低于 IMMA 峰值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
