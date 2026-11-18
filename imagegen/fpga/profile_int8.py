"""剖析 INT8 管线为什么慢：算子级 profiling。

跑一次 UNet 前向（单步），分别统计：
- CUDA kernel 时间（真正在 GPU 上执行的算子）
- CPU 时间（Python 侧组织数据、启动 kernel 的开销）
并按算子类型聚合，与纯 IMMA 的耗时对比。

用法: python profile_int8.py [--fp16]
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

MODEL = r"E:\FPGALLM\imagegen\models\sd-turbo-diffusers"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def top_by_us(evts, key_fn, n=18):
    agg = defaultdict(lambda: [0, 0.0])          # [次数, 总耗时us]
    for e in evts:
        k = key_fn(e)
        agg[k][0] += 1
        agg[k][1] += float(getattr(e, "self_device_time_total", 0) or 0) if hasattr(e, "self_device_time_total") else float(getattr(e, "self_cuda_time_total", 0) or 0)
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])[:n]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true", help="剖析 fp16 基线做对照")
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()

    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(torch.float32)

    if not args.fp16:
        import quant_unet
        quant_unet.quantize_unet(pipe.unet)
        pipe.unet.to("cuda").float()

    ctx = pipe.encode_prompt(PROMPT, device="cuda", num_images_per_prompt=1,
                             do_classifier_free_guidance=False)[0]
    ctx = ctx.float()
    lat = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float32)
    t = torch.tensor(999.0, device="cuda")

    # 预热一次（触发权重搬运 / 分配器 warmup）
    with torch.no_grad():
        pipe.unet(lat, t, encoder_hidden_states=ctx)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        with torch.no_grad():
            for _ in range(args.steps):
                pipe.unet(lat, t, encoder_hidden_states=ctx)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    evts = prof.key_averages()
    dev_key = "self_device_time_total" if hasattr(evts[0], "self_device_time_total") else "self_cuda_time_total"

    total_dev = sum(float(getattr(e, dev_key, 0) or 0) for e in evts) / 1e6     # s
    total_cpu = sum(float(getattr(e, "self_cpu_time_total", 0) or 0) for e in evts) / 1e6

    print(f"\n=== {'FP16 基线' if args.fp16 else 'INT8'} · {args.steps} 步 UNet ===")
    print(f"墙钟 {wall:.2f} s | CUDA kernel 合计 {total_dev:.2f} s | CPU(算子自身) 合计 {total_cpu:.2f} s")

    print(f"\n{'算子':<44}{'调用':>8}{'CUDA(ms)':>12}{'占比':>8}")
    rows = sorted(evts, key=lambda e: -float(getattr(e, dev_key, 0) or 0))[:18]
    for e in rows:
        dev_ms = float(getattr(e, dev_key, 0) or 0) / 1e3
        if dev_ms < 1:
            continue
        print(f"{e.key[:44]:<44}{e.count:>8}{dev_ms:>12.1f}{dev_ms/1000/total_dev*100 if total_dev else 0:>7.1f}%")

    # 纯 IMMA 时间
    imm = sum(float(getattr(e, dev_key, 0) or 0) for e in evts if "_int_mm" in e.key) / 1e6
    if imm:
        print(f"\n纯 _int_mm（矩阵乘本体）：{imm:.2f} s，占墙钟 {imm/wall*100:.1f}%")

    # CPU 侧 Top
    print(f"\nCPU 侧耗时 Top 10（Python / 调度开销）")
    for e in sorted(evts, key=lambda e: -float(getattr(e, "self_cpu_time_total", 0) or 0))[:10]:
        cpu_ms = float(getattr(e, "self_cpu_time_total", 0) or 0) / 1e3
        if cpu_ms < 50:
            continue
        print(f"  {e.key[:50]:<50}{cpu_ms:>10.1f} ms  ({e.count} 次)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
