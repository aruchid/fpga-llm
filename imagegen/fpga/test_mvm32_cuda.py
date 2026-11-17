"""CUDA 后端 vs numpy 后端的位精确回归 + 速度对比。"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "imagegen" / "fpga")]

import mvm32  # noqa: E402
import mvm32_cuda  # noqa: E402


def run_case(name, M, K, B, blocks=False, rng=None):
    rng = rng or np.random.default_rng(0)
    W = rng.normal(0, 0.02, size=(M, K)).astype(np.float32)
    wq, sw = mvm32.quantize_weight_rows(W)
    X = np.round(rng.normal(0, 5000, size=(K, B))).astype(np.int64)

    cpu = mvm32.MVM32()
    gpu = mvm32_cuda.MVM32CUDA()

    if blocks:
        q, sb = mvm32.quantize_blocks(X.T)          # (B, K) → 每行分块
        q = np.ascontiguousarray(q.T)
        sb = sb.T                                    # (nblk, B)
        t0 = time.perf_counter(); y_cpu = cpu.gemm_blocks(wq, sw, q, sb); t_cpu = time.perf_counter() - t0
        t0 = time.perf_counter(); y_gpu = gpu.gemm_blocks(wq, sw, q, sb); t_gpu = time.perf_counter() - t0
    else:
        from mvm32 import quantize_shift
        cols, ss = [], []
        for b in range(B):
            qb, s = quantize_shift(X[:, b])
            cols.append(qb); ss.append(s)
        q = np.stack(cols, axis=1)
        t0 = time.perf_counter(); y_cpu = cpu.gemm(wq, sw, q, np.array(ss)); t_cpu = time.perf_counter() - t0
        t0 = time.perf_counter(); y_gpu = gpu.gemm(wq, sw, q, np.array(ss)); t_gpu = time.perf_counter() - t0

    ok = np.array_equal(y_cpu, y_gpu)
    macs = M * K * B / 1e9
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<28} M={M:5d} K={K:5d} B={B:5d} "
          f"{macs:6.2f} GMAC  cpu {t_cpu*1e3:8.1f} ms ({macs/t_cpu:5.2f} G/s)  "
          f"cuda {t_gpu*1e3:7.1f} ms ({macs/t_gpu:6.1f} G/s)  加速 {t_cpu/max(t_gpu,1e-9):6.1f}×")
    return ok


def main() -> int:
    rng = np.random.default_rng(1)
    ok = True
    ok &= run_case("小 GEMM", 320, 1024, 77, rng=rng)
    ok &= run_case("VAE 卷积(128→128 @512²)", 128, 1152, 4096, rng=rng)
    ok &= run_case("UNet 卷积(1280 @64²)", 1280, 23040, 1024, rng=rng)
    ok &= run_case("CLIP MLP(1024→4096)", 4096, 1024, 77, blocks=True, rng=rng)
    ok &= run_case("分块量化 conv", 320, 2880, 4096, blocks=True, rng=rng)
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
