"""torch 版定点算子 vs numpy 版的位精确回归。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "imagegen" / "fpga")]

import fx  # noqa: E402
import fx_torch as ft  # noqa: E402
import mvm32  # noqa: E402
import sd_vae  # noqa: E402

rng = np.random.default_rng(0)
ok = True


def check(name, a, b):
    global ok
    a = np.asarray(a)
    b = b.cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
    same = np.array_equal(a, b)
    ok &= same
    print(f"[{'PASS' if same else 'FAIL'}] {name}")
    if not same:
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        print(f"       形状 {a.shape}/{b.shape} 最大差 {d.max() if d.size else 0} 不同元素 {int((d>0).sum())}")


def main() -> int:
    # 1) 舍入
    x = rng.integers(-10**9, 10**9, size=(64, 128)).astype(np.int64)
    s = rng.integers(-3, 30, size=(128,)).astype(np.int64)
    check("rshift_round", fx.rshift_round(x, s[None, :]), ft.rshift_round(ft.to_fx(0) * 0 + torch.from_numpy(x).cuda(), torch.from_numpy(s).cuda()))

    # 2) 逐行量化
    X = rng.integers(-3 * 2**19, 3 * 2**19, size=(77, 1024)).astype(np.int64)
    q1, s1 = sd_vae.quantize_shift_batch(X)
    q2, s2 = ft.quantize_shift_batch(torch.from_numpy(X).cuda())
    check("quantize_shift_batch q", q1, q2)
    check("quantize_shift_batch s", s1, s2)

    # 3) 分块量化
    q1, s1 = mvm32.quantize_blocks(X)
    q2, s2 = ft.quantize_blocks(torch.from_numpy(X).cuda())
    check("quantize_blocks q", q1, q2)
    check("quantize_blocks s", s1, s2)

    # 4) GroupNorm / LayerNorm
    xf = rng.standard_normal((256, 128)) * 1.3 + 0.4
    g = rng.uniform(0.5, 1.5, 128); b = rng.standard_normal(128) * 0.2
    xq = fx.to_fx(xf)
    check("groupnorm_fx",
          fx.groupnorm_fx(xq, fx.to_fx(g), fx.to_fx(b), 8),
          ft.groupnorm_fx(torch.from_numpy(xq).cuda(), ft.to_fx(g), ft.to_fx(b), 8))
    check("layernorm_fx",
          fx.layernorm_fx(xq, fx.to_fx(g), fx.to_fx(b)),
          ft.layernorm_fx(torch.from_numpy(xq).cuda(), ft.to_fx(g), ft.to_fx(b)))

    # 5) SiLU / softmax
    v = fx.to_fx(rng.standard_normal(2048) * 3)
    check("silu_fx", fx.silu_fx(v), ft.silu_fx(torch.from_numpy(v).cuda()))
    sc = fx.to_fx(rng.standard_normal((16, 64)) * 2)
    check("softmax_fx", fx.softmax_fx(sc, axis=1), ft.softmax_fx(torch.from_numpy(sc).cuda(), axis=1))

    # 6) 卷积
    H = W = 32; C = 64; O = 96
    x = fx.to_fx(rng.standard_normal((H, W, C)) * 0.7)
    Wt = rng.normal(0, 0.02, size=(O, C, 3, 3)).astype(np.float32)
    wq, sw = mvm32.quantize_weight_rows(sd_vae._w2d(Wt))
    bq = fx.to_fx(rng.standard_normal(O) * 0.1)
    ref = sd_vae.conv_fx(x, wq, sw, bq, mvm32.MVM32(), pad=1, chunk=4096)
    got = ft.conv_fx(torch.from_numpy(x).cuda(),
                     torch.from_numpy(wq).cuda(), torch.from_numpy(sw).cuda(),
                     torch.from_numpy(bq).cuda(), pad=1)
    check("conv_fx 3x3", ref, got)

    # 7) 线性层（分块量化）
    xl = fx.to_fx(rng.standard_normal((77, 1024)) * 1.1)
    Wl = rng.normal(0, 0.02, size=(4096, 1024)).astype(np.float32)
    wq, sw = mvm32.quantize_weight_rows(Wl)
    bq = fx.to_fx(rng.standard_normal(4096) * 0.05)
    ref = sd_vae.conv_fx(xl.reshape(1, 77, 1024), wq, sw, bq, mvm32.MVM32(), pad=0).reshape(77, 4096)
    # 注意 numpy 侧用的是逐行量化版 conv_fx，这里对不上；改用引擎的 blocks 直接比
    eng = mvm32.MVM32()
    q, sb = mvm32.quantize_blocks(xl)
    ref2 = eng.gemm_blocks(wq, sw, np.ascontiguousarray(q.T), sb.T).T + bq[None, :]
    got2 = ft.linear_fx(torch.from_numpy(xl).cuda(), torch.from_numpy(wq).cuda(),
                        torch.from_numpy(sw).cuda(), torch.from_numpy(bq).cuda(), blocks=True)
    check("linear_fx(分块)", ref2, got2)

    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
