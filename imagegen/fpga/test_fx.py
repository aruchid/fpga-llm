"""定点原语向量版 vs l0 标量版的一致性测试。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "imagegen" / "fpga")]

import fx  # noqa: E402
import l0.fixedpoint as fp  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(1)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok &= bool(cond)

    # 1) 1/√x：覆盖多个数量级
    ms = sorted(set(int(v) for v in rng.integers(1, 1 << 22, size=400)) |
                {1, 2, 3, (1 << 16), (1 << 20), (1 << 21) - 1})
    ms = np.array([m for m in ms if m > 0], dtype=np.int64)
    ref = np.array([fp.rsqrt_fx(int(m)) for m in ms], dtype=np.int64)
    got = fx.rsqrt_fx_v(ms)
    bad = np.nonzero(ref != got)[0]
    check("rsqrt 逐比特", bad.size == 0, f"n={ms.size} mism={bad.size}"
          + (f" first={ms[bad[0]]}: ref={ref[bad[0]]} got={got[bad[0]]}" if bad.size else ""))

    # 2) 1/x
    ref = np.array([fp.recip_fx(int(m)) for m in ms], dtype=np.int64)
    got = fx.recip_fx_v(ms)
    bad = np.nonzero(ref != got)[0]
    check("recip 逐比特", bad.size == 0, f"mism={bad.size}")

    # 3) softmax / sigmoid / silu（l0 本身向量化，只验证一致性）
    #    注意 l0.softmax_fx 是对整个数组取最大值，这里用 1D 输入对齐它的口径；
    #    fx.softmax_fx 支持按 axis 计算，行内 softmax 见 attention 的用法。
    s = np.round(rng.normal(0, 2.0, size=9) * 65536).astype(np.int64)
    check("softmax", np.array_equal(fp.softmax_fx(s), fx.softmax_fx(s)))
    x = np.round(rng.normal(0, 3.0, size=500) * 65536).astype(np.int64)
    check("sigmoid", np.array_equal(fp.sigmoid_fx(x), fx.sigmoid_fx(x)))
    check("silu", np.array_equal(fp.silu_fx(x), fx.silu_fx(x)))

    # 4) GroupNorm 与浮点参考的误差（统计跨 N 个位置 × 组内通道）
    n, c, groups = 64, 32, 8
    xf = rng.normal(0.7, 1.2, size=(n, c))          # 带直流偏移，检验均值精度
    gamma = rng.uniform(0.5, 1.5, size=c)
    beta = rng.normal(0, 0.2, size=c)
    xq = fx.to_fx(xf)
    yq = fx.groupnorm_fx(xq, fx.to_fx(gamma), fx.to_fx(beta), groups)
    xg = xf.reshape(n, groups, c // groups)
    mu = xg.mean(axis=(0, 2), keepdims=True)
    var = xg.var(axis=(0, 2), keepdims=True)
    yf = ((xg - mu) / np.sqrt(var + 1e-6)).reshape(n, c) * gamma + beta
    rel = np.max(np.abs(fx.from_fx(yq) - yf)) / np.max(np.abs(yf))
    check("groupnorm 相对误差 < 1e-3", rel < 1e-3, f"max_rel={rel:.2e}")

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
