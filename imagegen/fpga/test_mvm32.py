"""验证 mvm32 与仓库 l0/fixedpoint.py 的 int_matmul 逐比特一致。

跑法（工作目录 E:\\FPGALLM）:
    python imagegen/fpga/test_mvm32.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "imagegen" / "fpga"))

import l0.fixedpoint as fp  # noqa: E402
import mvm32  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True

    # 1) 量化函数与仓库一致
    w = rng.normal(0, 0.02, size=(96, 64))
    q_ref, sw_ref = fp.quantize_weight_rows(w)
    q_my, sw_my = mvm32.quantize_weight_rows(w)
    ok &= check("quantize_weight_rows", np.array_equal(q_ref, q_my) and np.array_equal(sw_ref, sw_my))

    # 2) int_matmul 逐比特一致（含真实量级）
    x = np.round(rng.normal(0, 2.5, size=64) * 65536).astype(np.int64)
    xq_ref, s_ref = fp.quantize_shift(x)
    xq_my, s_my = mvm32.quantize_shift(x)
    ok &= check("quantize_shift", np.array_equal(xq_ref, xq_my) and s_ref == s_my,
                f"s={s_ref}")

    y_ref = fp.int_matmul(q_ref, sw_ref, xq_ref, s_ref)
    eng = mvm32.MVM32()
    y = eng.gemm(q_my, sw_my, xq_my.reshape(-1, 1), np.array([s_my]))[:, 0]
    ok &= check("int_matmul bit-exact", np.array_equal(y_ref, y),
                f"max|Δ|={int(np.max(np.abs(y_ref.astype(np.int64) - y.astype(np.int64))))}")

    # 3) 分块累加与整体累加等价（tile_scan）
    y_tiled = eng.gemm(q_my, sw_my, xq_my.reshape(-1, 1), np.array([s_my]), tile_scan=True)[:, 0]
    ok &= check("tile 累加等价", np.array_equal(y_tiled, y))

    # 4) 批量多列 + 每列不同 s
    B = 5
    X = np.round(rng.normal(0, 3.0, size=(64, B)) * 65536).astype(np.int64)
    cols, ss = [], []
    for b in range(B):
        qb, sb = fp.quantize_shift(X[:, b])
        cols.append(qb); ss.append(sb)
    Xq = np.stack(cols, axis=1); S = np.array(ss)
    ref = np.stack([fp.int_matmul(q_ref, sw_ref, Xq[:, b], S[b]) for b in range(B)], axis=1)
    my = eng.gemm(q_my, sw_my, Xq, S)
    ok &= check("批量列 bit-exact", np.array_equal(ref, my), f"shapes {ref.shape}")

    # 5) s=0 与负移位（左移）路径
    y2 = eng.gemm(q_my, sw_my, Xq, np.zeros(B, dtype=np.int64))
    ref2 = np.stack([fp.int_matmul(q_ref, sw_ref, Xq[:, b], 0) for b in range(B)], axis=1)
    ok &= check("s=0 bit-exact", np.array_equal(ref2, y2))

    # 6) 饱和：把 SW 放大到溢出
    sw_big = sw_my * (1 << 16)
    y_sat = eng.gemm(q_my, sw_big, Xq, S)
    ok &= check("饱和触发", bool(np.any(y_sat == mvm32.INT32_MAX) or np.any(y_sat == mvm32.INT32_MIN)))

    # 7) 周期模型：M=320,K=1280 → nt=10, kb=40
    cyc = eng.tile_cycles(320, 1280)
    expect = 10 * (40 * 32 + 4)
    ok &= check("周期公式", cyc == expect, f"{cyc} vs {expect}, {eng.cycles_to_seconds(cyc)*1e3:.2f} ms @115MHz")

    st = eng.stats
    print(f"\n统计: calls={st.calls} MACs={st.macs:,} tiles={st.tiles} "
          f"cycles={st.cycles:,} 饱和={st.saturations}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
