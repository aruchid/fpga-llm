"""L1 性能模型校验：运算清单与 L0 实际调用一致，周期公式与 RTL 实测一致。"""
import numpy as np
import pytest

from l0 import fixedpoint as fp
from l0.config import STORIES15M, TINY
from l0.ref_hw import HWModel, exec_op
from l0.weights import random_weights
from l1 import perf


def test_mvm_cycles_match_rtl():
    """RTL 仿真实测（l2/tb_mvm.py、tb_sched.py 的输出）。"""
    # 第一版 RTL
    assert perf.mvm_cycles(288, 288, perf.V1) == 2917
    assert perf.mvm_cycles(768, 288, perf.V1) == 7777
    assert perf.mvm_cycles(288, 768, perf.V1) == 7237
    assert perf.mvm_cycles(32000, 288, perf.V1) == 324001
    jobs = perf.weight_jobs(STORIES15M, fuse=True)
    assert len(jobs) == 25
    assert sum(perf.mvm_cycles(M, K, perf.V1) for _, M, K in jobs) == 530737
    assert len(perf.weight_jobs(STORIES15M, fuse=False)) == 43
    # 当前可综合版 RTL（双组累加器、6 级乘加流水线）
    assert perf.mvm_cycles(288, 288) == 2683
    assert perf.mvm_cycles(768, 288) == 7078
    assert perf.mvm_cycles(288, 768) == 7003
    assert sum(perf.mvm_cycles(M, K) for _, M, K in jobs) == 483564


@pytest.mark.parametrize("pos", [0, 5])
def test_inventory_matches_l0(pos, monkeypatch):
    """逐项统计 L0 硬件等价模型实际执行的运算，与 inventory 公式比对。"""
    cfg = TINY
    hm = HWModel(cfg, random_weights(cfg))
    for p in range(pos):
        hm.forward_fx(1, p)

    count = {"mm": 0, "qs": 0, "rms": 0, "sm": 0, "silu": 0, "rope": 0}

    def wrap(name, fn, size):
        def f(*a, **k):
            count[name] += size(*a)
            return fn(*a, **k)
        return f

    monkeypatch.setattr(fp, "quantize_shift", wrap("qs", fp.quantize_shift, lambda x, *r: len(x)))
    monkeypatch.setattr(fp, "rmsnorm_fx", wrap("rms", fp.rmsnorm_fx, lambda x, *r: len(x)))
    monkeypatch.setattr(fp, "softmax_fx", wrap("sm", fp.softmax_fx, lambda x, *r: len(x)))
    monkeypatch.setattr(fp, "silu_fx", wrap("silu", fp.silu_fx, lambda x, *r: len(x)))
    monkeypatch.setattr(fp, "rope_fx", wrap("rope", fp.rope_fx, lambda x, *r: len(x)))
    gen = hm.forward_steps(1, pos)
    res = None
    while True:
        try:
            op = gen.send(res)
        except StopIteration:
            break
        if op.kind == "matmul":
            count["mm"] += sum(r.lin.q.size for r in op.args)
        res = exec_op(op)

    inv = perf.inventory(cfg, pos + 1)
    assert count["mm"] == inv["w_macs"]
    assert count["qs"] == inv["vec"]["量化移位 quantize_shift"][0]
    assert count["rms"] == inv["vec"]["RMSNorm"][0]
    assert count["sm"] == inv["vec"]["Softmax"][0]
    assert count["silu"] == inv["vec"]["SiLU × 门控"][0]
    assert count["rope"] == inv["vec"]["RoPE"][0]


def test_weight_bytes_match_packed_stream():
    """读权重字节数 = 实际打包的权重流（无补零的模型）减去 SW 拍对齐。"""
    inv = perf.inventory(STORIES15M, 1)
    assert inv["weight_bytes"] == pytest.approx(15.39e6, rel=0.01)


def test_decode_sane():
    b = perf.BOARDS["KU040"]
    r = perf.decode(STORIES15M, b, "A1", 1)
    assert 0 < r["合计"] < 0.01
    assert perf.decode(STORIES15M, b, "A0-MMIO", 1)["合计"] > r["合计"]
    # 更多并行度不会更慢
    e = perf.Engine(lanes=128, post=0)
    assert perf.decode(STORIES15M, b, "A1", 1, e)["合计"] < r["合计"]


def avg_vpu(cfg, steps, version):
    out = {}
    for pos in range(steps):
        for k, v in perf.vpu_rtl_cycles(cfg, pos + 1, version).items():
            out[k] = out.get(k, 0) + v / steps
    return {k: round(v) for k, v in out.items()}


def test_vpu_cycles_match_rtl():
    """tb_sched 实测：stories15M，pos 0..31 共 32 步，每 token 平均周期（含数据进出）。"""
    assert avg_vpu(STORIES15M, 32, 1) == {"quant": 30794, "rmsnorm": 7605, "rope": 6924,
                                          "silu_gate": 4620, "resadd": 3480, "attn": 4464}
    v2 = avg_vpu(STORIES15M, 32, 2)
    for k, want in {"rmsnorm": 8398, "rope": 3564, "silu_gate": 4710, "resadd": 3660, "attn": 10080}.items():
        assert v2[k] == want, (k, v2[k], want)
    # quant 加深输出 FIFO 后在小模型上实测：每 token 25 次、2,473 周期
    assert perf.vpu_rtl_cycles(TINY, 1)["quant"] == 2473
