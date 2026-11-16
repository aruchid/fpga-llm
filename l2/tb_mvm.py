"""mvm_top 的 cocotb 测试：RTL 输出必须与 l0.fixedpoint.int_matmul 逐比特一致。

由 l2/test_l2.py 通过 cocotb runner 启动（不要直接用 pytest 收集本文件）。
环境变量 L2_FULL=1 时回放 stories15M 一个 token 的全部 43 次矩阵乘（含 32000 行的 wcls）。
"""
import os
import random
from pathlib import Path

import cocotb
import numpy as np

from l0 import fixedpoint as fp
from l2 import pack as pk
from l2.driver import MVMDriver

ROOT = Path(__file__).resolve().parents[1]


async def setup(dut):
    drv = MVMDriver(dut)
    await drv.reset()
    return drv


def random_case(rng, M, K, sw_max=1 << 20, s=None):
    wq = rng.integers(-127, 128, size=(M, K)).astype(np.int8)
    SW = rng.integers(1, sw_max, size=M).astype(np.int64)
    xq = rng.integers(-127, 128, size=K).astype(np.int64)
    if s is None:
        s = int(rng.integers(0, 26))
    return wq, SW, xq, s


async def check(drv, wq, SW, xq, s, name):
    want = fp.int_matmul(wq, SW, xq, s, tag="tb")
    got = await drv.matmul(wq, SW, xq, s)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (f"{name}: {bad.size} 个不一致，首个 i={bad[0]} "
                           f"rtl={got[bad[0]]} ref={want[bad[0]]}")
    M, K = wq.shape
    drv.dut._log.info(f"{name}: {M}x{K} s={s} 逐比特一致，"
                      f"{drv.last_cycles} 周期（理论 {pk.ideal_cycles(M, K)}）")


@cocotb.test()
async def id_register(dut):
    drv = await setup(dut)
    assert await drv.read(pk.REG_ID) == pk.MVM_ID
    assert await drv.read(pk.REG_STATUS) == 0


@cocotb.test()
async def random_shapes(dut):
    """各种尺寸（含非 32 倍数、K 最大 1024）和移位量。"""
    drv = await setup(dut)
    rng = np.random.default_rng(1)
    shapes = [(32, 32), (64, 96), (40, 50), (1, 7), (96, 288), (32, 1024)]
    for M, K in shapes:
        await check(drv, *random_case(rng, M, K), name=f"rand{M}x{K}")
    # 移位量边界：s=0（右移 24）、s=24（不移）、s=25（左移 1）
    for s in (0, 1, 24, 25):
        await check(drv, *random_case(rng, 64, 64, s=s), name=f"s={s}")


@cocotb.test()
async def saturation(dut):
    """全 ±127 + 大 SW，结果超出 int32，必须饱和而不是回绕。"""
    drv = await setup(dut)
    K = 1024
    wq = np.full((64, K), 127, dtype=np.int8)
    wq[32:] = -127
    SW = np.full(64, (1 << 31) - 1, dtype=np.int64)
    xq = np.full(K, 127, dtype=np.int64)
    await check(drv, wq, SW, xq, 25, "saturation")


@cocotb.test()
async def backpressure(dut):
    """权重流断续、结果流随机反压，结果不变。"""
    drv = await setup(dut)
    r = random.Random(7)
    drv.w_src.set_pause_generator(iter(lambda: r.random() < 0.3, None))
    drv.y_sink.set_pause_generator(iter(lambda: r.random() < 0.5, None))
    rng = np.random.default_rng(2)
    for M, K in [(64, 64), (96, 160)]:
        await check(drv, *random_case(rng, M, K), name=f"bp{M}x{K}")


@cocotb.test()
async def stories15m_replay(dut):
    """回放 L0 硬件等价模型在真实 stories15M 上的矩阵乘调用。"""
    model = ROOT / "models" / "stories15M.bin"
    if not model.exists():
        dut._log.warning("models/stories15M.bin 不存在，跳过")
        return
    from l0.ref_hw import HWModel
    from l0.weights import load_llama2c

    cfg, w = load_llama2c(str(model))
    hm = HWModel(cfg, w)
    calls = []
    orig = fp.int_matmul

    def spy(wq, SW, xq, s, tag="matmul"):
        y = orig(wq, SW, xq, s, tag)
        calls.append((tag, wq, SW, np.array(xq), s, y))
        return y

    fp.int_matmul = spy
    try:
        hm.forward(1, 0)      # BOS @ pos 0
        hm.forward(9038, 1)   # "Once" @ pos 1
    finally:
        fp.int_matmul = orig
    per_tok = len(calls) // 2
    sel = calls[per_tok:] if os.environ.get("L2_FULL") else calls[per_tok:per_tok + 7]

    drv = await setup(dut)
    total = ideal = 0
    for i, (tag, wq, SW, xq, s, want) in enumerate(sel):
        got = await drv.matmul(wq, SW, xq, s)
        bad = np.flatnonzero(got != want)
        assert bad.size == 0, f"#{i} {tag}: {bad.size} 个不一致"
        total += drv.last_cycles
        ideal += pk.ideal_cycles(*wq.shape)
        dut._log.info(f"#{i} {tag:5s} {wq.shape[0]}x{wq.shape[1]} s={s} "
                      f"逐比特一致，{drv.last_cycles} 周期")
    dut._log.info(f"共 {len(sel)} 次矩阵乘，计算核 {total} 周期（理论 {ideal}）")
