"""向量单元 RTL 与 l0/fixedpoint.py 逐比特比对（随机 + 边界值）。由 l2/test_l2.py 启动。"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from l0 import fixedpoint as fp
from l2.vpu_driver import VPUDriver

Q = fp.ONE


async def setup(dut):
    import cocotb as _c
    _c.start_soon(Clock(dut.aclk, 4, unit="ns").start())
    vpu = VPUDriver(dut)
    dut.aresetn.value = 0
    for _ in range(5):
        await RisingEdge(dut.aclk)
    dut.aresetn.value = 1
    await RisingEdge(dut.aclk)
    return vpu


def same(name, got, want):
    got, want = np.asarray(got, np.int64), np.asarray(want, np.int64)
    bad = np.flatnonzero(got != want)
    assert got.shape == want.shape and bad.size == 0, (
        f"{name}: {bad.size} 个不一致，首个 i={bad[0] if bad.size else '-'} "
        f"rtl={got[bad[0]] if bad.size else '-'} ref={want[bad[0]] if bad.size else '-'}")


@cocotb.test()
async def quantize_shift(dut):
    vpu = await setup(dut)
    rng = np.random.default_rng(1)
    cases = [rng.integers(-Q, Q, 288), rng.integers(-(1 << 31), 1 << 31, 768),
             np.zeros(16, np.int64), np.array([127, -127, 128, 5]),
             rng.integers(-(1 << 33), 1 << 33, 64),          # RoPE 后可能超过 32 位
             np.array([1]), rng.integers(-300, 300, 48)]
    for i, x in enumerate(cases):
        q, s = await vpu.quant(x)
        qr, sr = fp.quantize_shift(x)
        same(f"quant#{i} q", q, qr)
        assert s == sr, f"quant#{i} s rtl={s} ref={sr}"


@cocotb.test()
async def rmsnorm(dut):
    vpu = await setup(dut)
    rng = np.random.default_rng(2)
    for n, scale in [(288, 1.0), (768, 30.0), (64, 0.001), (7, 300.0), (288, 0.0)]:
        x = fp.to_fx(rng.standard_normal(n) * scale)
        g = fp.to_fx(1 + 0.3 * rng.standard_normal(n))
        same(f"rmsnorm n={n} scale={scale}", await vpu.rmsnorm(x, g), fp.rmsnorm_fx(x, g))


@cocotb.test()
async def softmax(dut):
    vpu = await setup(dut)
    rng = np.random.default_rng(3)
    for x in [fp.to_fx(rng.standard_normal(40) * 3), fp.to_fx(rng.standard_normal(256) * 20),
              np.array([12345]), fp.to_fx(np.array([0.0, -50.0, -1000.0, 3.0]))]:
        same(f"softmax n={len(x)}", await vpu.softmax(x), fp.softmax_fx(x))


@cocotb.test()
async def elementwise(dut):
    vpu = await setup(dut)
    rng = np.random.default_rng(4)
    a = rng.integers(-(1 << 31), 1 << 31, 300)
    b = rng.integers(-(1 << 31), 1 << 31, 300)
    same("resadd（含饱和）", await vpu.ew(0, a, b, "resadd"), fp.resadd_fx(a, b))
    h1 = fp.to_fx(rng.uniform(-20, 20, 768))
    h3 = fp.to_fx(rng.standard_normal(768) * 5)
    same("silu_gate", await vpu.ew(1, h1, h3, "silu_gate"), fp.silu_gate_fx(h1, h3))
    big = rng.integers(-(1 << 31), 1 << 31, 64)
    same("silu_gate 大数", await vpu.ew(1, big, big, "silu_gate"), fp.silu_gate_fx(big, big))


@cocotb.test()
async def rope(dut):
    vpu = await setup(dut)
    rng = np.random.default_rng(5)
    for hs, n, pos in [(48, 288, 0), (48, 288, 200), (16, 32, 7), (64, 128, 1000)]:
        cos, sin = fp.rope_tables(pos + 1, hs)
        v = rng.integers(-(1 << 31), 1 << 31, n)
        same(f"rope hs={hs} pos={pos}", await vpu.rope(v, cos[pos], sin[pos], hs),
             fp.rope_fx(v, cos[pos], sin[pos], hs))


@cocotb.test()
async def attention_head(dut):
    vpu = await setup(dut)
    rng = np.random.default_rng(6)
    for hs, T in [(48, 1), (48, 37), (16, 5), (64, 100)]:
        qq = rng.integers(-127, 128, hs)
        Kq = rng.integers(-127, 128, (T, hs))
        Vq = rng.integers(-127, 128, (T, hs))
        qs = int(rng.integers(0, 20))
        ks = rng.integers(0, 20, T)
        vs = rng.integers(0, 20, T)
        inv = round(Q / np.sqrt(hs))
        got = await vpu.attn(qq, qs, Kq, ks, Vq, vs, inv)
        same(f"attn hs={hs} T={T}", got, fp.attention_head_fx(qq, qs, Kq, ks, Vq, vs, inv))
    for k, (n, cyc, el) in vpu.stats.items():
        dut._log.info(f"{k:10s} {n} 次，{cyc} 周期，{el} 元素")


@cocotb.test()
async def cycle_profile(dut):
    """各运算在不同尺寸下的周期（含数据进出），供 L1 拟合公式。"""
    vpu = await setup(dut)
    rng = np.random.default_rng(9)
    rows = []

    async def run(kind, n, coro):
        vpu.stats.clear()
        await coro
        rows.append((kind, n, vpu.stats[kind][1]))

    for n in (16, 64, 288, 768):
        x = rng.integers(-Q, Q, n)
        await run("quant", n, vpu.quant(x))
        await run("rmsnorm", n, vpu.rmsnorm(x, fp.to_fx(np.ones(n))))
        await run("resadd", n, vpu.ew(0, x, x, "resadd"))
        await run("silu_gate", n, vpu.ew(1, x, x, "silu_gate"))
        cos, sin = fp.rope_tables(2, 16)
        await run("rope", n, vpu.rope(x, cos[1], sin[1], 16))
    for T in (1, 8, 32, 100):
        hs = 48
        await run("attn", T, vpu.attn(rng.integers(-127, 128, hs), 5, rng.integers(-127, 128, (T, hs)),
                                      rng.integers(0, 9, T), rng.integers(-127, 128, (T, hs)),
                                      rng.integers(0, 9, T), round(Q / np.sqrt(hs))))
    for kind, n, cyc in rows:
        dut._log.info(f"PROFILE {kind} {n} {cyc}")
