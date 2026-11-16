"""cocotb 侧：驱动 npu_top 里的向量单元，执行 L0 运算流中的非线性运算。

exec(op) 的输入输出与 l0.ref_hw.exec_op 完全相同，可以直接替换。
"""
import logging

import numpy as np
from cocotb.utils import get_sim_time
from cocotbext.axi import AxiStreamBus, AxiStreamFrame, AxiStreamSink, AxiStreamSource

HSMAX = 64           # attn_head 的行宽（int8 个数），与 RTL 参数一致
PERIOD_NS = 4


def i64(a):
    return np.asarray(a, dtype=np.int64).astype("<i8").tobytes()


def from_i64(b):
    return np.frombuffer(bytes(b), dtype="<i8").astype(np.int64)


class VPUDriver:
    KINDS = {"rmsnorm", "quant", "rope", "attn", "silu_gate", "resadd"}

    def __init__(self, dut):
        self.dut = dut
        clk, rst = dut.aclk, dut.aresetn

        def src(p):
            logging.getLogger(f"{dut._log.name}.{p}").setLevel(logging.WARNING)
            return AxiStreamSource(AxiStreamBus.from_prefix(dut, p), clk, rst, reset_active_level=False)

        def snk(p):
            logging.getLogger(f"{dut._log.name}.{p}").setLevel(logging.WARNING)
            return AxiStreamSink(AxiStreamBus.from_prefix(dut, p), clk, rst, reset_active_level=False)

        self.q_a, self.q_y = src("q_a"), snk("q_y")
        self.rms_a, self.rms_b, self.rms_y = src("rms_a"), src("rms_b"), snk("rms_y")
        self.sm_a, self.sm_y = src("sm_a"), snk("sm_y")
        self.ew_a, self.ew_b, self.ew_y = src("ew_a"), src("ew_b"), snk("ew_y")
        self.rope_a, self.rope_b, self.rope_y = src("rope_a"), src("rope_b"), snk("rope_y")
        self.at_r, self.at_y = src("at_r"), snk("at_y")
        self.stats = {}         # kind → [次数, 周期, 元素数]

    def _account(self, kind, t0, n):
        st = self.stats.setdefault(kind, [0, 0, 0])
        st[0] += 1
        st[1] += round((get_sim_time("ns") - t0) / PERIOD_NS)
        st[2] += n

    # ------------------------------------------------------------ 各运算
    async def quant(self, x):
        t0 = get_sim_time("ns")
        await self.q_a.send(AxiStreamFrame(i64(x)))
        q = from_i64((await self.q_y.recv()).tdata)
        s = int(self.dut.q_s.value)
        self._account("quant", t0, len(x))
        return q, s

    async def rmsnorm(self, x, g):
        t0 = get_sim_time("ns")
        self.dut.rms_recip_n.value = round((1 << 24) / len(x))
        await self.rms_a.send(AxiStreamFrame(i64(x)))
        await self.rms_b.send(AxiStreamFrame(i64(g)))
        y = from_i64((await self.rms_y.recv()).tdata)
        self._account("rmsnorm", t0, len(x))
        return y

    async def softmax(self, x):
        t0 = get_sim_time("ns")
        await self.sm_a.send(AxiStreamFrame(i64(x)))
        y = from_i64((await self.sm_y.recv()).tdata)
        self._account("softmax", t0, len(x))
        return y

    async def ew(self, op, a, b, kind):
        t0 = get_sim_time("ns")
        self.dut.ew_op.value = op
        await self.ew_a.send(AxiStreamFrame(i64(a)))
        await self.ew_b.send(AxiStreamFrame(i64(b)))
        y = from_i64((await self.ew_y.recv()).tdata)
        self._account(kind, t0, len(a))
        return y

    async def rope(self, v, cos_row, sin_row, hs):
        t0 = get_sim_time("ns")
        n_pairs = len(v) // 2
        j = np.arange(n_pairs) % (hs // 2)
        cs = (np.asarray(sin_row)[j].astype(np.int64) << 32) | (np.asarray(cos_row)[j].astype(np.int64) & 0xFFFFFFFF)
        await self.rope_a.send(AxiStreamFrame(i64(v)))
        await self.rope_b.send(AxiStreamFrame(i64(cs)))
        y = from_i64((await self.rope_y.recv()).tdata)
        self._account("rope", t0, len(v))
        return y

    async def attn(self, qq, qs, Kq, ks, Vq, vs, inv_sqrt_hs):
        hs = len(qq)
        assert hs <= HSMAX
        t0 = get_sim_time("ns")
        self.dut.at_inv_sqrt_hs.value = int(inv_sqrt_hs)
        self.dut.at_hs.value = hs

        def rows(m, sh):
            r = np.zeros((len(m), HSMAX + 1), dtype=np.uint8)
            r[:, :hs] = np.asarray(m, dtype=np.int64).astype(np.int8).view(np.uint8)
            r[:, HSMAX] = np.asarray(sh, dtype=np.int64)
            return r.tobytes()

        await self.at_r.send(AxiStreamFrame(rows(np.asarray(qq)[None, :], [qs]) + rows(Kq, ks)))
        await self.at_r.send(AxiStreamFrame(rows(Vq, vs)))
        y = from_i64((await self.at_y.recv()).tdata)
        self._account("attn", t0, len(ks))
        return y

    # ------------------------------------------------------------ 运算流接口
    async def exec(self, op):
        k, a = op
        if k == "rmsnorm":
            return await self.rmsnorm(*a)
        if k == "quant":
            return await self.quant(a[0])
        if k == "rope":
            return await self.rope(*a)
        if k == "attn":
            return await self.attn(*a)
        if k == "silu_gate":
            return await self.ew(1, a[0], a[1], "silu_gate")
        if k == "resadd":
            return await self.ew(0, a[0], a[1], "resadd")
        raise ValueError(k)
