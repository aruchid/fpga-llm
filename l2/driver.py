"""cocotb 侧驱动：扮演"电脑 + PCIe/XDMA"，通过 AXI 操作 mvm_top。

上层调用 `await drv.matmul(wq, SW, xq, s)`，接口与 l0.fixedpoint.int_matmul 相同。
"""
import logging

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotbext.axi import (AxiLiteBus, AxiLiteMaster, AxiStreamBus,
                           AxiStreamFrame, AxiStreamSink, AxiStreamSource)

from l2 import pack as pk


class MVMDriver:
    def __init__(self, dut, period_ns=4):
        self.dut = dut
        cocotb.start_soon(Clock(dut.aclk, period_ns, unit="ns").start())
        self.axil = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axil"),
                                  dut.aclk, dut.aresetn, reset_active_level=False)
        self.w_src = AxiStreamSource(AxiStreamBus.from_prefix(dut, "s_axis_w"),
                                     dut.aclk, dut.aresetn, reset_active_level=False)
        self.y_sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "m_axis_y"),
                                    dut.aclk, dut.aresetn, reset_active_level=False)
        self.last_cycles = 0
        self.reset_stats()
        for bus in ("s_axil", "s_axis_w", "m_axis_y"):      # 关掉逐笔事务日志
            logging.getLogger(f"{dut._log.name}.{bus}").setLevel(logging.WARNING)

    async def reset(self):
        self.dut.aresetn.value = 0
        for _ in range(5):
            await RisingEdge(self.dut.aclk)
        self.dut.aresetn.value = 1
        for _ in range(2):
            await RisingEdge(self.dut.aclk)

    async def read(self, addr):
        return await self.axil.read_dword(addr)

    async def write(self, addr, val):
        await self.axil.write_dword(addr, val)

    async def matmul(self, wq, SW, xq, s):
        return await self.run(pk.Packed(wq, SW), xq, s)

    async def run(self, p: pk.Packed, xq, s):
        """用预先打包的权重做一次矩阵乘，返回 Q16 int64 向量。"""
        words = pk.pack_x(xq, p.kb)
        for i, word in enumerate(words):
            await self.write(pk.X_BASE + 4 * i, word)
        await self.write(pk.REG_KB, p.kb)
        await self.write(pk.REG_NT, p.nt)
        await self.write(pk.REG_SHIFT, pk.shift_reg(s))
        await self.write(pk.REG_CTRL, 1)
        await self.w_src.send(AxiStreamFrame(p.stream))
        frame = await self.y_sink.recv()
        status = await self.read(pk.REG_STATUS)
        assert status & 0b11 == 0b10, f"STATUS={status:#x}，任务应已完成"
        self.last_cycles = await self.read(pk.REG_CYCLES)
        assert len(frame.tdata) == p.nt * pk.N * 4
        st = self.stats
        st["jobs"] += 1
        st["core_cycles"] += self.last_cycles
        st["axil_writes"] += len(words) + 4
        st["axil_reads"] += 2
        st["w_bytes"] += len(p.stream)
        st["y_bytes"] += len(frame.tdata)
        return pk.unpack_y(frame.tdata, p.M)

    def reset_stats(self):
        self.stats = dict(jobs=0, core_cycles=0, axil_writes=0, axil_reads=0,
                          w_bytes=0, y_bytes=0)
