"""32 路矩阵引擎的 CUDA 后端：语义与 mvm32.MVM32 完全相同，只是把 GEMM 丢给 GPU。

对应关系：
    int8 × int8 → int32 累加   ←→  torch._int_mm（IMMA 指令，累加器就是 int32）
    行缩放 SW、移位、舍入、饱和  ←→  int64 元素级运算（与 CPU 版同公式）

用它可以几分钟内验证整条链，而不是几十分钟；验证通过后再逐层换回 RTL/Verilator。
位精确性由 test_mvm32_cuda.py 对 numpy 后端做回归。
"""

from __future__ import annotations

import math

import numpy as np
import torch

import mvm32
from mvm32 import INT32_MAX, INT32_MIN, SW_FRAC, rshift_round, sat32


def _pad8(x: np.ndarray, axis: int) -> np.ndarray:
    n = x.shape[axis]
    pad = (-n) % 8
    if not pad:
        return x
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, pad)
    return np.pad(x, widths)


class MVM32CUDA:
    """接口与 mvm32.MVM32 一致（gemm / gemm_blocks / dot_i8 / stats / 周期模型）。"""

    def __init__(self, fmax_mhz: float = 115.0, device: str = "cuda"):
        assert torch.cuda.is_available(), "需要 CUDA"
        self.device = device
        self.ref = mvm32.MVM32(fmax_mhz=fmax_mhz)     # 只用来借统计与周期模型
        self.stats = self.ref.stats

    # ---------------------------------------------------------------- 基础
    def _mm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a (M,K) int8, b (K,N) int8 → int32 (M,N)，与 CPU int32 累加逐比特一致。"""
        bp = _pad8(b, 1)
        ta = torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True)
        tb = torch.from_numpy(np.ascontiguousarray(bp)).to(self.device, non_blocking=True)
        out = torch._int_mm(ta, tb)[:, :b.shape[1]]
        return out.cpu().numpy()

    def _scale_round(self, acc: np.ndarray, SW: np.ndarray, s: np.ndarray) -> np.ndarray:
        """与 gemm 的收尾一致：sat32(round(acc*SW >> (24-s)))，在 GPU 上做 int64。"""
        M, B = acc.shape
        t_acc = torch.from_numpy(acc.astype(np.int64)).to(self.device)
        t_sw = torch.from_numpy(np.asarray(SW, dtype=np.int64).reshape(M, 1)).to(self.device)
        t_s = torch.from_numpy(np.asarray(s, dtype=np.int64).reshape(1, B)).to(self.device)
        prod = t_acc * t_sw
        shift = SW_FRAC - t_s
        pos = shift.clamp(min=0)
        half = torch.where(pos > 0, torch.ones_like(prod) << (pos.clamp(min=1) - 1), torch.zeros_like(prod))
        # 有符号 int64 的 >> 是算术右移，与 rshift_round 一致
        val = torch.where(shift >= 0, (prod + half) >> pos, prod << (-shift))
        val = val.clamp(INT32_MIN, INT32_MAX)
        return val.cpu().numpy().astype(np.int32)

    # ---------------------------------------------------------------- 矩阵乘
    def gemm(self, Wq, SW, Xq, s, tile_scan: bool = False):
        """与 mvm32.MVM32.gemm 同语义（单尺度版）。"""
        Wq = np.ascontiguousarray(Wq)
        Xq = np.ascontiguousarray(Xq)
        M, K = Wq.shape
        B = Xq.shape[1]
        assert K == Xq.shape[0]
        acc = self._mm(Wq, Xq)
        acc = mvm32.wrap_int32(acc.astype(np.int64))
        out = self._scale_round(acc, SW, np.broadcast_to(np.asarray(s, dtype=np.int64), (B,)))
        st = self.stats
        st.calls += 1
        st.macs += M * K * B
        st.tiles += math.ceil(M / 32) * math.ceil(K / 32) * math.ceil(B / 32)
        st.cycles += self.ref.tile_cycles(M, K, B)
        return out

    def gemm_blocks(self, Wq, SW, Xq, sb, blk=32):
        """与 mvm32.MVM32.gemm_blocks 同语义：每 32 个输入一个移位量。

        实现上按「相同移位量」分组，每组一次 _int_mm，避免 144 次小 GEMM 的启动开销。
        """
        Wq = np.ascontiguousarray(Wq)
        Xq = np.ascontiguousarray(Xq)
        M, K = Wq.shape
        B = Xq.shape[1]
        nblk = math.ceil(K / blk)
        sb = np.asarray(sb, dtype=np.int64)
        sref = sb.min(axis=0)                                    # (B,)
        acc = np.zeros((M, B), dtype=np.int64)
        # 每个块的移位量 → 同一 Shift 的块合并成一次 GEMM
        shifts = sb - sref[None, :]                              # (nblk, B)
        for blk_i in range(nblk):
            k0, k1 = blk_i * blk, min((blk_i + 1) * blk, K)
            part = self._mm(Wq[:, k0:k1], Xq[k0:k1, :]).astype(np.int64)
            sh = shifts[blk_i]
            if sh.max() == 0:
                acc += part
            else:
                acc += part << sh[None, :]
        out = self._scale_round(acc, SW, sref)
        st = self.stats
        st.calls += 1
        st.macs += M * K * B
        st.tiles += math.ceil(M / 32) * nblk * math.ceil(B / 32)
        st.cycles += self.ref.tile_cycles(M, K, B)
        return out

    def dot_i8(self, Wq, Xq):
        Wq = np.ascontiguousarray(Wq)
        Xq = np.ascontiguousarray(Xq)
        M, K = Wq.shape
        acc = mvm32.wrap_int32(self._mm(Wq, Xq).astype(np.int64))
        st = self.stats
        st.calls += 1
        st.macs += M * K * Xq.shape[1]
        st.tiles += math.ceil(M / 32) * math.ceil(K / 32) * math.ceil(Xq.shape[1] / 32)
        st.cycles += self.ref.tile_cycles(M, K, Xq.shape[1])
        return acc.astype(np.int32)


class HybridEngine:
    """按规模调度：大 GEMM 走 GPU，小算子留 CPU（GPU 启动开销对小矩阵是灾难）。

    两个后端都是位精确的，所以换后端不改变任何数值结果——可以直接用来做回归。
    """

    def __init__(self, fmax_mhz: float = 115.0, min_macs: int = 8_000_000):
        self.cpu = mvm32.MVM32(fmax_mhz=fmax_mhz)
        self.gpu = MVM32CUDA(fmax_mhz=fmax_mhz)
        self.min_macs = min_macs
        self.stats = mvm32.Stats()
        self.gpu_calls = 0
        self.cpu_calls = 0

    def _pick(self, macs: int):
        if macs >= self.min_macs:
            self.gpu_calls += 1
            return self.gpu
        self.cpu_calls += 1
        return self.cpu

    def _merge(self, macs, cycles, tiles):
        st = self.stats
        st.calls += 1
        st.macs += macs
        st.cycles += cycles
        st.tiles += tiles

    def gemm(self, Wq, SW, Xq, s, tile_scan=False):
        M, K = Wq.shape
        B = Xq.shape[1]
        eng = self._pick(M * K * B)
        out = eng.gemm(Wq, SW, Xq, s, tile_scan=tile_scan)
        self._merge(M * K * B, self.cpu.tile_cycles(M, K, B), 0)
        return out

    def gemm_blocks(self, Wq, SW, Xq, sb, blk=32):
        M, K = Wq.shape
        B = Xq.shape[1]
        eng = self._pick(M * K * B)
        out = eng.gemm_blocks(Wq, SW, Xq, sb, blk=blk)
        self._merge(M * K * B, self.cpu.tile_cycles(M, K, B), 0)
        return out

    def dot_i8(self, Wq, Xq):
        M, K = Wq.shape
        B = Xq.shape[1]
        eng = self._pick(M * K * B)
        out = eng.dot_i8(Wq, Xq)
        self._merge(M * K * B, self.cpu.tile_cycles(M, K, B), 0)
        return out

    def cycles_to_seconds(self, cycles):
        return self.cpu.cycles_to_seconds(cycles)