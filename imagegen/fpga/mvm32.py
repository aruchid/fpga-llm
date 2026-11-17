"""32 路矩阵单元的行为级模拟器。

对标 fpga-llm 的 rtl/mvm_core.sv（N=32）与 l0/fixedpoint.py 的 int_matmul：

    y = sat32(round((W · x) · SW >> (24 - s)))

- W：int8 对称量化权重，每个输出行一个缩放系数 SW（Q24）
- x：int8 + 每个输入向量一个移位量 s，x_real ≈ x_q · 2^(s-16)
- 累加：int32（与 RTL 一致，按 32 位环绕）
- 舍入：加半个 LSB 再算术右移，与 rshift_round 一致
- 输出：Q16.16，饱和到 int32

本模块只做**功能级**模拟（bit-exact），并通过 tile 扫描复现 RTL 的分块累加与
周期数；它不是 Verilator 仿真，时序数字来自对 mvm_core.sv 状态机的解析估算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

N = 32                 # 矩阵单元的通道数（RTL parameter N）
SW_FRAC = 24           # 行缩放系数的小数位
FRAC = 16              # 激活值 Q16.16
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


# ---------------------------------------------------------------- 定点原语
def rshift_round(x, s):
    """x · 2^-s 并舍入（与 l0/fixedpoint.rshift_round 一致，支持 s<0 左移）。"""
    x = np.asarray(x, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    pos = np.minimum(np.maximum(s, 0), 62)
    neg = np.maximum(-s, 0)
    half = np.where(pos > 0, np.left_shift(np.int64(1), np.maximum(pos - 1, 0)), 0)
    return np.where(s >= 0, np.right_shift(x + half, pos), np.left_shift(x, neg))


def sat32(x):
    return np.clip(np.asarray(x, dtype=np.int64), INT32_MIN, INT32_MAX)


def wrap_int32(x):
    """RTL 累加器是 32 位，溢出按补码环绕。"""
    x = np.asarray(x, dtype=np.int64)
    return ((x + (1 << 31)) & 0xFFFF_FFFF) - (1 << 31)


def quantize_weight_rows(w, frac=SW_FRAC, chunk_rows=512):
    """浮点权重 (out, in) → int8 + 每行 SW（Q24）。离线完成，等价于硬件 ROM 初始化。

    分块处理：大卷积（1280×23040）一次性转 float64 会要 225MB，分块后峰值只有几十 MB。
    """
    w = np.asarray(w, dtype=np.float32)
    n = w.shape[0]
    q = np.empty(w.shape, dtype=np.int8)
    SW = np.empty(n, dtype=np.int64)
    for r0 in range(0, n, chunk_rows):
        r1 = min(r0 + chunk_rows, n)
        blk = w[r0:r1].astype(np.float32)
        sw = np.max(np.abs(blk), axis=1).astype(np.float32) / 127.0
        sw = np.where(sw == 0, np.float32(1.0 / (1 << frac)), sw)
        SW[r0:r1] = np.round(sw * (1 << frac)).astype(np.int64)
        scale = (SW[r0:r1].astype(np.float32) / float(1 << frac))[:, None]
        q[r0:r1] = np.clip(np.round(blk / scale), -127, 127).astype(np.int8)
        del blk
    return q, SW


def quantize_shift(x, qmax=127):
    """Q16 向量 → int8 + 移位量 s。s 由最大绝对值的位宽决定，硬件里不需要除法。"""
    x = np.asarray(x, dtype=np.int64)
    m = int(np.max(np.abs(x))) if x.size else 0
    s = max(0, m.bit_length() - 7)
    if int(rshift_round(m, s)) > qmax:
        s += 1
    q = np.clip(rshift_round(x, s), -qmax, qmax).astype(np.int8)
    return q, s


# ---------------------------------------------------------------- 引擎
@dataclass
class Stats:
    calls: int = 0
    macs: int = 0          # 乘累加次数
    tiles: int = 0         # 32×32 输出 tile 数
    cycles: int = 0        # 估算周期（mvm_core 输入/流水线口径）
    saturations: int = 0
    max_acc_bits: int = 0


@dataclass
class MVM32:
    """一个 32 路矩阵单元。函数级 bit-exact，周期数按 mvm_core.sv 估算。"""

    lanes: int = N
    fmax_mhz: float = 115.0     # 与 docs/07 的粗估一致
    stats: Stats = field(default_factory=Stats)

    # -------- 周期估算
    def tile_cycles(self, M: int, K: int, B: int = 1) -> int:
        """按 mvm_core.sv 的输入口径：每拍一行权重（32 个 int8）。

        每个输出 tile 需要 kb_num 个 K 块 × N 行权重，外加 N/8 拍 SW；
        乘加流水线满速，不额外停顿。
        """
        nt = math.ceil(M / self.lanes)
        kb = math.ceil(K / self.lanes)
        sw_pkts = self.lanes // 8
        return B * nt * (kb * self.lanes + sw_pkts)

    def cycles_to_seconds(self, cycles: int) -> float:
        return cycles / (self.fmax_mhz * 1e6)

    # -------- 计算
    def gemm(self, Wq, SW, Xq, s, tile_scan: bool = False):
        """一次矩阵单元的整批运算（多行权重 × 多列输入）。

        Wq : (M, K) int8
        SW : (M,)   int64，Q24
        Xq : (K, B) int8
        s  : (B,)   int64，每列一个移位量
        返回 (M, B) int32，Q16.16
        """
        Wq = np.ascontiguousarray(Wq)
        Xq = np.ascontiguousarray(Xq)
        M, K = Wq.shape
        K2, B = Xq.shape
        assert K == K2, f"K 不匹配: {K} vs {K2}"
        s = np.broadcast_to(np.asarray(s, dtype=np.int64), (B,))

        if tile_scan:
            acc = self._accumulate_tiled(Wq, Xq)
        else:
            # 分块累加是精确整数加法，结果与分块方式无关。
            # numpy 的 int32 矩阵乘是慢速循环，这里改用 float64 BLAS：
            # 只要 K·127² < 2^53，整数在 float64 中可精确表示，且求和顺序不影响
            # 整数结果的精确性，所以结果与 RTL 的 int32 累加逐比特一致。
            assert K * 127 * 127 < (1 << 53), "K 太大，float64 累加不再精确"
            acc = np.rint(Wq.astype(np.float64) @ Xq.astype(np.float64)).astype(np.int64)

        acc = wrap_int32(acc.astype(np.int64))

        sw = np.asarray(SW, dtype=np.int64).reshape(M, 1)
        prod = acc * sw
        out = sat32(rshift_round(prod, SW_FRAC - s[None, :]))

        st = self.stats
        st.calls += 1
        st.macs += M * K * B
        st.tiles += math.ceil(M / self.lanes) * math.ceil(K / self.lanes) * math.ceil(B / self.lanes)
        st.cycles += self.tile_cycles(M, K, B)
        st.saturations += int(np.count_nonzero((out == INT32_MIN) | (out == INT32_MAX)))
        st.max_acc_bits = max(st.max_acc_bits, int(np.max(np.abs(acc))).bit_length() + 1 if acc.size else 0)
        return out.astype(np.int32)

    def _accumulate_tiled(self, Wq, Xq):
        """严格按 32×32 tile 累加，用于证明分块等价性（慢，仅测试用）。"""
        M, K = Wq.shape
        B = Xq.shape[1]
        acc = np.zeros((M, B), dtype=np.int64)
        for m0 in range(0, M, self.lanes):
            for k0 in range(0, K, self.lanes):
                w = Wq[m0:m0 + self.lanes, k0:k0 + self.lanes].astype(np.int32)
                x = Xq[k0:k0 + self.lanes, :].astype(np.int32)
                acc[m0:m0 + self.lanes, :] += w @ x
        return acc

    def dot_i8(self, Wq, Xq):
        """只做 int8 × int8 → int32 累加，不做行缩放（注意力打分这类用途）。

        与 gemm 共用同一条累加通路，因此与 RTL 累加器逐比特一致。
        """
        Wq = np.ascontiguousarray(Wq)
        Xq = np.ascontiguousarray(Xq)
        M, K = Wq.shape
        assert Xq.shape[0] == K, f"K 不匹配: {K} vs {Xq.shape[0]}"
        acc = np.rint(Wq.astype(np.float64) @ Xq.astype(np.float64)).astype(np.int64)
        acc = wrap_int32(acc)
        st = self.stats
        st.calls += 1
        st.macs += M * K * Xq.shape[1]
        st.tiles += math.ceil(M / self.lanes) * math.ceil(K / self.lanes) * math.ceil(Xq.shape[1] / self.lanes)
        st.cycles += self.tile_cycles(M, K, Xq.shape[1])
        return acc.astype(np.int32)

    def gemm_blocks(self, Wq, SW, Xq, sb, blk=32):
        """分块量化版矩阵乘：X 每 32 个元素一个移位量，累加器做对齐移位。

        Xq : (K, B) int8，第 b 块（K//blk 个块）的移位量在 sb[block, B]
        与单尺度版的关系：单尺度是 blk=K 的特例。

        真实硬件上这需要 64 位累加器 + 每个 K 块一次对齐移位（对 mvm_core.sv
        的扩展：权重流里每 32 个元素带一个指数，累加前先对齐）。
        输出仍是 Q16.16，最终只做一次 SW 乘法和饱和。
        """
        Wq = np.ascontiguousarray(Wq)
        Xq = np.ascontiguousarray(Xq)
        M, K = Wq.shape
        B = Xq.shape[1]
        assert K == Xq.shape[0]
        nblk = math.ceil(K / blk)
        sb = np.asarray(sb, dtype=np.int64)
        assert sb.shape == (nblk, B), f"sb {sb.shape} != {(nblk, B)}"
        sref = sb.min(axis=0)                        # 每个输出向量的公共指数
        acc = np.zeros((M, B), dtype=np.int64)
        for bi in range(nblk):
            k0, k1 = bi * blk, min((bi + 1) * blk, K)
            part = Wq[:, k0:k1].astype(np.float64) @ Xq[k0:k1, :].astype(np.float64)
            part = np.rint(part).astype(np.int64)
            acc += part << (sb[bi] - sref)[None, :]
        sw = np.asarray(SW, dtype=np.int64).reshape(M, 1)
        out = sat32(rshift_round(acc * sw, SW_FRAC - sref[None, :]))
        st = self.stats
        st.calls += 1
        st.macs += M * K * B
        st.tiles += math.ceil(M / self.lanes) * nblk * math.ceil(B / self.lanes)
        st.cycles += self.tile_cycles(M, K, B)
        return out.astype(np.int32)

    def linear_fx(self, Wq, SW, x, tile_scan: bool = False):
        """Q16 向量 → Q16 向量：先量化到 int8 + s，再过矩阵单元。"""
        x = np.asarray(x, dtype=np.int64).reshape(-1, 1)
        q, s = quantize_shift(x)
        return self.gemm(Wq, SW, q, np.array([s]), tile_scan=tile_scan)[:, 0]


def quantize_blocks(x, blk=32, qmax=127):
    """按 32 元素分块量化：每个块一个移位量 s（x ≈ q · 2^s）。

    CLIP 的 LayerNorm 输出有 16~26 的离群通道，整行一个尺度会把绝大多数
    元素压成 0；分块后每块的尺度跟着自己的量级走。与 32 路结构天然对齐。
    """
    x = np.asarray(x, dtype=np.int64)
    n, K = x.shape
    nblk = math.ceil(K / blk)
    pad = nblk * blk - K
    xp = np.pad(x, ((0, 0), (0, pad))) if pad else x
    xb = xp.reshape(n, nblk, blk)
    m = np.max(np.abs(xb), axis=2).astype(np.float64)
    # 取 s = ceil(log2(max/127))，让最大值尽量用满 ±127，比按位长取指数多抢 1 bit
    s = np.maximum(np.ceil(np.log2(np.maximum(m, 1.0) / qmax)).astype(np.int64), 0)
    q = np.clip(rshift_round(xb, s[:, :, None]), -qmax, qmax).astype(np.int8)
    over = np.max(np.abs(q.astype(np.int32)), axis=2) > qmax
    if np.any(over):
        s[over] += 1
        q[over] = np.clip(rshift_round(xb[over], s[over][:, None]), -qmax, qmax).astype(np.int8)
    return q.reshape(n, nblk * blk)[:, :K], s