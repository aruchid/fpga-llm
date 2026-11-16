"""硬件等价的定点运算原语。RTL 必须与这里逐比特一致。

数值格式约定
- 激活值 / 残差流：Q16.16，存在有符号 32 位寄存器里（FRAC = 16）。
- 权重：int8 对称量化（-127..127），每个输出行一个缩放系数 SW，SW = round(sw * 2^24)。
- 送进矩阵单元的激活值：int8 + 一个移位量 s（2 的幂缩放），x_real ≈ x_q * 2^(s-16)。
  s 由最大绝对值的位宽决定（前导零计数），硬件里不需要除法。
- 舍入：统一用 rshift_round，即加半个 LSB 后算术右移（向 +inf 方向舍入一半）。
- 查找表（LUT）的内容在“编译期”用浮点算好，相当于 ROM 初值。
"""
import math

import numpy as np

FRAC = 16
ONE = 1 << FRAC
SW_FRAC = 24  # 权重行缩放系数的小数位
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


# ---------------------------------------------------------------- 位宽监控
class Monitor:
    """记录各处数值实际用到的最大有符号位宽和饱和次数，供 RTL 定位宽参考。"""

    def __init__(self):
        self.bits = {}
        self.saturations = {}

    def record(self, tag, arr):
        a = np.asarray(arr)
        if a.size == 0:
            return
        m = int(np.max(np.abs(a)))
        self.bits[tag] = max(self.bits.get(tag, 0), m.bit_length() + 1)

    def report(self):
        lines = [f"  {k:<24} {v:>3} 位" for k, v in sorted(self.bits.items())]
        for k, v in sorted(self.saturations.items()):
            lines.append(f"  [饱和] {k:<17} {v} 次")
        return "\n".join(lines)


MON = Monitor()


# ---------------------------------------------------------------- 基础操作
def rshift_round(x, s):
    """x * 2^-s 并舍入。s 可以是数组；s<0 表示左移。"""
    x = np.asarray(x, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    pos = np.minimum(np.maximum(s, 0), 62)
    neg = np.maximum(-s, 0)
    half = np.where(pos > 0, np.left_shift(np.int64(1), np.maximum(pos - 1, 0)), 0)
    return np.where(s >= 0, np.right_shift(x + half, pos), np.left_shift(x, neg))


def sat32(x, tag=None):
    x = np.asarray(x, dtype=np.int64)
    if tag:
        MON.record(tag, x)
        n = int(np.count_nonzero((x < INT32_MIN) | (x > INT32_MAX)))
        if n:
            MON.saturations[tag] = MON.saturations.get(tag, 0) + n
    return np.clip(x, INT32_MIN, INT32_MAX)


def to_fx(x_float):
    """浮点 → Q16.16（只在“编译期”准备常量时用）。"""
    return np.round(np.asarray(x_float, dtype=np.float64) * ONE).astype(np.int64)


# ---------------------------------------------------------------- 量化
def quantize_weight_rows(w):
    """浮点权重 (out, in) → int8 + 每行缩放 SW（Q24）。离线完成。"""
    w = np.asarray(w, dtype=np.float64)
    sw = np.max(np.abs(w), axis=1) / 127.0
    sw = np.where(sw == 0, 1.0 / (1 << SW_FRAC), sw)
    SW = np.round(sw * (1 << SW_FRAC)).astype(np.int64)
    sw_used = SW / float(1 << SW_FRAC)
    q = np.clip(np.round(w / sw_used[:, None]), -127, 127).astype(np.int8)
    return q, SW


def quantize_shift(x, qmax=127):
    """Q16 向量 → int8 + 移位量 s，x_real ≈ q * 2^(s-16)。s 由最大值的位宽决定。"""
    x = np.asarray(x, dtype=np.int64)
    m = int(np.max(np.abs(x))) if x.size else 0
    s = max(0, m.bit_length() - 7)
    if int(rshift_round(m, s)) > qmax:
        s += 1
    q = np.clip(rshift_round(x, s), -qmax, qmax)
    return q, s


def int_matmul(wq, SW, xq, s, tag="matmul"):
    """矩阵单元：int8 × int8 → int32 累加，再乘行缩放得到 Q16。

    分块（32×32 tile）累加是精确整数加法，所以结果与分块方式无关。
    """
    acc = (wq.astype(np.int32) @ xq.astype(np.int32)).astype(np.int64)
    MON.record(tag + ".acc", acc)
    prod = acc * SW
    MON.record(tag + ".acc*SW", prod)
    return sat32(rshift_round(prod, SW_FRAC - s), tag + ".out")


# ---------------------------------------------------------------- 1/√x 与 1/x
# 1/√u，u ∈ [1,4)，按 n>>24 ∈ [64,256) 查表，取区间中点
RSQRT_LUT = [round(ONE / math.sqrt((i + 64 + 0.5) / 64)) for i in range(192)]
# 1/u，u ∈ [1,2)，按 n>>22 ∈ [256,512) 查表
RECIP_LUT = [round(ONE / ((i + 256 + 0.5) / 256)) for i in range(256)]


def rsqrt_fx(m, in_frac=FRAC, out_frac=FRAC, iters=2):
    """标量：输入 m（Q in_frac，>0），输出 1/√m（Q out_frac）。查表 + 牛顿迭代。"""
    m = int(m)
    assert m > 0 and in_frac % 2 == 0
    e = m.bit_length() - 1 - 30
    if e % 2:
        e -= 1                      # 指数取偶数，开方后才是整数
    n = m >> e if e >= 0 else m << -e   # n ∈ [2^30, 2^32)，u = n/2^30 ∈ [1,4)
    E = e + 30 - in_frac
    y = RSQRT_LUT[(n >> 24) - 64]       # Q16
    u16 = n >> 14                       # u 的 Q16
    for _ in range(iters):              # y ← y·(3 − u·y²)/2
        t = (u16 * ((y * y) >> 16)) >> 16
        y = (y * ((3 << 16) - t)) >> 17
    return int(rshift_round(y, 16 - out_frac + E // 2))


def recip_fx(m, in_frac=FRAC, out_frac=FRAC, iters=2):
    """标量：输入 m（Q in_frac，>0），输出 1/m（Q out_frac）。查表 + 牛顿迭代。"""
    m = int(m)
    assert m > 0
    e = m.bit_length() - 1 - 30
    n = m >> e if e >= 0 else m << -e   # n ∈ [2^30, 2^31)，u ∈ [1,2)
    E = e + 30 - in_frac
    y = RECIP_LUT[(n >> 22) - 256]
    u16 = n >> 14
    for _ in range(iters):              # y ← y·(2 − u·y)
        t = (u16 * y) >> 16
        y = (y * ((2 << 16) - t)) >> 16
    return int(rshift_round(y, 16 - out_frac + E))


# ---------------------------------------------------------------- exp / softmax
EXP2_LUT = np.array([round(ONE * 2 ** (i / 256)) for i in range(257)], dtype=np.int64)
LOG2E_Q16 = round(math.log2(math.e) * ONE)


def exp2_neg_fx(t):
    """t ≤ 0（Q16 数组）→ 2^t（Q16）。整数部分移位，小数部分查表 + 线性插值。"""
    t = np.asarray(t, dtype=np.int64)
    ip = t >> 16                     # 向下取整，≤ 0
    f = t - (ip << 16)               # [0, 1) 的 Q16
    idx, fr = f >> 8, f & 0xFF
    a, b = EXP2_LUT[idx], EXP2_LUT[idx + 1]
    v = a + (((b - a) * fr) >> 8)    # 2^f，Q16 ∈ [1, 2]
    return rshift_round(v, -ip)


def softmax_fx(scores):
    """Q16 分数 → Q16 概率。减最大值 → 2 的幂 → 求和 → 一次倒数 → 逐个相乘。"""
    scores = np.asarray(scores, dtype=np.int64)
    d = scores - np.max(scores)                      # ≤ 0
    t = rshift_round(d * LOG2E_Q16, 16)              # e^d = 2^(d·log2e)
    e = exp2_neg_fx(t)
    total = int(np.sum(e))                           # ≥ 1.0（最大值那项）
    MON.record("softmax.sum", total)
    r = recip_fx(total, in_frac=16, out_frac=32)     # 1/sum，Q32
    return rshift_round(e * r, 32)


# ---------------------------------------------------------------- SiLU
SIG_LUT = np.array([round(ONE / (1 + math.exp(16 - i / 16))) for i in range(513)], dtype=np.int64)


def sigmoid_fx(x):
    """Q16 → Q16。[-16, 16) 内 512 段查表（步长 1/16）+ 线性插值，范围外饱和到 0 / 1。"""
    x = np.asarray(x, dtype=np.int64)
    pos = np.clip(x + (16 << 16), 0, 512 << 12)
    idx = np.minimum(pos >> 12, 511)
    fr = pos - (idx << 12)
    a, b = SIG_LUT[idx], SIG_LUT[idx + 1]
    return a + (((b - a) * fr) >> 12)


def silu_fx(x):
    return rshift_round(np.asarray(x, dtype=np.int64) * sigmoid_fx(x), 16)


# ---------------------------------------------------------------- RMSNorm
EPS_Q16 = max(1, round(1e-5 * ONE))


def rmsnorm_fx(x, g_fx, tag="rmsnorm"):
    """x、g 均为 Q16。平方和在 Q8 下累加以控制位宽，除以 n 用常数倒数乘法。"""
    x = np.asarray(x, dtype=np.int64)
    n = x.size
    xs = rshift_round(x, 8)                       # Q8
    ss = int(np.sum(xs * xs))                     # Q16
    MON.record(tag + ".sumsq", ss)
    recip_n = round((1 << 24) / n)                # 常数，ROM
    mean = int(rshift_round(ss * recip_n, 24)) + EPS_Q16
    r = rsqrt_fx(mean, in_frac=16, out_frac=16)
    MON.record(tag + ".rsqrt", r)
    y = rshift_round(x * r, 16)
    return sat32(rshift_round(y * g_fx, 16), tag + ".out")


# ---------------------------------------------------------------- RoPE
ROPE_FRAC = 14


def rope_tables(seq_len, head_size):
    """cos/sin 表（Q14），[pos, j]，j 为头内第 j 对。"""
    j = np.arange(head_size // 2, dtype=np.float64)
    freq = 1.0 / np.power(10000.0, 2 * j / head_size)
    ang = np.arange(seq_len, dtype=np.float64)[:, None] * freq[None, :]
    scale = 1 << ROPE_FRAC
    return (np.round(np.cos(ang) * scale).astype(np.int64),
            np.round(np.sin(ang) * scale).astype(np.int64))


def rope_fx(vec, cos_row, sin_row, head_size):
    v = np.asarray(vec, dtype=np.int64).reshape(-1, head_size // 2, 2)
    v0, v1 = v[..., 0], v[..., 1]
    out = np.empty_like(v)
    out[..., 0] = rshift_round(v0 * cos_row - v1 * sin_row, ROPE_FRAC)
    out[..., 1] = rshift_round(v0 * sin_row + v1 * cos_row, ROPE_FRAC)
    return out.reshape(-1)


# ---------------------------------------------------------------- 注意力（一个头）
def attention_head_fx(qq, qs, Kq, ks, Vq, vs, inv_sqrt_hs):
    """一个注意力头。qq: int8 查询 + 移位 qs；Kq/Vq: (T, hs) int8 + 每行移位 ks/vs。返回 Q16 输出。"""
    dots = Kq @ qq                                   # 矩阵单元：int8 点积
    MON.record("att.dot", dots)
    # score = dots·2^(qs+ks-32)/√hs，转成 Q16
    scores = rshift_round(dots * inv_sqrt_hs, 32 - qs - ks)
    p = softmax_fx(scores)                           # Q16
    acc = np.sum((p[:, None] * Vq) << vs[:, None], axis=0)
    MON.record("att.pv_acc", acc)
    return rshift_round(acc, 16)


def silu_gate_fx(h1, h3):
    """SwiGLU 门控：silu(h1) · h3，Q16。"""
    return sat32(rshift_round(silu_fx(h1) * h3, 16), "swiglu")


def resadd_fx(x, y):
    return sat32(np.asarray(x, dtype=np.int64) + y, "residual")
