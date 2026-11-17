"""把 l0/fixedpoint.py 的定点原语向量化（逐比特对齐标量版）。

l0 的 rsqrt_fx / recip_fx 是标量实现，图像模型里要按像素批量算，这里给出
等价向量版；数值结果与 l0 的标量版逐比特一致（test_fx.py 验证）。
"""

from __future__ import annotations

import math

import numpy as np

FRAC = 16
ONE = 1 << FRAC
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def rshift_round(x, s):
    x = np.asarray(x, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    pos = np.minimum(np.maximum(s, 0), 62)
    neg = np.maximum(-s, 0)
    half = np.where(pos > 0, np.left_shift(np.int64(1), np.maximum(pos - 1, 0)), 0)
    return np.where(s >= 0, np.right_shift(x + half, pos), np.left_shift(x, neg))


def sat32(x, lo=INT32_MIN, hi=INT32_MAX):
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


def to_fx(x_float):
    return np.round(np.asarray(x_float, dtype=np.float64) * ONE).astype(np.int64)


def from_fx(x):
    return np.asarray(x, dtype=np.int64).astype(np.float64) / ONE


# ---------------------------------------------------------------- 位宽/指数
def bit_length(x):
    """int64 数组的位长（等价 Python int.bit_length，x>0）。"""
    x = np.asarray(x, dtype=np.int64)
    return np.frexp(x.astype(np.float64))[1].astype(np.int64)  # 2^(e-1) <= x < 2^e


def _floor_log2(x):
    x = np.asarray(x, dtype=np.int64)
    return bit_length(x) - 1


# ---------------------------------------------------------------- 1/√x、1/x
RSQRT_LUT = [round(ONE / math.sqrt((i + 64 + 0.5) / 64)) for i in range(192)]
RECIP_LUT = [round(ONE / ((i + 256 + 0.5) / 256)) for i in range(256)]
_RSQRT = np.array(RSQRT_LUT, dtype=np.int64)
_RECIP = np.array(RECIP_LUT, dtype=np.int64)


def rsqrt_fx_v(m, in_frac=FRAC, out_frac=FRAC, iters=2):
    """向量版 1/√m。与 l0.fixedpoint.rsqrt_fx 的标量算式逐步一致。"""
    m = np.asarray(m, dtype=np.int64)
    assert in_frac % 2 == 0
    e = _floor_log2(m) - 30
    e = np.where(e % 2 != 0, e - 1, e)          # 指数取偶数
    n = np.where(e >= 0, m >> np.maximum(e, 0), m << np.maximum(-e, 0))
    E = e + 30 - in_frac
    y = _RSQRT[(n >> 24) - 64]
    u16 = n >> 14
    for _ in range(iters):
        t = (u16 * ((y * y) >> 16)) >> 16
        y = (y * ((3 << 16) - t)) >> 17
    return rshift_round(y, 16 - out_frac + E // 2)


def recip_fx_v(m, in_frac=FRAC, out_frac=FRAC, iters=2):
    """向量版 1/m。与 l0.fixedpoint.recip_fx 的标量算式逐步一致。"""
    m = np.asarray(m, dtype=np.int64)
    e = _floor_log2(m) - 30
    n = np.where(e >= 0, m >> np.maximum(e, 0), m << np.maximum(-e, 0))
    E = e + 30 - in_frac
    y = _RECIP[(n >> 22) - 256]
    u16 = n >> 14
    for _ in range(iters):
        t = (u16 * y) >> 16
        y = (y * ((2 << 16) - t)) >> 16
    return rshift_round(y, 16 - out_frac + E)


# ---------------------------------------------------------------- exp / softmax / 激活
EXP2_LUT = np.array([round(ONE * 2 ** (i / 256)) for i in range(257)], dtype=np.int64)
LOG2E_Q16 = round(math.log2(math.e) * ONE)


def exp2_neg_fx(t):
    t = np.asarray(t, dtype=np.int64)
    ip = t >> 16
    f = t - (ip << 16)
    idx, fr = f >> 8, f & 0xFF
    a, b = EXP2_LUT[idx], EXP2_LUT[idx + 1]
    v = a + (((b - a) * fr) >> 8)
    return rshift_round(v, -ip)


def softmax_fx(scores, axis=-1):
    scores = np.asarray(scores, dtype=np.int64)
    mx = np.max(scores, axis=axis, keepdims=True)
    d = scores - mx
    t = rshift_round(d * LOG2E_Q16, 16)
    e = exp2_neg_fx(t)
    total = np.sum(e, axis=axis, keepdims=True)
    r = recip_fx_v(total, in_frac=16, out_frac=32)
    return rshift_round(e * r, 32)


SIG_LUT = np.array([round(ONE / (1 + math.exp(16 - i / 16))) for i in range(513)], dtype=np.int64)


def sigmoid_fx(x):
    x = np.asarray(x, dtype=np.int64)
    pos = np.clip(x + (16 << 16), 0, 512 << 12)
    idx = np.minimum(pos >> 12, 511)
    fr = pos - (idx << 12)
    a, b = SIG_LUT[idx], SIG_LUT[idx + 1]
    return a + (((b - a) * fr) >> 12)


def silu_fx(x):
    x = np.asarray(x, dtype=np.int64)
    return rshift_round(x * sigmoid_fx(x), 16)


# ---------------------------------------------------------------- GroupNorm / LayerNorm
GN_EPS_Q16 = max(1, round(1e-6 * ONE))


def groupnorm_fx(x, gamma_fx, beta_fx, groups, eps_q16=GN_EPS_Q16):
    """x: (N, C) Q16 → (N, C) Q16，N = H*W 个空间位置。

    与 PyTorch GroupNorm 一致：均值/方差在每组上跨 **N 个位置 × g 个通道** 统计。
    均值保持 Q16 精度（大直流偏移也不丢精度），平方和降到 Q8 控制位宽。
    要求 |x| < 2^20。
    """
    x = np.asarray(x, dtype=np.int64)
    n, c = x.shape
    assert c % groups == 0
    g = c // groups
    xg = x.reshape(n, groups, g)
    total = n * g
    recip = round((1 << 24) / total)
    mean = rshift_round(np.sum(xg, axis=(0, 2), keepdims=True) * recip, 24)   # Q16
    d = xg - mean                                                             # Q16
    d8 = d >> 8                                                               # Q8
    var16 = np.maximum(rshift_round(np.sum(d8 * d8, axis=(0, 2), keepdims=True) * recip, 24), 0)
    r = rsqrt_fx_v(np.maximum(var16 + eps_q16, 1), in_frac=16, out_frac=16)   # Q16
    y = sat32(rshift_round(d * np.repeat(r, g, axis=2), 16)).reshape(n, c)
    return sat32(rshift_round(y * np.asarray(gamma_fx, dtype=np.int64)[None, :], 16)
                 + np.asarray(beta_fx, dtype=np.int64)[None, :])


def layernorm_fx(x, gamma_fx, beta_fx, eps_q16=GN_EPS_Q16):
    """x: (N, C) Q16 → (N, C)，**逐行**归一化（LayerNorm 的语义，与 GroupNorm 不同）。"""
    x = np.asarray(x, dtype=np.int64)
    n, c = x.shape
    recip = round((1 << 24) / c)
    mean = rshift_round(np.sum(x, axis=1, keepdims=True) * recip, 24)        # Q16
    d = x - mean
    d8 = d >> 8
    var16 = np.maximum(rshift_round(np.sum(d8 * d8, axis=1, keepdims=True) * recip, 24), 0)
    r = rsqrt_fx_v(np.maximum(var16 + eps_q16, 1), in_frac=16, out_frac=16)
    y = sat32(rshift_round(d * r, 16))
    return sat32(rshift_round(y * np.asarray(gamma_fx, dtype=np.int64)[None, :], 16)
                 + np.asarray(beta_fx, dtype=np.int64)[None, :])
