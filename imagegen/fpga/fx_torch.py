"""定点算子的 torch/CUDA 版本：语义与 fx.py / sd_vae.conv_fx 完全一致。

目的：把「量化 + 舍入 + 卷积」整条算子链搬到 GPU，而不是只搬 GEMM。
位精确性由 test_fx_torch.py 对 numpy 版逐项回归。
"""

from __future__ import annotations

import math

import numpy as np
import torch

FRAC = 16
ONE = 1 << FRAC
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
SW_FRAC = 24


def dev(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 定点原语
def rshift_round(x: torch.Tensor, s) -> torch.Tensor:
    """x · 2^-s 并舍入（加半个 LSB 再算术右移），与 fx.rshift_round 一致。"""
    s = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.int64, device=x.device)
    if s.ndim < x.ndim:
        s = s.reshape((1,) * (x.ndim - s.ndim) + tuple(s.shape))
    s = s.expand_as(x)
    pos = s.clamp(min=0)
    neg = (-s).clamp(min=0)
    half = torch.where(pos > 0, torch.ones_like(x) << (pos.clamp(min=1) - 1), torch.zeros_like(x))
    return torch.where(s >= 0, (x + half) >> pos, x << neg)


def sat32(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(INT32_MIN, INT32_MAX)


def to_fx(x, device=None) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    return torch.round(t * ONE).to(torch.int64).to(dev(device))


def _bit_length(m: torch.Tensor) -> torch.Tensor:
    return torch.frexp(m.double())[1].to(torch.int64)


def quantize_shift_batch(x: torch.Tensor, qmax: int = 127):
    """(n,K) int64 Q16 → int8 (n,K) + 每行一个 s（与 sd_vae.quantize_shift_batch 同公式）。"""
    mx = x.abs().amax(dim=1).double()
    s = torch.ceil(torch.log2(torch.clamp(mx, min=1.0) / qmax)).to(torch.int64).clamp(min=0)
    q = rshift_round(x, s[:, None]).clamp(-qmax, qmax).to(torch.int8)
    over = q.to(torch.int32).abs().amax(dim=1) > qmax
    if bool(over.any()):
        s = torch.where(over, s + 1, s)
        q = rshift_round(x, s[:, None]).clamp(-qmax, qmax).to(torch.int8)
    return q, s


def quantize_blocks(x: torch.Tensor, blk: int = 32, qmax: int = 127):
    """(n,K) int64 → int8 (n,K) + (n, nblk) 每块一个 s。"""
    n, K = x.shape
    nblk = math.ceil(K / blk)
    if nblk * blk != K:
        x = torch.nn.functional.pad(x, (0, nblk * blk - K))
    xb = x.reshape(n, nblk, blk)
    m = xb.abs().amax(dim=2).double()
    s = torch.ceil(torch.log2(torch.clamp(m, min=1.0) / qmax)).to(torch.int64).clamp(min=0)
    q = rshift_round(xb, s[:, :, None]).clamp(-qmax, qmax).to(torch.int8)
    over = q.to(torch.int32).abs().amax(dim=2) > qmax
    if bool(over.any()):
        s = torch.where(over, s + 1, s)
        q = rshift_round(xb, s[:, :, None]).clamp(-qmax, qmax).to(torch.int8)
    return q.reshape(n, -1)[:, :K], s


# ---------------------------------------------------------------- 归一化
GN_EPS_Q16 = max(1, round(1e-6 * ONE))


def _rsqrt_lut():
    from fx import RSQRT_LUT
    return torch.tensor(RSQRT_LUT, dtype=torch.int64)


_RSQRT = None


def rsqrt_fx(m: torch.Tensor, in_frac: int = 16, out_frac: int = 16, iters: int = 2):
    """向量版 1/√m，与 fx.rsqrt_fx_v 同算法（LUT + 牛顿迭代）。"""
    global _RSQRT
    if _RSQRT is None or _RSQRT.device != m.device:
        _RSQRT = _rsqrt_lut().to(m.device)
    e = _bit_length(m) - 1 - 30
    e = torch.where(e % 2 != 0, e - 1, e)
    n = torch.where(e >= 0, m >> e.clamp(min=0), m << (-e).clamp(min=0))
    E = e + 30 - in_frac
    y = _RSQRT[(n >> 24) - 64]
    u16 = n >> 14
    for _ in range(iters):
        t = (u16 * ((y * y) >> 16)) >> 16
        y = (y * ((3 << 16) - t)) >> 17
    return rshift_round(y, 16 - out_frac + E // 2)


def groupnorm_fx(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, groups: int, eps_q16=GN_EPS_Q16):
    """x: (N, C) int64 Q16 → (N, C)，跨 N 位置 × 组内通道统计（与 fx.groupnorm_fx 一致）。"""
    n, c = x.shape
    g = c // groups
    xg = x.reshape(n, groups, g)
    total = n * g
    recip = round((1 << 24) / total)
    mean = rshift_round(xg.sum(dim=(0, 2), keepdim=True) * recip, 24)
    d = xg - mean
    d8 = d >> 8
    var = torch.clamp(rshift_round((d8 * d8).sum(dim=(0, 2), keepdim=True) * recip, 24), min=0)
    r = rsqrt_fx(torch.clamp(var + eps_q16, min=1))
    y = sat32(rshift_round(d * r.repeat_interleave(g, dim=2), 16)).reshape(n, c)
    return sat32(rshift_round(y * gamma[None, :], 16) + beta[None, :])


def layernorm_fx(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps_q16=GN_EPS_Q16):
    """逐行归一化（CLIP 用）。"""
    n, c = x.shape
    recip = round((1 << 24) / c)
    mean = rshift_round(x.sum(dim=1, keepdim=True) * recip, 24)
    d = x - mean
    d8 = d >> 8
    var = torch.clamp(rshift_round((d8 * d8).sum(dim=1, keepdim=True) * recip, 24), min=0)
    r = rsqrt_fx(torch.clamp(var + eps_q16, min=1))
    y = sat32(rshift_round(d * r, 16))
    return sat32(rshift_round(y * gamma[None, :], 16) + beta[None, :])


SIG_LUT = None


def silu_fx(x: torch.Tensor) -> torch.Tensor:
    """查表 + 线性插值，与 fx.silu_fx 一致。"""
    global SIG_LUT
    if SIG_LUT is None or SIG_LUT.device != x.device:
        from fx import SIG_LUT as L
        SIG_LUT = torch.tensor(L, dtype=torch.int64).to(x.device)
    pos = (x + (16 << 16)).clamp(0, 512 << 12)
    idx = (pos >> 12).clamp(max=511)
    fr = pos - (idx << 12)
    a = SIG_LUT[idx]
    b = SIG_LUT[idx + 1]
    return rshift_round(x * (a + (((b - a) * fr) >> 12)), 16)


def softmax_fx(scores: torch.Tensor, axis: int = -1) -> torch.Tensor:
    from fx import EXP2_LUT, LOG2E_Q16
    lut = torch.tensor(EXP2_LUT, dtype=torch.int64, device=scores.device)
    d = scores - scores.amax(dim=axis, keepdim=True)
    t = rshift_round(d * LOG2E_Q16, 16)
    ip = t >> 16
    f = t - (ip << 16)
    idx = f >> 8
    fr = f & 0xFF
    a = lut[idx]
    b = lut[idx + 1]
    v = a + (((b - a) * fr) >> 8)
    e = rshift_round(torch.where(ip == 0, v, v >> (-ip).clamp(min=0)), 0) if False else rshift_round(v, -ip)
    total = e.sum(dim=axis, keepdim=True)
    r = recip_fx(total, in_frac=16, out_frac=32)
    return rshift_round(e * r, 32)


RECIP_LUT = None


def recip_fx(m: torch.Tensor, in_frac: int = 16, out_frac: int = 16, iters: int = 2):
    """向量版 1/m，与 fx.recip_fx_v 同算法。"""
    global RECIP_LUT
    if RECIP_LUT is None or RECIP_LUT.device != m.device:
        from fx import RECIP_LUT as L
        RECIP_LUT = torch.tensor(L, dtype=torch.int64).to(m.device)
    e = _bit_length(m) - 1 - 30
    n = torch.where(e >= 0, m >> e.clamp(min=0), m << (-e).clamp(min=0))
    E = e + 30 - in_frac
    y = RECIP_LUT[(n >> 22) - 256]
    u16 = n >> 14
    for _ in range(iters):
        t = (u16 * y) >> 16
        y = (y * ((2 << 16) - t)) >> 16
    return rshift_round(y, 16 - out_frac + E)


# ---------------------------------------------------------------- 卷积 / 线性
def _imm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8 矩阵乘 → int32；_int_mm 要求 K 和 N 都是 8 的倍数，这里补零。"""
    k = a.shape[1]
    k8 = (k + 7) // 8 * 8
    if k8 != k:
        a = torch.nn.functional.pad(a, (0, k8 - k))
        b = torch.nn.functional.pad(b, (0, 0, 0, k8 - k))
    n = b.shape[1]
    n8 = (n + 7) // 8 * 8
    if n8 != n:
        b = torch.nn.functional.pad(b, (0, n8 - n))
    try:
        return torch._int_mm(a.contiguous(), b.contiguous())[:, :n]
    except RuntimeError:
        # 少数形状 cuBLAS 的 int8 内核不支持：退回 CPU 的 int32 整数乘（仍是整数运算，无浮点）
        aa = a.cpu().numpy().astype(np.int32)
        bb = b.cpu().numpy().astype(np.int32)
        return torch.from_numpy(aa @ bb).to(a.device)[:, :n]


def conv_fx(x: torch.Tensor, wq: torch.Tensor, sw: torch.Tensor, bq: torch.Tensor,
            pad: int = 1, stride: int = 1, chunk: int = 8192):
    """x: (H,W,C) int64 Q16 → (Hout,Wout,O)。权重 int8 (O, kh*kw*C)。

    按输出像素分块：512² 的 im2col 一次要 2.4GB int64，分块后每块只有几十 MB。
    """
    kh = kw = 3 if pad else 1
    h, w, c = x.shape
    xp = torch.nn.functional.pad(x, (0, 0, pad, pad, pad, pad)) if pad else x
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (w + 2 * pad - kw) // stride + 1
    B = ho * wo
    out = torch.empty((B, wq.shape[0]), dtype=torch.int32, device=x.device)
    for s0 in range(0, B, chunk):
        s1 = min(s0 + chunk, B)
        idx = torch.arange(s0, s1, device=x.device)
        if kh * kw > 1:
            rr = (idx // wo) * stride
            cc = (idx % wo) * stride
            blk = torch.empty((s1 - s0, kh * kw * c), dtype=x.dtype, device=x.device)
            tpos = 0
            for i in range(kh):
                for j in range(kw):
                    blk[:, tpos * c:(tpos + 1) * c] = xp[rr + i, cc + j, :]
                    tpos += 1
        else:
            rr = (idx // wo) * stride
            cc = (idx % wo) * stride
            blk = xp[rr, cc, :]
        q, s = quantize_shift_batch(blk.contiguous())
        acc = _imm(wq, q.T)                                     # (O, chunk)
        out[s0:s1] = _scale_round(acc, sw, s).T
        del blk
    return (out.to(torch.int64) + bq[None, :]).reshape(ho, wo, wq.shape[0])


def _scale_round(acc: torch.Tensor, SW: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """acc (M,B) int32，SW (M,) Q24，s (B,) → int32 Q16。"""
    prod = acc.to(torch.int64) * SW.to(torch.int64)[:, None]
    return sat32(rshift_round(prod, SW_FRAC - s[None, :]))


def linear_fx(x: torch.Tensor, wq: torch.Tensor, sw: torch.Tensor, bq: torch.Tensor, blocks: bool = True):
    """x: (n,K) int64 Q16 → (n,O)。缺省走分块量化（残差流有离群通道）。"""
    if blocks:
        q, sb = quantize_blocks(x)
        nblk = sb.shape[1]
        sref = sb.min(dim=1).values
        acc = torch.zeros((wq.shape[0], q.shape[0]), dtype=torch.int64, device=x.device)
        for bi in range(nblk):
            k0, k1 = bi * 32, min((bi + 1) * 32, q.shape[1])
            part = _imm(wq[:, k0:k1], q[:, k0:k1].T).to(torch.int64)
            sh = (sb[:, bi] - sref).clamp(min=0)
            acc += part << sh[None, :]
        return _scale_round(acc, sw, sref).T + bq[None, :]
    q, s = quantize_shift_batch(x)
    return _scale_round(_imm(wq, q.T), sw, s).T + bq[None, :]
