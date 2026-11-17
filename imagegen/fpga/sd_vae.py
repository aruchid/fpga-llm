"""SD VAE 解码器：numpy 浮点版（对照） + 32 路定点版（FPGA 路径）。

权重直接读 ComfyUI 的 sd_turbo.safetensors（first_stage_model.*）。
浮点版用来验证网络实现与 ComfyUI 的 float 解码一致；定点版把每个卷积/注意力
都送进 mvm32.MVM32（int8 权重 + Q16.16 激活 + 每像素一个移位量 s）。
"""

from __future__ import annotations

import numpy as np

import fx
import mvm32

VAE_PREFIX = "first_stage_model"


# ---------------------------------------------------------------- 权重
class Weights:
    def __init__(self, path, prefix=VAE_PREFIX):
        self.path = path
        self.prefix = prefix
        self._cache = {}
        self._handle = None

    def _open(self):
        if self._handle is None:
            from safetensors import safe_open
            self._handle = safe_open(self.path, framework="numpy")
        return self._handle

    def raw(self, name):
        if name not in self._cache:
            self._cache[name] = self._open().get_tensor(f"{self.prefix}.{name}")
        return self._cache[name]

    def conv(self, name):
        """→ (weight (O,I,kh,kw) float32, bias (O,) float32)"""
        return self.raw(name + ".weight").astype(np.float32), self.raw(name + ".bias").astype(np.float32)

    def vec(self, name):
        return self.raw(name + ".weight").astype(np.float32), self.raw(name + ".bias").astype(np.float32)


# ---------------------------------------------------------------- 浮点原语
def _w2d(w):
    o, c, kh, kw = w.shape
    return w.transpose(0, 2, 3, 1).reshape(o, kh * kw * c)


def conv_float(x, w, b, pad=1, chunk=8192, stride=1):
    """x: (H,W,C) float32 → (Hout,Wout,O)。stride=2 用于 UNet 的下采样块。"""
    _, c, kh, kw = w.shape
    h, wd = x.shape[:2]
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wd + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0))) if pad else x
    w2d = _w2d(w).astype(np.float32)
    y = np.empty((ho * wo, w.shape[0]), dtype=np.float32)
    for s0 in range(0, ho * wo, chunk):
        s1 = min(s0 + chunk, ho * wo)
        rr, cc = np.divmod(np.arange(s0, s1), wo)
        rr = rr * stride
        cc = cc * stride
        blk = np.empty((s1 - s0, kh * kw * c), dtype=np.float32)
        t = 0
        for i in range(kh):
            for j in range(kw):
                blk[:, t * c:(t + 1) * c] = xp[rr + i, cc + j, :]
                t += 1
        y[s0:s1] = blk @ w2d.T
    return (y + b).reshape(ho, wo, w.shape[0])


def gn_float(x, gamma, beta, groups=32, eps=1e-6):
    h, w, c = x.shape
    xg = x.reshape(-1, groups, c // groups)
    mu = xg.mean(axis=(0, 2), keepdims=True)
    var = xg.var(axis=(0, 2), keepdims=True)
    y = ((xg - mu) / np.sqrt(var + eps)).reshape(h, w, c)
    return y * gamma + beta


def silu_f32(x):
    xc = np.clip(x, -60, 60)
    return (x / (1.0 + np.exp(-xc))).astype(np.float32)


def upsample(x):
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


def attn_float(x, wt, gamma, beta, prefix="decoder.mid.attn_1"):
    h, w, c = x.shape
    n = h * w
    xn = gn_float(x, gamma, beta).reshape(n, c)
    ws = {k: wt.conv(f"{prefix}.{k}") for k in ("q", "k", "v", "proj_out")}
    q = xn @ ws["q"][0].reshape(c, c).T + ws["q"][1]
    k_ = xn @ ws["k"][0].reshape(c, c).T + ws["k"][1]
    v = xn @ ws["v"][0].reshape(c, c).T + ws["v"][1]
    sc = (q @ k_.T) / np.sqrt(np.float32(c))
    sc -= sc.max(axis=1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(axis=1, keepdims=True)
    o = p @ v
    o = o @ ws["proj_out"][0].reshape(c, c).T + ws["proj_out"][1]
    return x + o.reshape(h, w, c)


# ---------------------------------------------------------------- 定点原语
def _wq(w):
    """浮点权重 (O,I,kh,kw) → int8 (O, kh*kw*I) + SW (O,)。"""
    return mvm32.quantize_weight_rows(_w2d(w))


def quantize_shift_batch(x, qmax=127):
    """x: (n,K) int64 Q16 → int8 (n,K) + 每行一个 s（与 l0.quantize_shift 等价）。"""
    x = np.asarray(x, dtype=np.int64)
    mx = np.max(np.abs(x), axis=1).astype(np.float64)
    s = np.maximum(np.ceil(np.log2(np.maximum(mx, 1.0) / qmax)).astype(np.int64), 0)
    q = np.clip(fx.rshift_round(x, s[:, None]), -qmax, qmax).astype(np.int8)
    over = np.max(np.abs(q.astype(np.int32)), axis=1) > qmax
    if np.any(over):
        s[over] += 1
        q[over] = np.clip(fx.rshift_round(x[over], s[over, None]), -qmax, qmax).astype(np.int8)
    return q, s


def conv_fx(x, wq, sw, bq, engine, pad=1, chunk=1024, stride=1, blockq=False):
    """x: (H,W,C) Q16 int64 → (Hout,Wout,O) Q16 int64，矩阵乘全走 32 路引擎。"""
    kh = kw = 3 if pad else 1
    h, wd = x.shape[:2]
    c = x.shape[2]
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wd + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0))) if pad else x
    out = np.empty((ho * wo, wq.shape[0]), dtype=np.int64)
    for s0 in range(0, ho * wo, chunk):
        s1 = min(s0 + chunk, ho * wo)
        rr, cc = np.divmod(np.arange(s0, s1), wo)
        rr = rr * stride
        cc = cc * stride
        if kh * kw > 1:
            blk = np.empty((s1 - s0, kh * kw * c), dtype=np.int64)
            t = 0
            for i in range(kh):
                for j in range(kw):
                    blk[:, t * c:(t + 1) * c] = xp[rr + i, cc + j, :]
                    t += 1
        else:
            blk = xp[rr, cc, :]
        if blockq:
            q, sb = mvm32.quantize_blocks(blk)
            out[s0:s1] = engine.gemm_blocks(wq, sw, np.ascontiguousarray(q.T), sb.T).T
        else:
            q, s = quantize_shift_batch(blk)
            out[s0:s1] = engine.gemm(wq, sw, np.ascontiguousarray(q.T), s).T
        del blk
    del xp
    return (out + bq).reshape(ho, wo, wq.shape[0])


def gn_fx(x, gamma_q, beta_q, groups=32):
    h, w, c = x.shape
    return fx.groupnorm_fx(x.reshape(h * w, c), gamma_q, beta_q, groups).reshape(h, w, c)


def silu_fx(x):
    return fx.silu_fx(x)


def attn_fx(x, wt, gamma_q, beta_q, engine, prefix="decoder.mid.attn_1", qchunk=256):
    """VAE mid 单头自注意力：Q/K/V/输出投影走矩阵单元，打分与 PV 走 int8 点积。"""
    h, wd, c = x.shape
    n = h * wd
    xn = gn_fx(x, gamma_q, beta_q).reshape(1, n, c)

    def proj1x1(name):
        w, b = wt.conv(f"{prefix}.{name}")
        wq, sw = _wq(w)
        return conv_fx(xn, wq, sw, fx.to_fx(b.astype(np.float64)), engine, pad=0).reshape(n, c)

    q = proj1x1("q")
    k_ = proj1x1("k")
    v = proj1x1("v")

    scale_q16 = fx.to_fx(1.0 / np.sqrt(float(c)))
    q = fx.rshift_round(q * scale_q16, 16)          # 把 1/√d 折进 q

    kq, ks = quantize_shift_batch(k_)
    # v 沿“键”方向收缩，缩放必须按输出通道（列）而不是按行，否则提不出来
    vq_c, vs = quantize_shift_batch(np.ascontiguousarray(v.T))
    out = np.empty((n, c), dtype=np.int64)
    for r0 in range(0, n, qchunk):
        r1 = min(r0 + qchunk, n)
        qq, qs = quantize_shift_batch(q[r0:r1])
        dots = engine.dot_i8(kq, np.ascontiguousarray(qq.T))          # (n, m)
        sc = fx.rshift_round(dots.astype(np.int64), 32 - qs[None, :] - ks[:, None])
        p = fx.softmax_fx(sc.T, axis=1)                               # (m, n) Q16
        pq, ps = quantize_shift_batch(p)                              # 概率按 int8 进矩阵单元
        pv = engine.dot_i8(pq, np.ascontiguousarray(vq_c.T))          # (m, n)@(n,c) → (m,c)
        out[r0:r1] = fx.rshift_round(pv.astype(np.int64), 16 - ps[:, None] - vs[None, :])

    w, b = wt.conv(f"{prefix}.proj_out")
    wq, sw = _wq(w)
    o = conv_fx(out.reshape(1, n, c), wq, sw, fx.to_fx(b.astype(np.float64)), engine, pad=0)
    return fx.sat32(x + o.reshape(h, wd, c))


# ---------------------------------------------------------------- 解码器主干
def _resnet_float(wt, x, prefix, in_c, out_c):
    g1, b1 = wt.vec(prefix + ".norm1")
    h = conv_float(silu_f32(gn_float(x, g1, b1)), *wt.conv(prefix + ".conv1"), pad=1)
    g2, b2 = wt.vec(prefix + ".norm2")
    h = conv_float(silu_f32(gn_float(h, g2, b2)), *wt.conv(prefix + ".conv2"), pad=1)
    sc = conv_float(x, *wt.conv(prefix + ".nin_shortcut"), pad=0) if in_c != out_c else x
    return h + sc


def decoder_float(wt: Weights, z, scale_factor=1.0):
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 4:
        z = z[0]
    x = np.transpose(z / scale_factor, (1, 2, 0)).astype(np.float32)

    x = conv_float(x, *wt.conv("post_quant_conv"), pad=0)
    x = conv_float(x, *wt.conv("decoder.conv_in"), pad=1)

    x = _resnet_float(wt, x, "decoder.mid.block_1", 512, 512)
    g, bb = wt.vec("decoder.mid.attn_1.norm")
    x = attn_float(x, wt, g, bb)
    x = _resnet_float(wt, x, "decoder.mid.block_2", 512, 512)

    for blk_idx, (in_c, out_c) in enumerate([(512, 512), (512, 512), (512, 256), (256, 128)]):
        name = f"decoder.up.{3 - blk_idx}"
        for r in range(3):
            x = _resnet_float(wt, x, f"{name}.block.{r}", in_c if r == 0 else out_c, out_c)
        if blk_idx < 3:
            x = conv_float(upsample(x), *wt.conv(f"{name}.upsample.conv"), pad=1)

    g, bb = wt.vec("decoder.norm_out")
    x = silu_f32(gn_float(x, g, bb))
    x = conv_float(x, *wt.conv("decoder.conv_out"), pad=1)
    return np.clip(x / 2 + 0.5, 0, 1)


def decoder_fx(wt: Weights, z, engine, scale_factor=1.0, chunk=1024):
    """定点版：全部卷积/注意力走 32 路引擎。返回 (H,W,3) float32 in [0,1]。"""
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 4:
        z = z[0]
    x = fx.to_fx(np.transpose(z / scale_factor, (1, 2, 0)).astype(np.float64))

    cache = {}

    def wts(name):
        if name not in cache:
            w, b = wt.conv(name)
            wq, sw = _wq(w)
            cache[name] = (wq, sw, fx.to_fx(b.astype(np.float64)))
        return cache[name]

    def run(name, x, pad):
        wq, sw, bq = wts(name)
        return conv_fx(x, wq, sw, bq, engine, pad=pad, chunk=chunk)

    def resnet(x, prefix, in_c, out_c):
        g1, b1 = wt.vec(prefix + ".norm1")
        h_ = run(prefix + ".conv1", silu_fx(gn_fx(x, fx.to_fx(g1), fx.to_fx(b1))), 1)
        g2, b2 = wt.vec(prefix + ".norm2")
        h_ = run(prefix + ".conv2", silu_fx(gn_fx(h_, fx.to_fx(g2), fx.to_fx(b2))), 1)
        sc = run(prefix + ".nin_shortcut", x, 0) if in_c != out_c else x
        return fx.sat32(h_ + sc)

    x = run("post_quant_conv", x, 0)
    x = run("decoder.conv_in", x, 1)

    x = resnet(x, "decoder.mid.block_1", 512, 512)
    g, bb = wt.vec("decoder.mid.attn_1.norm")
    x = attn_fx(x, wt, fx.to_fx(g), fx.to_fx(bb), engine)
    x = resnet(x, "decoder.mid.block_2", 512, 512)

    for blk_idx, (in_c, out_c) in enumerate([(512, 512), (512, 512), (512, 256), (256, 128)]):
        name = f"decoder.up.{3 - blk_idx}"
        for r in range(3):
            x = resnet(x, f"{name}.block.{r}", in_c if r == 0 else out_c, out_c)
        if blk_idx < 3:
            x = run(f"{name}.upsample.conv", upsample(x), 1)

    g, bb = wt.vec("decoder.norm_out")
    x = silu_fx(gn_fx(x, fx.to_fx(g), fx.to_fx(bb)))
    x = run("decoder.conv_out", x, 1)
    return np.clip(fx.from_fx(x) / 2 + 0.5, 0, 1)
