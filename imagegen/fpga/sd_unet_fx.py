"""SD-Turbo UNet 的定点版：所有矩阵乘走 32 路引擎（Q16.16 + int8 权重）。

结构与 sd_unet.py 一一对应，只是把 conv/linear 换成硬件路径：
- 卷积：每像素一个 s（9C 长的输入向量）
- 线性层/注意力投影：按 32 元素分块量化（残差流里有离群通道）
- 打分与 PV：int8 点积进矩阵单元，softmax 在 Q16 里做
"""

from __future__ import annotations

import numpy as np

import fx
import mvm32
import sd_vae as V
from sd_clip import gelu_fx
from sd_unet import timestep_embedding

HEADS = lambda c: max(1, c // 64)


class QuantCache:
    def __init__(self, wt):
        self.wt = wt
        self._c = {}

    def conv(self, name):
        if name not in self._c:
            w, b = self.wt.conv(name)
            wq, sw = mvm32.quantize_weight_rows(V._w2d(w))
            self._c[name] = (wq, sw, fx.to_fx(b.astype(np.float64)))
            self.wt.forget(name)
        return self._c[name]

    def lin(self, name):
        if name not in self._c:
            w, b = self.wt.linear(name)
            wq, sw = mvm32.quantize_weight_rows(w)
            self._c[name] = (wq, sw, fx.to_fx(b.astype(np.float64)))
            self.wt.forget(name)
        return self._c[name]

    def gn(self, name):
        g, b = self.wt.raw(name + ".weight"), self.wt.raw(name + ".bias")
        return fx.to_fx(g.astype(np.float64)), fx.to_fx(b.astype(np.float64))


def conv(qc, name, x, engine, pad=1, stride=1, blockq=False):
    wq, sw, bq = qc.conv(name)
    return V.conv_fx(x, wq, sw, bq, engine, pad=pad, stride=stride, blockq=blockq)


def lin(qc, name, x, engine):
    """x: (n, K) Q16 → (n, O) Q16，激活按 32 分块量化。"""
    wq, sw, bq = qc.lin(name)
    q, sb = mvm32.quantize_blocks(x)
    y = engine.gemm_blocks(wq, sw, np.ascontiguousarray(q.T), sb.T).T
    return y + bq[None, :]


def resblock(qc, x, emb, name, in_c, out_c, engine):
    g1, b1 = qc.gn(name + ".in_layers.0")
    h = conv(qc, name + ".in_layers.2", fx.silu_fx(V.gn_fx(x, g1, b1)), engine)
    e = fx.sat32(lin(qc, name + ".emb_layers.1", emb.reshape(1, -1), engine))[0]
    h = fx.sat32(h + e[None, None, :])
    g2, b2 = qc.gn(name + ".out_layers.0")
    h = conv(qc, name + ".out_layers.3", fx.silu_fx(V.gn_fx(h, g2, b2)), engine)
    if in_c != out_c:
        h = fx.sat32(h + conv(qc, name + ".skip_connection", x, engine, pad=0))
    else:
        h = fx.sat32(h + x)
    return h


def _heads(x, n):
    npos, c = x.shape
    return x.reshape(npos, n, c // n).transpose(1, 0, 2)


def _merge(x):
    h, n, d = x.shape
    return x.transpose(1, 0, 2).reshape(n, h * d)


def _attn_q16(q, k, v, engine, qchunk=256):
    """q,k: (H,N,D) Q16；v: (H,M,D) Q16 → (H,N,D) Q16。"""
    nh, n, d = q.shape
    m = k.shape[1]
    out = np.empty((nh, n, d), dtype=np.int64)
    for h in range(nh):
        kh_, ks = V.quantize_shift_batch(k[h])
        # v 沿序列收缩，量化尺度必须按输出维度（列）
        vq, vs = V.quantize_shift_batch(np.ascontiguousarray(v[h].T))
        # 1/√d 折进 q（与 CLIP/浮点版一致），否则 softmax 过尖
        scale_q16 = fx.to_fx(1.0 / np.sqrt(float(d)))
        qh = fx.rshift_round(q[h] * scale_q16, 16)
        for r0 in range(0, n, qchunk):
            r1 = min(r0 + qchunk, n)
            qq, qs = V.quantize_shift_batch(qh[r0:r1])
            dots = engine.dot_i8(kh_, np.ascontiguousarray(qq.T))            # (m, rows)
            sc = fx.rshift_round(dots.astype(np.int64), 32 - qs[None, :] - ks[:, None]).T
            p = fx.softmax_fx(sc, axis=1)
            pq, ps = V.quantize_shift_batch(p)                             # 概率也进矩阵单元
            pv = engine.dot_i8(pq, np.ascontiguousarray(vq.T))               # (rows, d)
            out[h, r0:r1] = fx.rshift_round(pv.astype(np.int64), 16 - ps[:, None] - vs[None, :])
    return out


def spatial_transformer(qc, x, context, name, ch, engine):
    h, w, c = x.shape
    n = h * w
    nh = HEADS(c)
    g, b = qc.gn(name + ".norm")
    hn = fx.sat32(lin(qc, name + ".proj_in", V.gn_fx(x, g, b).reshape(n, c), engine))

    tb = name + ".transformer_blocks.0"
    # 自注意力
    g1, b1 = qc.gn(tb + ".norm1")
    a = V.gn_fx(hn.reshape(1, n, c), g1, b1).reshape(n, c)
    q = _heads(fx.sat32(lin(qc, tb + ".attn1.to_q", a, engine)), nh)
    k = _heads(fx.sat32(lin(qc, tb + ".attn1.to_k", a, engine)), nh)
    v = _heads(fx.sat32(lin(qc, tb + ".attn1.to_v", a, engine)), nh)
    o = _merge(_attn_q16(q, k, v, engine))
    hn = fx.sat32(hn + lin(qc, tb + ".attn1.to_out.0", o, engine))

    # 交叉注意力
    g2, b2 = qc.gn(tb + ".norm2")
    a2 = V.gn_fx(hn.reshape(1, n, c), g2, b2).reshape(n, c)
    q2 = _heads(fx.sat32(lin(qc, tb + ".attn2.to_q", a2, engine)), nh)
    k2 = _heads(fx.sat32(lin(qc, tb + ".attn2.to_k", context, engine)), nh)
    v2 = _heads(fx.sat32(lin(qc, tb + ".attn2.to_v", context, engine)), nh)
    o2 = _merge(_attn_q16(q2, k2, v2, engine))
    hn = fx.sat32(hn + lin(qc, tb + ".attn2.to_out.0", o2, engine))

    # GEGLU 前馈
    g3, b3 = qc.gn(tb + ".norm3")
    a3 = V.gn_fx(hn.reshape(1, n, c), g3, b3).reshape(n, c)
    pr = fx.sat32(lin(qc, tb + ".ff.net.0.proj", a3, engine))
    half = pr.shape[1] // 2
    ff = fx.sat32(fx.rshift_round(pr[:, :half] * gelu_fx(pr[:, half:]), 16))
    hn = fx.sat32(hn + lin(qc, tb + ".ff.net.2", ff, engine))

    out = fx.sat32(lin(qc, name + ".proj_out", hn, engine)).reshape(h, w, c)
    return fx.sat32(x + out)


def forward(wt, x, t, context, engine, qc=None):
    """x: (H,W,4) Q16；context: (77,1024) Q16 → (H,W,4) Q16。"""
    qc = qc or QuantCache(wt)
    temb = fx.to_fx(timestep_embedding(t).astype(np.float64))
    emb = fx.sat32(lin(qc, "time_embed.0", temb.reshape(1, -1), engine))[0]
    emb = fx.sat32(lin(qc, "time_embed.2", fx.silu_fx(emb).reshape(1, -1), engine))[0]

    hs = []
    hcur = conv(qc, "input_blocks.0.0", x, engine)
    hs.append(hcur)

    chans = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    for ib in range(1, 12):
        if ib in (3, 6, 9):
            hcur = conv(qc, f"input_blocks.{ib}.0.op", hcur, engine, stride=2)
            hs.append(hcur)
            continue
        in_c = hcur.shape[-1]
        hcur = resblock(qc, hcur, emb, f"input_blocks.{ib}.0", in_c, chans[ib], engine)
        if ib in (1, 2, 4, 5, 7, 8):
            hcur = spatial_transformer(qc, hcur, context, f"input_blocks.{ib}.1", chans[ib], engine)
        hs.append(hcur)

    hcur = resblock(qc, hcur, emb, "middle_block.0", 1280, 1280, engine)
    hcur = spatial_transformer(qc, hcur, context, "middle_block.1", 1280, engine)
    hcur = resblock(qc, hcur, emb, "middle_block.2", 1280, 1280, engine)

    up_chans = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    for ob in range(12):
        skip = hs.pop()
        hcur = fx.sat32(np.concatenate([hcur, skip], axis=-1))
        in_c = hcur.shape[-1]
        out_c = up_chans[ob]
        hcur = resblock(qc, hcur, emb, f"output_blocks.{ob}.0", in_c, out_c, engine)
        if ob in (3, 4, 5, 6, 7, 8, 9, 10, 11):
            hcur = spatial_transformer(qc, hcur, context, f"output_blocks.{ob}.1", out_c, engine)
        if ob in (2, 5, 8):
            idx = "2" if ob > 2 else "1"
            hcur = conv(qc, f"output_blocks.{ob}.{idx}.conv", V.upsample(hcur), engine)

    g, b = qc.gn("out.0")
    hcur = fx.silu_fx(V.gn_fx(hcur, g, b))
    return conv(qc, "out.2", hcur, engine)
