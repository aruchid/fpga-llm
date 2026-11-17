"""SD-Turbo UNet 的 torch/CUDA 版：算子语义与 sd_unet_fx.py 完全一致（位精确）。

激活值放在 GPU 上（int64 Q16），权重 int8 常驻显存，卷积走 _int_mm。
"""

from __future__ import annotations

import numpy as np
import torch

import fx_torch as ft
from fx_torch import dev
from sd_unet import timestep_embedding

HEADS = lambda c: max(1, c // 64)


class TWeights:
    """权重按需量化到 GPU：int8 + SW(Q24) + bias(Q16)。"""

    def __init__(self, wt, device=None):
        self.wt = wt
        self.device = dev(device)
        self._c = {}

    def conv(self, name):
        key = "c:" + name
        if key not in self._c:
            import mvm32
            import sd_vae
            w, b = self.wt.conv(name)
            wq, sw = mvm32.quantize_weight_rows(sd_vae._w2d(w))
            self._c[key] = (torch.from_numpy(wq).to(self.device),
                            torch.from_numpy(sw).to(self.device),
                            ft.to_fx(b.astype(np.float64), self.device))
        return self._c[key]

    def lin(self, name):
        key = "l:" + name
        if key not in self._c:
            import mvm32
            w, b = self.wt.linear(name)
            wq, sw = mvm32.quantize_weight_rows(w)
            self._c[key] = (torch.from_numpy(wq).to(self.device),
                            torch.from_numpy(sw).to(self.device),
                            ft.to_fx(b.astype(np.float64), self.device))
        return self._c[key]

    def gn(self, name):
        key = "n:" + name
        if key not in self._c:
            g, b = self.wt.raw(name + ".weight"), self.wt.raw(name + ".bias")
            self._c[key] = (ft.to_fx(g.astype(np.float64), self.device),
                            ft.to_fx(b.astype(np.float64), self.device))
        return self._c[key]


def conv(tw, name, x, pad=1, stride=1):
    wq, sw, bq = tw.conv(name)
    return ft.conv_fx(x, wq, sw, bq, pad=pad, stride=stride)


def lin(tw, name, x):
    wq, sw, bq = tw.lin(name)
    return ft.linear_fx(x, wq, sw, bq, blocks=True)


def resblock(tw, x, emb, name, in_c, out_c):
    g1, b1 = tw.gn(name + ".in_layers.0")
    h = conv(tw, name + ".in_layers.2", ft.silu_fx(ft.groupnorm_fx(x.reshape(-1, x.shape[-1]), g1, b1, 32)).reshape(x.shape))
    e = lin(tw, name + ".emb_layers.1", emb.reshape(1, -1))[0]
    h = ft.sat32(h + e[None, None, :])
    g2, b2 = tw.gn(name + ".out_layers.0")
    h = conv(tw, name + ".out_layers.3", ft.silu_fx(ft.groupnorm_fx(h.reshape(-1, h.shape[-1]), g2, b2, 32)).reshape(h.shape))
    sc = conv(tw, name + ".skip_connection", x, pad=0) if in_c != out_c else x
    return ft.sat32(h + sc)


def _heads(x, n):
    npos, c = x.shape
    return x.reshape(npos, n, c // n).transpose(0, 1)


def _merge(x):
    h, n, d = x.shape
    return x.transpose(0, 1).reshape(n, h * d)


def _attn(q, k, v, qchunk: int = 512):
    """q,k: (H,N,D) int64 Q16；v: (H,M,D)。返回 (H,N,D) Q16。

    按查询分块：64×64 latent 的自注意力有 4096 个位置，整块算会吃掉几百 MB 显存。
    """
    nh, n, d = q.shape
    out = torch.empty((nh, n, d), dtype=torch.int64, device=q.device)
    scale_q16 = ft.to_fx(1.0 / float(np.sqrt(float(d))), q.device)
    for h in range(nh):
        kh, ks = ft.quantize_shift_batch(k[h])
        # v 沿序列收缩，量化尺度按输出维度（列）
        vq, vs = ft.quantize_shift_batch(v[h].T.contiguous())
        qh = ft.rshift_round(q[h] * scale_q16, 16)
        for r0 in range(0, n, qchunk):
            r1 = min(r0 + qchunk, n)
            qq, qs = ft.quantize_shift_batch(qh[r0:r1])
            dots = ft._imm(kh, qq.T)                              # (m, rows)
            sc = ft.rshift_round(dots.to(torch.int64), 32 - qs[None, :] - ks[:, None]).T
            p = ft.softmax_fx(sc, axis=1)
            pq, ps = ft.quantize_shift_batch(p)                   # 概率也量化成 int8
            pv = ft._imm(pq, vq.T.contiguous())                   # (rows, d)
            out[h, r0:r1] = ft.rshift_round(pv.to(torch.int64), 16 - ps[:, None] - vs[None, :])
    return out


def spatial_transformer(tw, x, context, name, ch):
    h, w, c = x.shape
    n = h * w
    nh = HEADS(c)
    g, b = tw.gn(name + ".norm")
    hn = ft.sat32(lin(tw, name + ".proj_in", ft.groupnorm_fx(x.reshape(n, c), g, b, 32)))

    tb = name + ".transformer_blocks.0"
    g1, b1 = tw.gn(tb + ".norm1")
    a = ft.groupnorm_fx(hn, g1, b1, 32)
    q = _heads(ft.sat32(lin(tw, tb + ".attn1.to_q", a)), nh)
    k = _heads(ft.sat32(lin(tw, tb + ".attn1.to_k", a)), nh)
    v = _heads(ft.sat32(lin(tw, tb + ".attn1.to_v", a)), nh)
    hn = ft.sat32(hn + lin(tw, tb + ".attn1.to_out.0", _merge(_attn(q, k, v))))

    g2, b2 = tw.gn(tb + ".norm2")
    a2 = ft.groupnorm_fx(hn, g2, b2, 32)
    q2 = _heads(ft.sat32(lin(tw, tb + ".attn2.to_q", a2)), nh)
    k2 = _heads(ft.sat32(lin(tw, tb + ".attn2.to_k", context)), nh)
    v2 = _heads(ft.sat32(lin(tw, tb + ".attn2.to_v", context)), nh)
    hn = ft.sat32(hn + lin(tw, tb + ".attn2.to_out.0", _merge(_attn(q2, k2, v2))))

    g3, b3 = tw.gn(tb + ".norm3")
    a3 = ft.groupnorm_fx(hn, g3, b3, 32)
    pr = ft.sat32(lin(tw, tb + ".ff.net.0.proj", a3))
    half = pr.shape[1] // 2
    gelu = _gelu_fx(pr[:, half:])
    ff = ft.sat32(ft.rshift_round(pr[:, :half] * gelu, 16))
    hn = ft.sat32(hn + lin(tw, tb + ".ff.net.2", ff))

    out = ft.sat32(lin(tw, name + ".proj_out", hn)).reshape(h, w, c)
    return ft.sat32(x + out)


_GELU_LUT = {}


def _gelu_fx(x):
    """GELU 查表（与 sd_clip.gelu_fx 同一张表）。"""
    from sd_clip import _GELU_X, _GELU_Y
    key = str(x.device)
    if key not in _GELU_LUT:
        _GELU_LUT[key] = (torch.tensor(_GELU_X).to(x.device),
                          torch.tensor(_GELU_Y).to(x.device))
    xs, ys = _GELU_LUT[key]
    step = xs[1] - xs[0]
    pos = ((x.double() / 65536.0 - xs[0]) / step).clamp(0, len(xs) - 1.0001)
    idx = pos.floor().to(torch.int64)
    frac = torch.round((pos - idx) * 65536).to(torch.int64)
    nxt = ys[torch.clamp(idx + 1, max=len(xs) - 1)]
    return ys[idx] + ft.rshift_round((nxt - ys[idx]) * frac, 16)


def forward(tw, x, t, context):
    """x: (H,W,4) int64 Q16；context: (77,1024) int64 Q16。"""
    temb = ft.to_fx(timestep_embedding(t).astype(np.float64), tw.device)
    emb = ft.sat32(lin(tw, "time_embed.0", temb.reshape(1, -1))[0])
    emb = ft.sat32(lin(tw, "time_embed.2", ft.silu_fx(emb).reshape(1, -1))[0])

    hs = []
    hcur = conv(tw, "input_blocks.0.0", x)
    hs.append(hcur)

    chans = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    for ib in range(1, 12):
        if ib in (3, 6, 9):
            hcur = conv(tw, f"input_blocks.{ib}.0.op", hcur, stride=2)
            hs.append(hcur)
            continue
        in_c = hcur.shape[-1]
        hcur = resblock(tw, hcur, emb, f"input_blocks.{ib}.0", in_c, chans[ib])
        if ib in (1, 2, 4, 5, 7, 8):
            hcur = spatial_transformer(tw, hcur, context, f"input_blocks.{ib}.1", chans[ib])
        hs.append(hcur)

    hcur = resblock(tw, hcur, emb, "middle_block.0", 1280, 1280)
    hcur = spatial_transformer(tw, hcur, context, "middle_block.1", 1280)
    hcur = resblock(tw, hcur, emb, "middle_block.2", 1280, 1280)

    up_chans = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    for ob in range(12):
        skip = hs.pop()
        hcur = ft.sat32(torch.cat([hcur, skip], dim=-1))
        in_c = hcur.shape[-1]
        out_c = up_chans[ob]
        hcur = resblock(tw, hcur, emb, f"output_blocks.{ob}.0", in_c, out_c)
        if ob in (3, 4, 5, 6, 7, 8, 9, 10, 11):
            hcur = spatial_transformer(tw, hcur, context, f"output_blocks.{ob}.1", out_c)
        if ob in (2, 5, 8):
            idx = "2" if ob > 2 else "1"
            hcur = conv(tw, f"output_blocks.{ob}.{idx}.conv",
                        torch.repeat_interleave(torch.repeat_interleave(hcur, 2, dim=0), 2, dim=1))

    g, b = tw.gn("out.0")
    hcur = ft.silu_fx(ft.groupnorm_fx(hcur.reshape(-1, hcur.shape[-1]), g, b, 32).reshape(hcur.shape))
    return conv(tw, "out.2", hcur)
