"""INT8 版 SD-Turbo UNet：权重 per-channel int8、激活动态 int8、矩阵乘 int8→int32。

非矩阵运算（GroupNorm/SiLU/softmax/注意力打分）保持原样。
"""

from __future__ import annotations

import numpy as np
import torch

import q8
from q8 import DEV
from sd_unet import timestep_embedding

HEADS = lambda c: max(1, c // 64)
PREFIX = "model.diffusion_model"


class W(q8.QWeights):
    def __init__(self, path):
        super().__init__(path, PREFIX)


def resblock(qw, x, emb, name, in_c, out_c, act_block=0):
    g1 = qw.raw(name + ".in_layers.0.weight").astype(np.float32)
    b1 = qw.raw(name + ".in_layers.0.bias").astype(np.float32)
    h = q8.conv(q8.silu(q8.gn(x, torch.from_numpy(g1).to(DEV), torch.from_numpy(b1).to(DEV))),
                *qw.conv(name + ".in_layers.2")[:3], *qw.conv(name + ".in_layers.2")[3],
                act_block=act_block)
    e = q8.linear(emb.reshape(1, -1), *qw.lin(name + ".emb_layers.1")[:3])[0]
    h = h + e[None, None, :]
    g2 = qw.raw(name + ".out_layers.0.weight").astype(np.float32)
    b2 = qw.raw(name + ".out_layers.0.bias").astype(np.float32)
    h = q8.conv(q8.silu(q8.gn(h, torch.from_numpy(g2).to(DEV), torch.from_numpy(b2).to(DEV))),
                *qw.conv(name + ".out_layers.3")[:3], *qw.conv(name + ".out_layers.3")[3],
                act_block=act_block)
    if in_c != out_c:
        h = h + q8.conv(x, *qw.conv(name + ".skip_connection")[:3],
                        *qw.conv(name + ".skip_connection")[3], pad=0, act_block=act_block)
    else:
        h = h + x
    return h


def _layernorm_like(qw, name, x):
    g = torch.from_numpy(qw.raw(name + ".weight").astype(np.float32)).to(DEV)
    b = torch.from_numpy(qw.raw(name + ".bias").astype(np.float32)).to(DEV)
    return q8.gn(x, g, b)


def _heads(x, n):
    npos, c = x.shape
    return x.reshape(npos, n, c // n).transpose(0, 1)


def _merge(x):
    h, n, d = x.shape
    return x.transpose(0, 1).reshape(n, h * d)


def spatial_transformer(qw, x, context, name, ch, act_block=0):
    h, w, c = x.shape
    n = h * w
    nh = HEADS(c)
    tb = name + ".transformer_blocks.0"
    hn = q8.linear(_layernorm_like(qw, name + ".norm", x).reshape(n, c),
                   *qw.lin(name + ".proj_in")[:3], act_block=act_block)

    a = _layernorm_like(qw, tb + ".norm1", hn.reshape(1, n, c)).reshape(n, c)
    q = _heads(q8.linear(a, *qw.lin(tb + ".attn1.to_q")[:3], act_block=act_block), nh)
    k = _heads(q8.linear(a, *qw.lin(tb + ".attn1.to_k")[:3], act_block=act_block), nh)
    v = _heads(q8.linear(a, *qw.lin(tb + ".attn1.to_v")[:3], act_block=act_block), nh)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    hn = hn + q8.linear(_merge(o), *qw.lin(tb + ".attn1.to_out.0")[:3], act_block=act_block)

    a2 = _layernorm_like(qw, tb + ".norm2", hn.reshape(1, n, c)).reshape(n, c)
    q2 = _heads(q8.linear(a2, *qw.lin(tb + ".attn2.to_q")[:3], act_block=act_block), nh)
    k2 = _heads(q8.linear(context, *qw.lin(tb + ".attn2.to_k")[:3], act_block=act_block), nh)
    v2 = _heads(q8.linear(context, *qw.lin(tb + ".attn2.to_v")[:3], act_block=act_block), nh)
    o2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)
    hn = hn + q8.linear(_merge(o2), *qw.lin(tb + ".attn2.to_out.0")[:3], act_block=act_block)

    a3 = _layernorm_like(qw, tb + ".norm3", hn.reshape(1, n, c)).reshape(n, c)
    pr = q8.linear(a3, *qw.lin(tb + ".ff.net.0.proj")[:3], act_block=act_block)
    half = pr.shape[1] // 2
    ff = pr[:, :half] * torch.nn.functional.gelu(pr[:, half:])
    hn = hn + q8.linear(ff, *qw.lin(tb + ".ff.net.2")[:3], act_block=act_block)

    out = q8.linear(hn, *qw.lin(name + ".proj_out")[:3], act_block=act_block).reshape(h, w, c)
    return x + out


def forward(qw, x, t, context, act_block=0):
    """x: (H,W,4) float32；context: (77,1024) float32。"""
    temb = torch.from_numpy(timestep_embedding(t)).to(DEV)
    emb = q8.linear(q8.silu(q8.linear(temb.reshape(1, -1), *qw.lin("time_embed.0")[:3]))[0].reshape(1, -1),
                    *qw.lin("time_embed.2")[:3])[0]

    hs = []
    hcur = q8.conv(x, *qw.conv("input_blocks.0.0")[:3], *qw.conv("input_blocks.0.0")[3],
                   act_block=act_block)
    hs.append(hcur)
    chans = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    for ib in range(1, 12):
        if ib in (3, 6, 9):
            hcur = q8.conv(hcur, *qw.conv(f"input_blocks.{ib}.0.op")[:3],
                           *qw.conv(f"input_blocks.{ib}.0.op")[3], stride=2, act_block=act_block)
            hs.append(hcur)
            continue
        hcur = resblock(qw, hcur, emb, f"input_blocks.{ib}.0", hcur.shape[-1], chans[ib], act_block)
        if ib in (1, 2, 4, 5, 7, 8):
            hcur = spatial_transformer(qw, hcur, context, f"input_blocks.{ib}.1", chans[ib], act_block)
        hs.append(hcur)

    hcur = resblock(qw, hcur, emb, "middle_block.0", 1280, 1280, act_block)
    hcur = spatial_transformer(qw, hcur, context, "middle_block.1", 1280, act_block)
    hcur = resblock(qw, hcur, emb, "middle_block.2", 1280, 1280, act_block)

    up = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    for ob in range(12):
        skip = hs.pop()
        hcur = torch.cat([hcur, skip], dim=-1)
        hcur = resblock(qw, hcur, emb, f"output_blocks.{ob}.0", hcur.shape[-1], up[ob], act_block)
        if ob >= 3:
            hcur = spatial_transformer(qw, hcur, context, f"output_blocks.{ob}.1", up[ob], act_block)
        if ob in (2, 5, 8):
            idx = "2" if ob > 2 else "1"
            up2 = torch.repeat_interleave(torch.repeat_interleave(hcur, 2, dim=0), 2, dim=1)
            hcur = q8.conv(up2, *qw.conv(f"output_blocks.{ob}.{idx}.conv")[:3],
                           *qw.conv(f"output_blocks.{ob}.{idx}.conv")[3], act_block=act_block)

    hcur = q8.silu(_layernorm_like(qw, "out.0", hcur))
    return q8.conv(hcur, *qw.conv("out.2")[:3], *qw.conv("out.2")[3], act_block=act_block)
