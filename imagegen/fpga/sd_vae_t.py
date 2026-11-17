"""SD VAE 解码器的 torch/CUDA 版：与 sd_vae.decoder_fx 位精确一致。

激活 int64 Q16 在 GPU，卷积走 _int_mm；mid 的单头自注意力同样进矩阵单元。
"""

from __future__ import annotations

import numpy as np
import torch

import fx_torch as ft
import mvm32
import sd_vae
from fx_torch import dev

VAE_PREFIX = "first_stage_model"


class TWeights:
    def __init__(self, wt, device=None):
        self.wt = wt
        self.device = dev(device)
        self._c = {}

    def conv(self, name):
        if name not in self._c:
            w, b = self.wt.conv(name)
            wq, sw = mvm32.quantize_weight_rows(sd_vae._w2d(w))
            self._c[name] = (torch.from_numpy(wq).to(self.device),
                             torch.from_numpy(sw).to(self.device),
                             ft.to_fx(b.astype(np.float64), self.device))
        return self._c[name]

    def gn(self, name):
        key = "n:" + name
        if key not in self._c:
            g, b = self.wt.raw(name + ".weight"), self.wt.raw(name + ".bias")
            self._c[key] = (ft.to_fx(g.astype(np.float64), self.device),
                            ft.to_fx(b.astype(np.float64), self.device))
        return self._c[key]


def conv(tw, name, x, pad=1):
    wq, sw, bq = tw.conv(name)
    return ft.conv_fx(x, wq, sw, bq, pad=pad)


def gn(tw, name, x):
    g, b = tw.gn(name)
    return ft.groupnorm_fx(x.reshape(-1, x.shape[-1]), g, b, 32).reshape(x.shape)


def resnet(tw, x, prefix, in_c, out_c):
    h = conv(tw, prefix + ".conv1", ft.silu_fx(gn(tw, prefix + ".norm1", x)))
    h = conv(tw, prefix + ".conv2", ft.silu_fx(gn(tw, prefix + ".norm2", h)))
    sc = conv(tw, prefix + ".nin_shortcut", x, pad=0) if in_c != out_c else x
    return ft.sat32(h + sc)


def attn(tw, x, prefix="decoder.mid.attn_1"):
    """单头自注意力：q/k/v/proj 都是 1×1 卷积（矩阵单元里的 [C,C] 乘）。"""
    h, w, c = x.shape
    n = h * w
    xn = gn(tw, prefix + ".norm", x).reshape(1, n, c)

    def proj1x1(name):
        return conv(tw, f"{prefix}.{name}", xn, pad=0).reshape(n, c)

    q = proj1x1("q")
    k = proj1x1("k")
    v = proj1x1("v")
    scale_q16 = ft.to_fx(1.0 / float(np.sqrt(float(c))), q.device)
    q = ft.rshift_round(q * scale_q16, 16)

    kq, ks = ft.quantize_shift_batch(k)
    vq, vs = ft.quantize_shift_batch(v.T.contiguous())          # 按输出通道量化
    out = torch.empty((n, c), dtype=torch.int64, device=q.device)
    for r0 in range(0, n, 512):
        r1 = min(r0 + 512, n)
        qq, qs = ft.quantize_shift_batch(q[r0:r1])
        dots = ft._imm(kq, qq.T.contiguous())                    # (n_keys, m_queries)
        sc = ft.rshift_round(dots.to(torch.int64), 32 - qs[None, :] - ks[:, None])
        p = ft.softmax_fx(sc.T, axis=1)                          # (m, n) 行=query
        pq, ps = ft.quantize_shift_batch(p)
        pv = ft._imm(pq, vq.T.contiguous())                      # (m, c)
        out[r0:r1] = ft.rshift_round(pv.to(torch.int64), 16 - ps[:, None] - vs[None, :])
    o = conv(tw, f"{prefix}.proj_out", out.reshape(1, n, c), pad=0)
    return ft.sat32(x + o.reshape(h, w, c))


def upsample(x):
    return torch.repeat_interleave(torch.repeat_interleave(x, 2, dim=0), 2, dim=1)


def decoder(tw: TWeights, z, scale_factor=1.0):
    """z: (1,4,H,W) 或 (4,H,W) float → (H*8, W*8, 3) float32 in [0,1]。"""
    z = np.asarray(z, dtype=np.float32).squeeze()
    if z.ndim == 4:
        z = z[0]
    x = ft.to_fx(np.transpose(z / scale_factor, (1, 2, 0)).astype(np.float64), tw.device)

    x = conv(tw, "post_quant_conv", x, pad=0)
    x = conv(tw, "decoder.conv_in", x)
    x = resnet(tw, x, "decoder.mid.block_1", 512, 512)
    x = attn(tw, x)
    x = resnet(tw, x, "decoder.mid.block_2", 512, 512)

    for blk_idx, (in_c, out_c) in enumerate([(512, 512), (512, 512), (512, 256), (256, 128)]):
        name = f"decoder.up.{3 - blk_idx}"
        for r in range(3):
            x = resnet(tw, x, f"{name}.block.{r}", in_c if r == 0 else out_c, out_c)
        if blk_idx < 3:
            x = conv(tw, f"{name}.upsample.conv", upsample(x))

    x = ft.silu_fx(gn(tw, "decoder.norm_out", x))
    x = conv(tw, "decoder.conv_out", x)
    out = x.cpu().numpy().astype(np.float64) / 65536.0
    return np.clip(out / 2 + 0.5, 0, 1)
