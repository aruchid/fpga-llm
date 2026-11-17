"""SD-Turbo UNet（numpy 浮点版）+ Euler 采样。

先只做浮点，用来验证网络实现；定点版在 sd_unet_fx.py 里复用同一套结构。
权重前缀：model.diffusion_model.*（ComfyUI 命名，和 ldm 完全对应）
"""

from __future__ import annotations

import numpy as np

from sd_vae import conv_float, gn_float, silu_f32, upsample

PREFIX = "model.diffusion_model"
TIME_EMB_DIM = 320
TIME_EMB_MAX = 10000


class UNetWeights:
    """cache=False 时每次从 safetensors 读，峰值内存只有单层；大模型上 3.5GB 缓存会顶爆内存。"""

    def __init__(self, path, prefix=PREFIX, cache=True):
        self.path = path
        self.prefix = prefix
        self._h = None
        self._c = {}
        self.cache = cache

    def forget(self, name):
        self._c.pop(f"{self.prefix}.{name}.weight", None)
        self._c.pop(f"{self.prefix}.{name}.bias", None)

    def _open(self):
        if self._h is None:
            from safetensors import safe_open
            self._h = safe_open(self.path, framework="numpy")
        return self._h

    def raw(self, name):
        if name in self._c:
            return self._c[name]
        t = self._open().get_tensor(f"{self.prefix}.{name}")
        if self.cache:
            self._c[name] = t
        return t

    def _bias_or_zero(self, name):
        """q/k/v 这类层没有 bias（只有 to_out 有），缺省补 0。"""
        try:
            return self.raw(name + ".bias").astype(np.float32)
        except Exception:
            w = self.raw(name + ".weight")
            return np.zeros(w.shape[0], dtype=np.float32)

    def conv(self, name):
        return self.raw(name + ".weight").astype(np.float32), self._bias_or_zero(name)

    def linear(self, name):
        return self.raw(name + ".weight").astype(np.float32), self._bias_or_zero(name)


# ---------------------------------------------------------------- 基础块
def timestep_embedding(t: float, dim: int = TIME_EMB_DIM) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-np.log(TIME_EMB_MAX) * np.arange(half, dtype=np.float64) / half)
    args = float(t) * freqs
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)


def linear(x, w, b):
    return x @ w.T + b


def resblock(wt: UNetWeights, x, emb, name, in_c, out_c):
    g1, b1 = wt.raw(name + ".in_layers.0.weight"), wt.raw(name + ".in_layers.0.bias")
    h = conv_float(silu_f32(gn_float(x, g1, b1)), *wt.conv(name + ".in_layers.2"), pad=1)
    ew, eb = wt.linear(name + ".emb_layers.1")
    h = h + linear(silu_f32(emb), ew, eb)[None, None, :]
    g2, b2 = wt.raw(name + ".out_layers.0.weight"), wt.raw(name + ".out_layers.0.bias")
    h = conv_float(silu_f32(gn_float(h, g2, b2)), *wt.conv(name + ".out_layers.3"), pad=1)
    if in_c != out_c:
        h = h + conv_float(x, *wt.conv(name + ".skip_connection"), pad=0)
    else:
        h = h + x
    return h


def _gelu(x):
    from sd_clip import gelu
    return gelu(x)


def _attn_heads(x, n_heads):
    """(H*W, C) → (n_heads, H*W, C/n_heads)"""
    n, c = x.shape
    return x.reshape(n, n_heads, c // n_heads).transpose(1, 0, 2)


def _merge_heads(x):
    h, n, d = x.shape
    return x.transpose(1, 0, 2).reshape(n, h * d)


def sdpa(q, k, v):
    """标准注意力：q (H, N, D)，k (H, M, D)，v (H, M, D)。"""
    sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(np.float32(q.shape[-1]))
    sc -= sc.max(axis=-1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def spatial_transformer(wt: UNetWeights, x, context, name, ch, n_heads=8, ctx_dim=1024):
    """SpatialTransformer（use_linear_in_transformer=True → proj 是 Linear）。"""
    h, w, c = x.shape
    n = h * w
    g, b = wt.raw(name + ".norm.weight"), wt.raw(name + ".norm.bias")
    xn = gn_float(x, g, b).reshape(n, c)
    pw, pb = wt.linear(name + ".proj_in")
    hh = linear(xn, pw, pb)

    tb = name + ".transformer_blocks.0"
    # ---- 自注意力
    g1, b1 = wt.raw(tb + ".norm1.weight"), wt.raw(tb + ".norm1.bias")
    a = _attn_heads(gn_float(hh.reshape(1, n, c), g1, b1).reshape(n, c), n_heads)
    wq, bq = wt.linear(tb + ".attn1.to_q")
    wk, bk = wt.linear(tb + ".attn1.to_k")
    wv, bv = wt.linear(tb + ".attn1.to_v")
    wo, bo = wt.linear(tb + ".attn1.to_out.0")
    q = _attn_heads(linear(hh, wq, bq), n_heads)
    k = _attn_heads(linear(hh, wk, bk), n_heads)
    v = _attn_heads(linear(hh, wv, bv), n_heads)
    o = _merge_heads(sdpa(q, k, v))
    hh = hh + linear(o, wo, bo)

    # ---- 交叉注意力（context 来自 CLIP）
    g2, b2 = wt.raw(tb + ".norm2.weight"), wt.raw(tb + ".norm2.bias")
    xn2 = gn_float(hh.reshape(1, n, c), g2, b2).reshape(n, c)
    wq, bq = wt.linear(tb + ".attn2.to_q")
    wk, bk = wt.linear(tb + ".attn2.to_k")
    wv, bv = wt.linear(tb + ".attn2.to_v")
    wo, bo = wt.linear(tb + ".attn2.to_out.0")
    q = _attn_heads(linear(xn2, wq, bq), n_heads)
    k = _attn_heads(linear(context, wk, bk), n_heads)
    v = _attn_heads(linear(context, wv, bv), n_heads)
    o = _merge_heads(sdpa(q, k, v))
    hh = hh + linear(o, wo, bo)

    # ---- GEGLU 前馈
    g3, b3 = wt.raw(tb + ".norm3.weight"), wt.raw(tb + ".norm3.bias")
    xn3 = gn_float(hh.reshape(1, n, c), g3, b3).reshape(n, c)
    w0, b0 = wt.linear(tb + ".ff.net.0.proj")
    w2, b2f = wt.linear(tb + ".ff.net.2")
    pr = linear(xn3, w0, b0)
    a, gate = np.split(pr, 2, axis=-1)
    hh = hh + linear(a * _gelu(gate), w2, b2f)

    ow, ob = wt.linear(name + ".proj_out")
    out = linear(hh, ow, ob).reshape(h, w, c)
    return x + out


def forward(wt: UNetWeights, x, t, context):
    """x: (H,W,4)，t: 时间步（0-999），context: (77,1024) → 返回 (H,W,4)。"""
    temb = timestep_embedding(t)
    w0, b0 = wt.linear("time_embed.0")
    w2, b2 = wt.linear("time_embed.2")
    emb = linear(silu_f32(linear(temb, w0, b0)), w2, b2)               # (1280,)

    hs = []
    hcur = conv_float(x, *wt.conv("input_blocks.0.0"), pad=1)
    hs.append(hcur)

    chans = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    heads = lambda c: max(1, c // 64)
    for ib in range(1, 12):
        if ib in (3, 6, 9):     # 纯下采样块（不是 resblock）
            hcur = conv_float(hcur, *wt.conv(f"input_blocks.{ib}.0.op"), pad=1, stride=2)
            hs.append(hcur)
            continue
        in_c = hcur.shape[-1]
        hcur = resblock(wt, hcur, emb, f"input_blocks.{ib}.0", in_c, chans[ib])
        out_c = chans[ib]
        # 注意力模块编号：1,2,4,5,7,8 有
        if ib in (1, 2, 4, 5, 7, 8):
            hcur = spatial_transformer(wt, hcur, context, f"input_blocks.{ib}.1", out_c,
                                       n_heads=heads(out_c))
        hs.append(hcur)

    # mid
    hcur = resblock(wt, hcur, emb, "middle_block.0", 1280, 1280)
    hcur = spatial_transformer(wt, hcur, context, "middle_block.1", 1280, n_heads=heads(1280))
    hcur = resblock(wt, hcur, emb, "middle_block.2", 1280, 1280)

    # up
    up_chans = [
        (1280, 1280), (1280, 1280), (1280, 1280),
        (1280, 1280), (1280, 1280), (1280, 1280),
        (1280, 640), (640, 640), (640, 640),
        (640, 320), (320, 320), (320, 320),
    ]
    for ob in range(12):
        skip = hs.pop()
        hcur = np.concatenate([hcur, skip], axis=-1)
        in_c = hcur.shape[-1]
        out_c = up_chans[ob][1]
        hcur = resblock(wt, hcur, emb, f"output_blocks.{ob}.0", in_c, out_c)
        if ob in (3, 4, 5, 6, 7, 8, 9, 10, 11):
            hcur = spatial_transformer(wt, hcur, context, f"output_blocks.{ob}.1", out_c,
                                       n_heads=heads(out_c))
        if ob in (2, 5, 8):     # 上采样；有注意力的块里 upsample 排在第 2 个，否则第 1 个
            idx = "2" if ob in (3, 4, 5, 6, 7, 8, 9, 10, 11) else "1"
            hcur = conv_float(upsample(hcur), *wt.conv(f"output_blocks.{ob}.{idx}.conv"), pad=1)

    g, b = wt.raw("out.0.weight"), wt.raw("out.0.bias")
    hcur = silu_f32(gn_float(hcur, g, b))
    return conv_float(hcur, *wt.conv("out.2"), pad=1)
