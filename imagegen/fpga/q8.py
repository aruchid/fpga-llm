"""INT8 量化推理算子（W8A8）：权重 per-channel int8，激活动态 int8，累加 int32。

这一层只关心「量化后的模型能不能出好图」：
    acc = Σ w_q · x_q          （int8 × int8 → int32，正是 32 路矩阵单元的运算）
    y   = acc · (s_w · s_x) + b （反量化 + 偏置，在向量单元里做）

非矩阵运算（GroupNorm / SiLU / softmax / 注意力打分）保持原样。
"""

from __future__ import annotations

import math

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 量化
def quant_weight(w: np.ndarray, per_channel: bool = True):
    """浮点权重 (O, K) → int8 (O,K) + 每行 scale。"""
    w = np.asarray(w, dtype=np.float32)
    if per_channel:
        amax = np.max(np.abs(w), axis=1, keepdims=True)
    else:
        amax = np.full((w.shape[0], 1), np.max(np.abs(w)), dtype=np.float32)
    amax = np.maximum(amax, 1e-8)
    s = (amax / 127.0).astype(np.float32)
    q = np.clip(np.round(w / s), -127, 127).astype(np.int8)
    return q, s[:, 0]


def quant_rows(x: torch.Tensor):
    """按行（每个像素/每个 token）量化：x (n,K) → int8 (n,K) + s (n,)。

    这与 32 路矩阵单元的「每个输入向量一个移位量」一一对应，也是硬件最自然的做法。
    """
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    s = amax / 127.0
    q = torch.clamp(torch.round(x / s), -127, 127).to(torch.int8)
    return q, s[:, 0]


def quant_tensor(x: torch.Tensor):
    amax = x.abs().amax().clamp(min=1e-8)
    s = amax / 127.0
    q = torch.clamp(torch.round(x / s), -127, 127).to(torch.int8)
    return q, s


def mm8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8 (M,K) × int8 (K,N) → int32 (M,N)，K/N 不足 8 的补零。"""
    k, n = a.shape[1], b.shape[1]
    k8, n8 = (k + 7) // 8 * 8, (n + 7) // 8 * 8
    if k8 != k:
        a = torch.nn.functional.pad(a, (0, k8 - k))
        b = torch.nn.functional.pad(b, (0, 0, 0, k8 - k))
    if n8 != n:
        b = torch.nn.functional.pad(b, (0, n8 - n))
    try:
        out = torch._int_mm(a.contiguous(), b.contiguous())
    except RuntimeError:
        out = torch.round(a.double() @ b.double()).to(torch.int32)
    return out[:, :n]


# ---------------------------------------------------------------- 层
class QWeights:
    def __init__(self, path, prefix="model.diffusion_model"):
        self.path = path
        self.prefix = prefix
        self._h = None
        self._c = {}

    def _open(self):
        if self._h is None:
            from safetensors import safe_open
            self._h = safe_open(self.path, framework="numpy")
        return self._h

    def raw(self, name):
        it = self._open().get_tensor(f"{self.prefix}.{name}")
        return it

    def conv(self, name):
        key = "c:" + name
        if key not in self._c:
            w = self.raw(name + ".weight").astype(np.float32)       # (O,I,kh,kw)
            o, c, kh, kw = w.shape
            w2d = w.transpose(0, 2, 3, 1).reshape(o, kh * kw * c)
            q, s = quant_weight(w2d)
            b = self.raw(name + ".bias").astype(np.float32)
            self._c[key] = (torch.from_numpy(q).to(DEV), torch.from_numpy(s).to(DEV),
                            torch.from_numpy(b).to(DEV), (kh, kw))
        return self._c[key]

    def lin(self, name):
        key = "l:" + name
        if key not in self._c:
            try:
                w = self.raw(name + ".weight").astype(np.float32)
                b = self.raw(name + ".bias").astype(np.float32)
            except Exception:
                w = self.raw(name + ".weight").astype(np.float32)
                b = np.zeros(w.shape[0], np.float32)
            q, s = quant_weight(w)
            self._c[key] = (torch.from_numpy(q).to(DEV), torch.from_numpy(s).to(DEV),
                            torch.from_numpy(b).to(DEV), None)
        return self._c[key]


def conv(x, wq, sw, b, kh, kw, pad=1, stride=1, act_block=0):
    """x: (H,W,C) float32 → (Hout,Wout,O) float32。激活按行（每个像素）量化。"""
    h, w, c = x.shape
    xp = torch.nn.functional.pad(x, (0, 0, pad, pad, pad, pad)) if pad else x
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (w + 2 * pad - kw) // stride + 1
    rr = (torch.arange(ho, device=x.device) * stride)[:, None]
    cc = (torch.arange(wo, device=x.device) * stride)[None, :]
    cols = torch.empty((ho * wo, kh * kw * c), dtype=torch.float32, device=x.device)
    t = 0
    for i in range(kh):
        for j in range(kw):
            cols[:, t * c:(t + 1) * c] = xp[rr + i, cc + j, :].reshape(ho * wo, c)
            t += 1
    q, s = quant_rows(cols)
    acc = mm8(wq, q.T)                                   # (O, ho*wo) int32
    y = acc.float() * (sw[:, None] * s[None, :])         # 反量化：权重 scale × 激活 scale
    y = y + b[:, None]
    return y.T.reshape(ho, wo, -1)


def linear(x, wq, sw, b, act_block=0):
    q, s = quant_rows(x)
    acc = mm8(wq, q.T)
    return (acc.float() * (sw[:, None] * s[None, :])).T + b[None, :]


def gn(x, g, b, groups=32, eps=1e-6):
    h, w, c = x.shape
    xg = x.reshape(-1, groups, c // groups)
    mu = xg.mean(dim=(0, 2), keepdim=True)
    var = xg.var(dim=(0, 2), unbiased=False, keepdim=True)
    y = ((xg - mu) / torch.sqrt(var + eps)).reshape(h, w, c)
    return y * g + b


def silu(x):
    return x * torch.sigmoid(x)
