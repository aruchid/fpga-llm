"""INT8 量化原语（W8A8）。

- 权重：per-channel（每个输出通道一个 scale）
- 激活：per-row（每个像素 / 每个 token 一个 scale），对应矩阵单元「每向量一个移位量」
- 矩阵乘：int8 × int8 → int32 累加（IMMA 指令，即 32×32 矩阵单元做的运算）

反量化（acc × s_w × s_a + bias）由调用方在向量单元完成。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

QMAX = 127


def quant_rows(x: torch.Tensor):
    """按行量化：x (n, K) → int8 (n, K) + 每行一个 scale (n,)。"""
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    s = amax / QMAX
    q = torch.clamp(torch.round(x / s), -QMAX, QMAX).to(torch.int8)
    return q, s[:, 0]


def quant_weight(w: torch.Tensor):
    """权重 (O, K) → int8 + 每输出通道一个 scale。"""
    return quant_rows(w)


def mm8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8 (M,K) × int8 (K,N) → int32 (M,N)。

    `torch._int_mm` 要求 K、N 都是 8 的倍数，这里补零；少数形状 cuBLAS 不支持时
    退回 float64（整数在 float64 中精确，结果一致）。
    """
    k, n = a.shape[1], b.shape[1]
    k8, n8 = (k + 7) // 8 * 8, (n + 7) // 8 * 8
    if k8 != k:
        a = F.pad(a, (0, k8 - k))
        b = F.pad(b, (0, 0, 0, k8 - k))
    if n8 != n:
        b = F.pad(b, (0, n8 - n))
    try:
        out = torch._int_mm(a.contiguous(), b.contiguous())
    except RuntimeError:
        out = torch.round(a.double() @ b.double()).to(torch.int32)
    return out[:, :n]
