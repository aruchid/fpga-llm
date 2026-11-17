"""按 32×32 tile 调度的 INT8 矩阵乘引擎 —— 与 rtl/mvm_core.sv 的累加模型一致。

RTL 的做法（N = 32）：
    - 输出按 32 行一 tile 切；K 按 32 个元素一「块」切（kb 循环）
    - 每个 tile 内：32 个 int8 乘加，累加到 int32
    - 权重逐行流式送入，每拍 32 个 int8

本模块把同样的切分显式做出来：
    对每个 K 块（32 个元素）：acc += W[:, k0:k0+32] · X[k0:k0+32, :]
累加在 float64 里做——只要 K·127² < 2^53，整数在 float64 中精确可表示，
加法顺序不影响结果，所以与 RTL 的 int32 累加逐比特一致。

M 方向的 32 行 tile 不改变算术（只是把输出行分组），这里用 tile 数统计即可。
"""

from __future__ import annotations

import math

import numpy as np
import torch

BLK = 32          # 矩阵单元的通道数 N
FMAX_MHZ = 115.0


class TileStats:
    def __init__(self):
        self.calls = 0
        self.macs = 0
        self.tiles = 0
        self.cycles = 0
        self.saturations = 0


class MVM32Tile:
    """32×32 tile 级 INT8 GEMM。输入输出都是 torch 张量（CUDA）。"""

    def __init__(self, device="cuda", fmax_mhz=FMAX_MHZ, blk=BLK):
        self.device = device
        self.blk = blk
        self.fmax_mhz = fmax_mhz
        self.stats = TileStats()

    # ---------------------------------------------------------------- 核心
    def gemm_int8(self, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """W (M,K) int8 × X (K,B) int8 → int32 (M,B)。K 按 32 分块累加。"""
        M, K = W.shape
        B = X.shape[1]
        acc = torch.zeros((M, B), dtype=torch.float64, device=self.device)
        for k0 in range(0, K, self.blk):
            k1 = min(k0 + self.blk, K)
            acc += W[:, k0:k1].double() @ X[k0:k1, :].double()
        out = acc.round().to(torch.int32)
        # 统计：tile 数、周期（每拍一行权重，每 tile 32 拍 + 4 拍 SW）
        nt = math.ceil(M / self.blk)
        kb = math.ceil(K / self.blk)
        nb = math.ceil(B / self.blk)
        self.stats.calls += 1
        self.stats.macs += M * K * B
        self.stats.tiles += nt * kb * nb
        self.stats.cycles += B * nt * (kb * self.blk + self.blk // 8)
        return out

    def hw_seconds(self) -> float:
        return self.stats.cycles / (self.fmax_mhz * 1e6)

    # ---------------------------------------------------------------- 卷积/线性
    def conv(self, x: torch.Tensor, qw: torch.Tensor, sw: torch.Tensor, b: torch.Tensor,
             kh: int, kw: int, pad: int = 1, stride: int = 1) -> torch.Tensor:
        """x: (N,C,H,W) float32；qw: (O, kh*kw*C) int8；sw: (O,)；(kh,kw) 顺序与 qw 一致。"""
        x = x.float()
        n, c, h, w = x.shape
        outs = []
        for bi in range(n):
            xh = x[bi].permute(1, 2, 0).contiguous()           # NHWC
            xp = torch.nn.functional.pad(xh, (0, 0, pad, pad, pad, pad)) if pad else xh
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
            q, sa = _q_rows(cols)
            acc = self.gemm_int8(qw, q.T)                      # (O, ho*wo) int32
            y = acc.float() * (sw[:, None] * sa[None, :]) + b[:, None]
            outs.append(y.T.reshape(ho, wo, -1).permute(2, 0, 1))
        return torch.stack(outs, dim=0)

    def linear(self, x: torch.Tensor, qw: torch.Tensor, sw: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        q, sa = _q_rows(x2)
        acc = self.gemm_int8(qw, q.T)                          # (O, n) int32
        y = acc.float() * (sw[:, None] * sa[None, :]) + b[:, None]
        return y.T.reshape(*shape[:-1], -1).to(x.dtype)


def _q_rows(x: torch.Tensor):
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    s = amax / 127.0
    q = torch.clamp(torch.round(x / s), -127, 127).to(torch.int8)
    return q, s[:, 0]


def quant_weight(w: torch.Tensor):
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    s = amax / 127.0
    q = torch.clamp(torch.round(w / s), -127, 127).to(torch.int8)
    return q, s[:, 0]
