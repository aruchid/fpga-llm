"""严格的 32×32 tile 矩阵引擎：输出和 K 都按 32 切块，逐 tile 累加。

对应 rtl/mvm_core.sv（N = 32）的硬件行为：
    - 权重按 32 行一个输出 tile，K 按 32 个元素一块
    - 每个 tile：32×32 的 int8 乘加，累加到 int32
    - 权重逐行（每行 32 个 int8）流式送入

本模块把这个切分**显式**做出来（nt × kb 个 tile），而不是只做统计：
    for 每个 K 块 kk:
        for 每个输出 tile t:  acc[t] += W[t, kk] · X[kk]      # 32×32 × 32×B
为了速度，把 nt 个 tile 用广播一次算完（每个 tile 仍是独立的 32×32 乘加）。

整数精确性：int8 乘积 ≤ 127²，累加 ≤ K·127² < 2^53，float64 可精确表示，
加法顺序不影响结果，因此与 RTL 的 int32 累加逐比特一致（溢出时按 32 位环绕）。
"""

from __future__ import annotations

import math

import torch

BLK = 32
FMAX_MHZ = 115.0


class TileStats:
    def __init__(self):
        self.calls = 0
        self.macs = 0
        self.tiles = 0
        self.cycles = 0
        self.max_abs_acc = 0


class MVM32Tiled:
    def __init__(self, device="cuda", fmax_mhz=FMAX_MHZ, blk=BLK):
        self.device = device
        self.blk = blk
        self.fmax_mhz = fmax_mhz
        self.stats = TileStats()

    def gemm_int8(self, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """W (M,K) int8 × X (K,B) int8 → int32 (M,B)，严格按 32×32 tile 累加。"""
        M, K = W.shape
        B = X.shape[1]
        nb = self.blk
        nt = math.ceil(M / nb)
        kb = math.ceil(K / nb)

        # 补齐到 32 的整数倍后按 tile 重排
        Wp = torch.zeros((nt * nb, kb * nb), dtype=torch.int8, device=W.device)
        Wp[:M, :K] = W
        Xp = torch.zeros((kb * nb, B), dtype=torch.int8, device=X.device)
        Xp[:K] = X

        Wt = Wp.view(nt, nb, kb, nb).permute(0, 2, 1, 3).double()   # (nt, kb, 32, 32)
        Xt = Xp.view(kb, nb, B).double()                            # (kb, 32, B)

        acc = torch.zeros((nt, nb, B), dtype=torch.float64, device=W.device)
        for kk in range(kb):
            # 一次广播算 nt 个 tile，每个 tile 是 32×32 的乘加
            acc += Wt[:, kk] @ Xt[kk]

        acc = acc.reshape(nt * nb, B)[:M]
        # 与 RTL 的 int32 环绕一致
        wrapped = torch.remainder(acc.round().to(torch.int64) + (1 << 31), 1 << 32) - (1 << 31)

        st = self.stats
        st.calls += 1
        st.macs += M * K * B
        st.tiles += nt * kb * math.ceil(B / nb)
        st.cycles += B * nt * (kb * nb + nb // 8)
        st.max_abs_acc = max(st.max_abs_acc, int(acc.abs().max()))
        return wrapped.to(torch.int32)

    # ---------------------------------------------------------------- 周期
    def hw_seconds(self) -> float:
        return self.stats.cycles / (self.fmax_mhz * 1e6)
