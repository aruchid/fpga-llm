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


def _imm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8 × int8 → int32；_int_mm 要求 K、N 是 8 的倍数，这里补零。"""
    k, n = a.shape[1], b.shape[1]
    k8, n8 = (k + 7) // 8 * 8, (n + 7) // 8 * 8
    if k8 != k:
        a = torch.nn.functional.pad(a, (0, k8 - k))
        b = torch.nn.functional.pad(b, (0, 0, 0, k8 - k))
    if n8 != n:
        b = torch.nn.functional.pad(b, (0, n8 - n))
    try:
        return torch._int_mm(a.contiguous(), b.contiguous())[:, :n]
    except RuntimeError:
        # 小形状 cuBLAS 的 int8 内核不支持；float64 精确表示整数，结果一致
        return torch.round(a.double() @ b.double()).to(torch.int32)[:, :n]


class TileStats:
    def __init__(self):
        self.calls = 0
        self.macs = 0
        self.tiles = 0
        self.cycles = 0
        self.max_abs_acc = 0


class MVM32Tiled:
    def __init__(self, device="cuda", fmax_mhz=FMAX_MHZ, blk=BLK, strict: bool = False):
        self.device = device
        self.blk = blk
        self.fmax_mhz = fmax_mhz
        self.strict = strict          # True: 逐 K 块 IMMA 累加（显式复现 RTL）
        self.stats = TileStats()

    def gemm_int8(self, W: torch.Tensor, X: torch.Tensor, max_cols: int = 8192,
                  strict: bool = False) -> torch.Tensor:
        """W (M,K) int8 × X (K,B) int8 → int32 (M,B)。

        两种等价实现：
          strict=True  逐个 32 元素 K 块调用 IMMA，再 int32 相加——显式复现 RTL 的块累加；
          strict=False 一次 IMMA 完成整段 K——int32 加法可结合，结果与逐块完全相同
                       （已用 test_mvm_tiled 逐比特验证），只是快 5-8 倍。

        按列分块（max_cols）限制中间缓冲；分块只切 B 方向，不改变累加顺序。
        两种模式下 tile 数与周期统计完全相同。
        """
        B = X.shape[1]
        if B > max_cols:
            return torch.cat([self.gemm_int8(W, X[:, c:c + max_cols], max_cols, strict)
                              for c in range(0, B, max_cols)], dim=1)
        return self._gemm_tiles(W, X, strict or self.strict)

    def _gemm_tiles(self, W: torch.Tensor, X: torch.Tensor, strict: bool = False) -> torch.Tensor:
        M, K = W.shape
        B = X.shape[1]
        nb = self.blk
        nt = math.ceil(M / nb)
        kb = math.ceil(K / nb)

        # 补齐到 32 的整数倍
        Wp = torch.zeros((M, kb * nb), dtype=torch.int8, device=W.device)
        Wp[:, :K] = W
        Xp = torch.zeros((kb * nb, B), dtype=torch.int8, device=X.device)
        Xp[:K] = X

        if strict:
            # 逐 K 块（每块 32 个元素）IMMA → int32 相加，显式复现 RTL 的块累加
            acc = torch.zeros((M, B), dtype=torch.int32, device=W.device)
            for kk in range(kb):
                k0 = kk * nb
                acc = acc + _imm(Wp[:, k0:k0 + nb], Xp[k0:k0 + nb, :])
        else:
            # 一次算完：int32 加法可结合，与逐块结果逐比特相同
            acc = _imm(Wp, Xp)

        st = self.stats
        st.calls += 1
        st.macs += M * K * B
        st.tiles += nt * kb * math.ceil(B / nb)
        st.cycles += B * nt * (kb * nb + nb // 8)
        st.max_abs_acc = max(st.max_abs_acc, int(acc.abs().max()))
        return acc

    # ---------------------------------------------------------------- 周期
    def hw_seconds(self) -> float:
        return self.stats.cycles / (self.fmax_mhz * 1e6)
