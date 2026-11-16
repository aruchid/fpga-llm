"""L1 性能模型：整个模型在 FPGA 上推理的运算量、数据量和耗时估算。

- 运算清单（inventory）按 l0/ref_hw.py 的流程逐项统计，单元测试保证与 L0 实际调用一致
- 矩阵单元周期公式与 rtl/mvm_core.sv 一致，单元测试用 RTL 仿真实测值校验
- 板卡、PCIe、驱动延迟等是假设参数（见 BOARDS / HOST），不是实测值

  python -m l1.perf                 # 全部报告
  python -m l1.perf --model stories15M --board KU040
"""
import argparse
import math
from dataclasses import dataclass, replace

from l0.config import STORIES15M, Config

# ------------------------------------------------------------------ 模型
MODELS = {
    "stories15M": (STORIES15M, False),
    "stories110M": (Config(768, 2048, 12, 12, 12, 32000, 1024), False),
    "Llama-3.2-1B": (Config(2048, 8192, 16, 32, 8, 128256, 2048), True),   # 词表与嵌入共享
    "Llama-2-7B": (Config(4096, 11008, 32, 32, 32, 32000, 4096), False),
}


# ------------------------------------------------------------------ 硬件
@dataclass(frozen=True)
class Board:
    name: str
    f_mhz: float      # 计算核时钟
    ddr_gbs: float    # 片外存储有效带宽 GB/s
    ddr_gb: float     # 片外存储容量 GB
    pcie_gbs: float   # PCIe DMA 有效带宽 GB/s
    dsp: int          # DSP 数量（估算并行度上限）
    note: str


# 典型参数，按公开规格粗估（2026-11-16），有效带宽按峰值约 70% 计
BOARDS = {
    "A7-200T": Board("A7-200T", 150, 4.5, 1, 1.6, 740, "Artix-7，PCIe Gen2 x4，DDR3 32 位"),
    "K7-325T": Board("K7-325T", 200, 9.0, 1, 3.2, 840, "Kintex-7，PCIe Gen2 x8，DDR3 64 位"),
    "KU040": Board("KU040", 250, 14.0, 2, 6.5, 1920, "Kintex UltraScale，PCIe Gen3 x8，DDR4 64 位"),
    "HBM 卡": Board("HBM 卡", 300, 300.0, 16, 12.0, 9024, "Alveo U55C 类，PCIe Gen3 x16，HBM2"),
}


@dataclass(frozen=True)
class Host:
    """电脑端经 PCIe 操作 FPGA 的固定开销（XDMA 驱动的典型量级，假设值）。"""
    mmio_wr_us: float = 0.25   # 一次寄存器写（posted）
    mmio_rd_us: float = 1.0    # 一次寄存器读（要等返回）
    dma_us: float = 8.0        # 一次 DMA 的固定开销（系统调用 + 描述符 + 完成）
    irq_us: float = 10.0       # 中断到用户态被唤醒
    elem_ns: float = 1.0       # 电脑端非线性运算每元素耗时（C 实现量级）


@dataclass(frozen=True)
class Engine:
    """矩阵单元（与 rtl/mvm_core.sv 同构）和向量单元的参数。"""
    lanes: int = 32            # 每拍处理的 int8 个数 = 每周期乘加数
    post: int = 32             # 每 tile 输出阶段停顿周期；双缓冲累加器后为 0
    start: int = 1             # 每次任务的启动周期
    vec_lanes: int = 16        # 向量单元每周期处理元素数
    sfu_lat: int = 16          # 1/√x、1/x 一次的延迟（查表 + 2 次牛顿）
    op_lat: int = 20           # 片上调度时每个算子的流水线填充开销


V1 = Engine()                       # 第一版 RTL：每 tile 输出停 32 拍
V2 = Engine(post=1, start=46)       # 可综合版 RTL：双组累加器，每 tile 只多 1 拍，任务固定开销 46 拍
CUR = V2                            # 当前 RTL


def ceil(a, b):
    return -(-a // b)


# ------------------------------------------------------------------ 矩阵单元周期
TILE = 32   # tile 尺寸，与 RTL 的 N 一致


def mvm_cycles(M, K, e: Engine = CUR):
    """一次 (M,K) 矩阵 × 向量任务的周期数。

    lanes=32 时与 rtl/mvm_core.sv 完全一致（V1 / V2 分别对应第一版与可综合版）；lanes 更大时视为 lanes/32 个相同的核并行，
    各自从存储读一路权重流，按输出 tile 分工。
    """
    n = TILE
    cores = max(1, e.lanes // n)
    nt, kb = ceil(M, n), ceil(K, n)
    return ceil(nt, cores) * (n // 8 + kb * n + e.post) + e.start


def weight_jobs(cfg: Config, fuse=True):
    """一个 token 的权重矩阵乘任务列表 [(名称, M, K)]。fuse：q/k/v、w1/w3 合并。"""
    d, h, kv = cfg.dim, cfg.hidden_dim, cfg.kv_dim
    if fuse:
        layer = [("qkv", d + 2 * kv, d), ("wo", d, d), ("w1w3", 2 * h, d), ("w2", d, h)]
    else:
        layer = [("wq", d, d), ("wk", kv, d), ("wv", kv, d), ("wo", d, d),
                 ("w1", h, d), ("w3", h, d), ("w2", d, h)]
    return layer * cfg.n_layers + [("wcls", cfg.vocab_size, d)]


def attn_jobs(cfg: Config, T):
    """注意力在矩阵单元上的任务：每层每头 q·K（M=T,K=hs）和 P·V（M=hs,K=T）。"""
    hs = cfg.head_size
    return ([("qk", T, hs), ("pv", hs, T)] * cfg.n_heads) * cfg.n_layers


# ------------------------------------------------------------------ 运算清单
def inventory(cfg: Config, T, shared_emb=False):
    """一个 token（上下文长度 T = pos+1）的运算量与数据量。"""
    d, h, kv, L, H, V = cfg.dim, cfg.hidden_dim, cfg.kv_dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size
    hs = cfg.head_size
    w_macs = sum(M * K for _, M, K in weight_jobs(cfg))
    w_rows = sum(M for _, M, _ in weight_jobs(cfg))
    att_macs = 2 * L * H * T * hs
    vec = {   # 向量运算：(元素数, 遍数)。遍数 = 需要扫过数据几次
        "RMSNorm": ((2 * L + 1) * d, 2),
        "量化移位 quantize_shift": ((2 * L + 1) * d + L * (d + h + d + 2 * kv), 2),
        "RoPE": (L * (d + kv), 1),
        "注意力分数缩放": (L * H * T, 1),
        "Softmax": (L * H * T, 3),
        "SiLU × 门控": (L * h, 1),
        "残差相加": (2 * L * d, 1),
    }
    sfu = {"1/√x": 2 * L + 1, "1/x": L * H}
    emb_bytes = V * d + 4 * V
    weight_bytes = w_macs + 4 * w_rows            # int8 + 每行 SW
    params_bytes = weight_bytes + (0 if shared_emb else emb_bytes) + (2 * L + 1) * d * 4
    kv_read = 2 * L * T * kv + 2 * L * T * cfg.n_kv_heads   # int8 + 每位置每头移位量
    kv_total = 2 * L * cfg.seq_len * (kv + cfg.n_kv_heads)
    return dict(w_macs=w_macs, att_macs=att_macs, vec=vec, sfu=sfu,
                weight_bytes=weight_bytes, kv_read=kv_read,
                ddr_footprint=params_bytes + kv_total, kv_total=kv_total)


# ------------------------------------------------------------------ 当前 RTL 向量单元（1 元素/周期）
SFU_LAT = 8   # 第一版 RTL 的 1/√x、1/x 等待周期


def vpu_rtl_cycles(cfg: Config, T, version=2):
    """RTL 向量单元执行一个 token 的非线性运算的周期（每单元 1 元素/周期，串行，含数据进出）。

    version=2（当前可综合版，tb_vpu.cycle_profile 实测拟合）：
      quant 2n+17；rmsnorm 2n+70；rope n+9；silu_gate / resadd n+17；
      注意力头 10T+hs+67（K、V 每行 4 拍，指数、概率各 1 拍/行）
    version=1（第一版）：载入 n + 输出 n；RoPE 每对 4 拍；注意力头 4T+hs+10
    """
    d, h, kv, L, H, hs = cfg.dim, cfg.hidden_dim, cfg.kv_dim, cfg.n_layers, cfg.n_heads, cfg.head_size
    quant_sizes = [d] * (2 * L + 1) + [hs] * (L * (H + 2 * cfg.n_kv_heads)) + [d] * L + [h] * L
    if version == 1:
        return {
            "quant": sum(2 * n + 2 for n in quant_sizes),
            "rmsnorm": (2 * L + 1) * (2 * d + SFU_LAT + 1),
            "rope": L * (2 * (d + kv) + 2),
            "silu_gate": L * (h + 2),
            "resadd": 2 * L * (d + 2),
            "attn": L * H * (4 * T + hs + SFU_LAT + 2),
        }
    return {
        "quant": sum(2 * n + 17 for n in quant_sizes),
        "rmsnorm": (2 * L + 1) * (2 * d + 70),
        "rope": L * ((d + 9) + (kv + 9)),
        "silu_gate": L * (h + 17),
        "resadd": 2 * L * (d + 17),
        "attn": L * H * (10 * T + hs + 67),
    }


# ------------------------------------------------------------------ 解码耗时
def device_matmul_s(jobs, b: Board, e: Engine, extra_bytes=0):
    cyc = sum(mvm_cycles(M, K, e) for _, M, K in jobs)
    byts = sum(M * K + 4 * M for _, M, K in jobs) + extra_bytes
    t_c, t_m = cyc / (b.f_mhz * 1e6), byts / (b.ddr_gbs * 1e9)
    return max(t_c, t_m), t_c, t_m, cyc


def vec_cycles(inv, e: Engine):
    el = sum(n * p for n, p in inv["vec"].values())
    return ceil(el, e.vec_lanes) + sum(inv["sfu"].values()) * e.sfu_lat


def decode(cfg: Config, b: Board, arch: str, T=1, e: Engine = CUR, host=Host(), shared=False):
    """返回一个 token 的耗时分解（秒）。

    arch:
      A0-MMIO  只有权重矩阵乘在 FPGA；x 逐字写寄存器，轮询完成，结果 DMA 回传（当前 L2 RTL）
      A0-DMA   同上，但 x/y 都用 DMA，中断通知
      A1-RTL   整个模型在 FPGA 上，按当前 RTL（V2）：32 路矩阵单元 + 每周期 1 元素的向量单元，串行执行
      A1       整个模型在 FPGA 上（矩阵单元 + 16 路向量单元 + 注意力走矩阵单元 + 片上调度），电脑每 token 只发一条命令
    """
    inv = inventory(cfg, T, shared)
    jobs = weight_jobs(cfg)
    r = {}
    if arch.startswith("A0"):
        r["矩阵乘（FPGA）"], *_ = device_matmul_s(jobs, b, e)
        n_el = sum(n * p for n, p in inv["vec"].values()) + 2 * cfg.n_layers * cfg.n_heads * T * cfg.head_size
        r["非线性 + 注意力（电脑）"] = n_el * host.elem_ns * 1e-9
        io = 0.0
        for _, M, K in jobs:
            y_bytes = ceil(M, e.lanes) * e.lanes * 4
            if arch == "A0-MMIO":
                io += (ceil(K, 4) + 4) * host.mmio_wr_us + 2 * host.mmio_rd_us
            else:
                io += 10 * host.mmio_wr_us + host.dma_us + K / (b.pcie_gbs * 1e3) + host.irq_us
            io += host.dma_us + y_bytes / (b.pcie_gbs * 1e3)
        r["PCIe 往返"] = io * 1e-6
    elif arch == "A1-RTL":
        t, *_ = device_matmul_s(jobs, b, e)
        r["矩阵乘（FPGA）"] = t
        r["向量单元 + 注意力（FPGA）"] = sum(vpu_rtl_cycles(cfg, T).values()) / (b.f_mhz * 1e6)
        r["PCIe 往返"] = (2 * host.mmio_wr_us + host.irq_us + host.mmio_rd_us) * 1e-6
    else:
        t, *_ = device_matmul_s(jobs, b, e)
        r["矩阵乘（FPGA）"] = t
        a_jobs = attn_jobs(cfg, T)
        t_att, *_ = device_matmul_s(a_jobs, b, e, extra_bytes=inv["kv_read"]
                                    - sum(M * K + 4 * M for _, M, K in a_jobs))
        r["注意力（FPGA）"] = t_att
        n_ops = cfg.n_layers * 14 + 4
        r["向量单元（FPGA）"] = (vec_cycles(inv, e) + ceil(cfg.vocab_size, e.vec_lanes)
                             + n_ops * e.op_lat) / (b.f_mhz * 1e6)
        r["PCIe 往返"] = (2 * host.mmio_wr_us + host.irq_us + host.mmio_rd_us) * 1e-6
    r["合计"] = sum(r.values())
    return r


def lanes_matched(b: Board):
    """让矩阵单元刚好吃满存储带宽的并行度（32 的倍数，受 DSP 限制，1 个 DSP 算 2 个 int8 乘法）。"""
    need = b.ddr_gbs * 1e9 / (b.f_mhz * 1e6)
    lanes = max(32, 32 * round(need / 32))
    cap = 32 * (int(b.dsp * 2 * 0.7) // 32)
    return min(lanes, cap)


def prefill(cfg: Config, b: Board, Tp, gemm_macs=None, shared=False, e: Engine = CUR):
    """预填充 Tp 个 token 的耗时（秒）。gemm_macs=None 表示只有 MVM（逐 token 解码）。"""
    if gemm_macs is None:
        return sum(decode(cfg, b, "A1", T + 1, e, shared=shared)["合计"] for T in range(Tp))
    inv = inventory(cfg, 1, shared)
    t_w = max(Tp * inv["w_macs"] / (gemm_macs * b.f_mhz * 1e6), inv["weight_bytes"] / (b.ddr_gbs * 1e9))
    att = cfg.n_layers * cfg.n_heads * cfg.head_size * Tp * (Tp + 1)       # 因果注意力 q·K + P·V
    t_a = att / (gemm_macs * b.f_mhz * 1e6)
    t_v = Tp * vec_cycles(inventory(cfg, Tp // 2 + 1, shared), e) / (b.f_mhz * 1e6)
    return t_w + t_a + t_v


# ------------------------------------------------------------------ 报告
def fmt_bytes(n):
    for u, s in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= s:
            return f"{n / s:.2f} {u}"
    return f"{n} B"


def report(model="stories15M", board="KU040"):
    cfg, shared = MODELS[model]
    b = BOARDS[board]
    out = []
    p = out.append
    p(f"# L1 性能模型报告：{model} @ {board}（{b.note}）\n")

    p("## 1. 每 token 运算清单（上下文长度 T）\n")
    p("| 项目 | T=1 | T=seq_len |")
    p("|---|---|---|")
    i1, i2 = inventory(cfg, 1, shared), inventory(cfg, cfg.seq_len, shared)
    p(f"| 权重矩阵乘 MAC | {i1['w_macs']:,} | {i2['w_macs']:,} |")
    p(f"| 注意力 MAC | {i1['att_macs']:,} | {i2['att_macs']:,} |")
    for k in i1["vec"]:
        p(f"| {k}（元素×遍） | {i1['vec'][k][0] * i1['vec'][k][1]:,} | {i2['vec'][k][0] * i2['vec'][k][1]:,} |")
    p(f"| 读权重 | {fmt_bytes(i1['weight_bytes'])} | {fmt_bytes(i2['weight_bytes'])} |")
    p(f"| 读 KV Cache | {fmt_bytes(i1['kv_read'])} | {fmt_bytes(i2['kv_read'])} |")
    p(f"\n片外存储占用：权重 + 满长度 KV Cache = {fmt_bytes(i1['ddr_footprint'])}"
      f"（KV Cache {fmt_bytes(i1['kv_total'])}）\n")

    p("## 2. 解码耗时分解（每 token，T = seq_len/2）\n")
    T = cfg.seq_len // 2
    archs = ["A0-MMIO", "A0-DMA", "A1-RTL", "A1"]
    rs = {a: decode(cfg, b, a, T, shared=shared) for a in archs}
    keys = []
    for a in archs:
        keys += [k for k in rs[a] if k not in keys and k != "合计"]
    keys.append("合计")
    p("| 部分 | " + " | ".join(archs) + " |")
    p("|---|" + "---|" * len(archs))
    for k in keys:
        p(f"| {k} | " + " | ".join(f"{rs[a][k] * 1e3:.3f} ms" if k in rs[a] else "—" for a in archs) + " |")
    p("| token/s | " + " | ".join(f"{1 / rs[a]['合计']:.0f}" for a in archs) + " |\n")
    return "\n".join(out)


def report_matrix():
    out = []
    p = out.append
    p("## 3. 各模型 × 各板卡：A1 解码速度（token/s，T = seq_len/2）\n")
    p("每格：32 路矩阵单元 / 并行度匹配存储带宽后。✗ 表示片外存储放不下。\n")
    p("| 模型 | 片外占用 | " + " | ".join(BOARDS) + " |")
    p("|---|---|" + "---|" * len(BOARDS))
    for m, (cfg, shared) in MODELS.items():
        inv = inventory(cfg, 1, shared)
        cells = []
        for b in BOARDS.values():
            if inv["ddr_footprint"] > b.ddr_gb * 1e9:
                cells.append("✗")
                continue
            T = cfg.seq_len // 2
            t32 = decode(cfg, b, "A1", T, shared=shared)["合计"]
            e = replace(CUR, lanes=lanes_matched(b))
            tm = decode(cfg, b, "A1", T, e, shared=shared)["合计"]
            cells.append(f"{1 / t32:.1f} / {1 / tm:.1f}（{e.lanes} 路）")
        p(f"| {m} | {fmt_bytes(inv['ddr_footprint'])} | " + " | ".join(cells) + " |")

    p("\n## 4. 瓶颈：A1 解码时计算与存储哪个先到极限（32 路，T=1）\n")
    p("| 模型 | 板卡 | 权重矩阵乘：计算时间 | 存储时间 | 瓶颈 |")
    p("|---|---|---|---|---|")
    for m in ("stories15M", "Llama-3.2-1B"):
        cfg, shared = MODELS[m]
        for b in BOARDS.values():
            t, tc, tm, _ = device_matmul_s(weight_jobs(cfg), b, CUR)
            p(f"| {m} | {b.name} | {tc * 1e3:.2f} ms | {tm * 1e3:.2f} ms | {'计算' if tc > tm else '存储带宽'} |")

    p("\n## 5. 预填充（提示词 128 token）\n")
    p("| 模型 | 板卡 | 只有 MVM | 加 GEMM 阵列 | GEMM 规模 |")
    p("|---|---|---|---|---|")
    for m in ("stories15M", "stories110M"):
        cfg, shared = MODELS[m]
        for bn in ("A7-200T", "KU040"):
            b = BOARDS[bn]
            g = min(1024, 32 * (int(b.dsp * 2 * 0.7) // 32))
            t0 = prefill(cfg, b, 128, None, shared)
            t1 = prefill(cfg, b, 128, g, shared)
            p(f"| {m} | {bn} | {t0 * 1e3:.1f} ms | {t1 * 1e3:.1f} ms | {g} MAC/周期 |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stories15M", choices=MODELS)
    ap.add_argument("--board", default="KU040", choices=BOARDS)
    a = ap.parse_args()
    print(report(a.model, a.board))
    print(report_matrix())


if __name__ == "__main__":
    main()
