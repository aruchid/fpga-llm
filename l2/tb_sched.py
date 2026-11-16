"""整模型推理：调度器把全部权重矩阵乘交给 mvm_top RTL，逐步与 L0 硬件等价模型逐比特比对。

环境变量：
  L2_GEN_MODEL   tiny（默认，随机权重小模型）或 stories15M
  L2_GEN_STEPS   生成步数（默认 tiny 8 步，stories15M 16 步）
  L2_GEN_PROMPT  提示词（stories15M 用，默认 "Once upon a time"）
  L2_GEN_FUSE    1（默认）合并共享输入的矩阵乘，0 不合并
  L2_GEN_VPU     1（默认）非线性运算也交给 RTL 向量单元，0 在电脑端用 L0 计算
"""
import os
import time
from pathlib import Path

import cocotb
import numpy as np
from cocotb.utils import get_sim_time

from l0.config import TINY
from l0.ref_hw import HWModel
from l0.weights import random_weights
from l2.driver import MVMDriver
from l2.scheduler import RTLScheduler
from l2.vpu_driver import VPUDriver

ROOT = Path(__file__).resolve().parents[1]


def load():
    name = os.environ.get("L2_GEN_MODEL") or "tiny"
    if name == "tiny":
        return TINY, random_weights(TINY), None, [1, 5, 17, 42], 8
    from l0.tokenizer import Tokenizer
    from l0.weights import load_llama2c
    cfg, w = load_llama2c(str(ROOT / "models" / "stories15M.bin"))
    tok = Tokenizer.load(str(ROOT / "models" / "tokenizer.bin"), cfg.vocab_size)
    prompt = tok.encode(os.environ.get("L2_GEN_PROMPT") or "Once upon a time")
    return cfg, w, tok, prompt, 16


@cocotb.test()
async def generate_with_rtl(dut):
    log = dut._log
    cfg, w, tok, prompt, default_steps = load()
    steps = min(int(os.environ.get("L2_GEN_STEPS") or default_steps), cfg.seq_len)
    fuse = (os.environ.get("L2_GEN_FUSE") or "1") == "1"
    show = tok.decode if tok else str

    hm = HWModel(cfg, w)
    drv = MVMDriver(dut)
    await drv.reset()
    use_vpu = (os.environ.get("L2_GEN_VPU") or "1") == "1"
    vpu = VPUDriver(dut) if use_vpu else None
    sched = RTLScheduler(hm, drv, fuse=fuse, vpu=vpu)
    n_mat, nbytes, t_pack = sched.preload()
    log.info(f"模型 {cfg}")
    log.info(f"预加载权重：{n_mat} 个矩阵（合并={fuse}），{nbytes / 1e6:.2f} MB 流数据，打包 {t_pack:.1f} s")

    t0, sim0 = time.time(), get_sim_time("ns")

    def on_token(pos, t):
        st = drv.stats
        log.info(f"pos {pos:3d} → {t:5d}  累计 {st['core_cycles']:>9,} 周期  "
                 f"{show([t]) if tok else ''}")

    tokens, logits = await sched.generate(prompt, steps, on_token)
    wall, sim_ns = time.time() - t0, get_sim_time("ns") - sim0

    # ---- 与 L0 逐比特比对
    hm.reset()
    for pos, want_tok in enumerate(tokens[:len(logits)]):
        want = hm.forward_fx(want_tok, pos)
        bad = np.flatnonzero(logits[pos] != want)
        assert bad.size == 0, f"pos {pos}: {bad.size} 个 logits 与 L0 不一致"

    n = len(logits)
    st = drv.stats
    log.info("=" * 60)
    log.info(f"生成结果：{show(tokens)}")
    log.info(f"{n} 步全部与 L0 硬件等价模型逐比特一致（logits 完全相同）")
    log.info(f"每 token：{st['jobs'] / n:.0f} 次 RTL 任务，计算核 {st['core_cycles'] / n:,.0f} 周期，"
             f"仿真时间 {sim_ns / n / 1e3:,.0f} µs（250 MHz）")
    log.info(f"每 token 传输：权重流 {st['w_bytes'] / n / 1e6:.2f} MB，结果 {st['y_bytes'] / n / 1e3:.1f} KB，"
             f"AXI-Lite 写 {st['axil_writes'] / n:.0f} 次、读 {st['axil_reads'] / n:.0f} 次")
    log.info(f"仿真耗时 {wall:.1f} s（{wall / n:.1f} s/token）")
    log.info("各运算在 RTL 上的周期（每 token 平均，含数据进出）：")
    log.info(f"  matmul     {st['jobs'] / n:6.0f} 次  {st['core_cycles'] / n:>10,.0f} 周期（矩阵单元计数器）")
    if vpu:
        for k, (cnt, cyc, el) in sorted(vpu.stats.items(), key=lambda kv: -kv[1][1]):
            log.info(f"  {k:10s} {cnt / n:6.0f} 次  {cyc / n:>10,.0f} 周期  {el / n:>8,.0f} 元素")
    log.info(f"留在电脑端的运算：{sched.host_ops or '无'}（另有词嵌入查表与 argmax）")
