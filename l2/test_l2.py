"""pytest 入口：Verilator 编译 RTL 并运行 cocotb 测试（l2/tb_mvm.py）。"""
import os
from pathlib import Path

import numpy as np
from cocotb_tools.runner import get_results, get_runner

from l0 import fixedpoint as fp
from l2 import pack as pk

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "rtl" / f for f in ("mul_pipe.sv", "vpu_units.sv", "mvm_core.sv", "mvm_top.sv", "npu_top.sv")]


def test_pack_layout():
    """打包格式自检：用 numpy 按流格式重算，结果等于 int_matmul（不跑仿真）。"""
    rng = np.random.default_rng(0)
    M, K = 40, 70
    wq = rng.integers(-127, 128, size=(M, K)).astype(np.int8)
    SW = rng.integers(1, 1 << 20, size=M)
    xq = rng.integers(-127, 128, size=K)
    stream, nt, kb = pk.pack_weights(wq, SW)
    x = np.frombuffer(np.array(pk.pack_x(xq, kb), dtype="<u4").tobytes(), dtype=np.int8)
    beat, ys = 0, []
    raw = np.frombuffer(stream, dtype=np.uint8)
    for _ in range(nt):
        sw = np.frombuffer(raw[beat * 32:(beat + 4) * 32].tobytes(), dtype="<u4").astype(np.int64)
        beat += 4
        acc = np.zeros(pk.N, dtype=np.int64)
        for b in range(kb):
            for r in range(pk.N):
                row = raw[beat * 32:(beat + 1) * 32].view(np.int8).astype(np.int64)
                acc[r] += row @ x[b * 32:(b + 1) * 32]
                beat += 1
        ys.append(fp.sat32(fp.rshift_round(acc * sw, 24 - 5)))
    y = np.concatenate(ys)[:M]
    assert beat * 32 == len(stream)
    assert np.array_equal(y, fp.int_matmul(wq, SW, xq, 5))


ENV_KEYS = ("L2_GEN_VPU", "L2_FULL", "L2_GEN_MODEL", "L2_GEN_STEPS", "L2_GEN_PROMPT", "L2_GEN_FUSE")


def run_cocotb(module, results_name):
    """编译（有缓存）并运行一个 cocotb 测试模块，返回 (测试数, 失败数)。"""
    build_dir = ROOT / "sim_build" / "npu"
    runner = get_runner("verilator")
    runner.build(sources=SOURCES, hdl_toplevel="npu_top", build_dir=build_dir,
                 includes=[ROOT / "rtl"], build_args=["-Wall", "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY"])
    env = {"PYTHONPATH": str(ROOT)}
    env.update({k: os.environ.get(k, "") for k in ENV_KEYS})
    xml = runner.test(hdl_toplevel="npu_top", test_module=module,
                      build_dir=build_dir, test_dir=ROOT, extra_env=env,
                      results_xml=str(build_dir / results_name))
    return get_results(xml)


def test_mvm_rtl():
    n, fails = run_cocotb("l2.tb_mvm", "results_mvm.xml")
    assert n > 0 and fails == 0, f"{fails}/{n} 个 cocotb 测试失败"


def test_generate_rtl():
    """整模型推理（默认随机权重小模型）：RTL 做全部矩阵乘，与 L0 逐比特一致。"""
    n, fails = run_cocotb("l2.tb_sched", "results_sched.xml")
    assert n > 0 and fails == 0, f"{fails}/{n} 个 cocotb 测试失败"


def test_vpu_rtl():
    """向量单元（非线性运算）RTL 与 L0 逐比特一致。"""
    n, fails = run_cocotb("l2.tb_vpu", "results_vpu.xml")
    assert n > 0 and fails == 0, f"{fails}/{n} 个 cocotb 测试失败"
