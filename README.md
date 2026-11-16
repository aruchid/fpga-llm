# fpga-llm

在 FPGA 上运行大模型推理的分层设计与仿真项目。以 [llama2.c](https://github.com/karpathy/llama2.c) 的 stories15M（约 1500 万参数）为目标模型，从 Python 参考模型、性能模型一直做到可综合的 RTL，并在 RTL 仿真里跑通了整个模型的推理。

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big
>
> —— 全部运算在 RTL 仿真中执行生成的文本，每一步 logits 与 Python 参考模型逐比特相同

## 现状

| 层 | 内容 | 状态 |
|---|---|---|
| **L0** | Python 参考模型：浮点版作标准答案；硬件等价定点版（int8 权重、Q16.16 激活）是 RTL 的"硬件契约"，前向计算拆成 7 种运算的运算流 | ✅ |
| **L1** | 性能模型：全模型运算清单、解码 / 预填充耗时、4 种架构 × 4 种板卡 × 4 种模型规模 | ✅ |
| **L2** | SystemVerilog RTL + Verilator / cocotb 仿真：32 路矩阵单元 + 向量单元（量化、RMSNorm、Softmax、RoPE、注意力、SwiGLU、残差），AXI 接口 | ✅ 整模型逐比特一致，可综合 |
| 综合 | yosys 综合 + 自写粗略时序估算：约 1.5 万 LUT、118 DSP，估算约 115 MHz（7 系列） | ✅ 粗估；Vivado 待做 |
| **L3** | 上板（PCIe XDMA + 板载 DDR） | 未开始 |

主要结论（详见 [进度报告](docs/00-进度报告.md)）：

- 定点化精度：与浮点版相比，下一个词预测一致率 96.9%，logits 余弦相似度最低 0.98
- 解码速度由**存储带宽**决定：每生成一个 token 都要读一遍全部权重（stories15M 为 15.4 MB）
- 只把矩阵乘放 FPGA 时，PCIe 往返占 1/4 以上时间；整个模型放上 FPGA 后 PCIe 开销可忽略
- 计算资源不是瓶颈：整套计算单元在 Artix-7 200T 级别就放得下

## 文档

| 文档 | 内容 |
|---|---|
| [00 进度报告](docs/00-进度报告.md) | 完成情况、全部测试结果、问题与计划、选板建议 |
| [01 总体设计](docs/01-总体设计.md) | 电脑 ↔ PCIe ↔ FPGA 全流程、接口选型、分层仿真路线 |
| [02 非线性算子原理](docs/02-非线性算子原理.md) | Softmax、RMSNorm、SiLU、RoPE 在 FPGA 上的实现方法 |
| [03 L0 定点参考模型](docs/03-L0-定点参考模型.md) | 数值格式、各运算的定点算法与误差、位宽需求 |
| [04 L1 性能模型](docs/04-L1-性能模型.md) | 运算量、各架构耗时、各模型 × 各板卡速度 |
| [05 L2 矩阵单元](docs/05-L2-矩阵单元.md) | MVM 计算核、AXI 接口、数据流格式、整模型调度 |
| [06 L2 向量单元](docs/06-L2-向量单元.md) | 运算流、7 种运算的 RTL、整模型 RTL 仿真结果 |
| [07 RTL 可综合与时序](docs/07-RTL-可综合与时序.md) | 位宽限定、流水线、乘法拆分、资源与时序估算 |
| [08 开发环境](docs/08-开发环境-Mac写代码Windows编译.md) | Mac 写代码、Windows 上跑 Vivado 的远程方案 |
| [09 设计思路与验证](docs/09-设计思路与验证.md) | 整体怎么考虑、做了哪些测试、每类测试证明了什么、还没覆盖什么 |

## 目录

```
l0/        Python 参考模型（浮点版、硬件等价定点版、分词器、权重读写）
l1/        性能模型 perf.py，报告 report.md
l2/        cocotb 驱动、调度器、测试；gen_luts.py 从 L0 生成 RTL 查找表
rtl/       SystemVerilog：矩阵单元、向量单元、乘法流水线、仿真顶层
synth/     yosys 综合统计（*_xc7.txt）与时序估算脚本 sta_est.py
tests/     L0 / L1 单元测试
docs/      设计文档
models/    模型文件（不入库，见下）
```

## 环境

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pytest cocotb cocotbext-axi
brew install verilator yosys        # macOS；Linux 用发行版的包
```

需要 Verilator 5.x（测试用 5.054）、cocotb 2.x、yosys（仅资源估算用）。

模型文件放在 `models/`（不入库）：

```bash
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
```

没有模型文件时，大部分测试用随机权重的小模型，照样能跑。

## 运行

```bash
# 全部测试（含 RTL 仿真，约 10 秒）
.venv/bin/python -m pytest -q

# L0：浮点版 vs 硬件等价版，生成文本并打印位宽需求
.venv/bin/python -m l0.run --mode compare --prompt "Once upon a time" --steps 64
.venv/bin/python -m l0.run --random --mode compare      # 无需模型文件

# L1：性能报告
.venv/bin/python -m l1.perf

# L2：整模型在 RTL 上推理并逐步与 L0 比对（约 10 s/token）
L2_GEN_MODEL=stories15M L2_GEN_STEPS=32 .venv/bin/python -m pytest -q -s l2/test_l2.py::test_generate_rtl
#   L2_GEN_VPU=0：非线性运算留在电脑端，只有矩阵乘走 RTL
#   L2_FULL=1 pytest l2：回放一个 token 的全部 43 次矩阵乘

# 综合与粗略时序估算（yosys，7 系列）
for t in mvm_top vpu_quant vpu_rmsnorm vpu_softmax vpu_ew vpu_rope attn_head; do
  yosys -q -p "read_verilog -sv -Irtl rtl/mul_pipe.sv rtl/vpu_units.sv rtl/mvm_core.sv rtl/mvm_top.sv; \
    synth_xilinx -family xc7 -top $t -flatten; tee -q -o synth/${t}_xc7.txt stat; write_json synth/${t}_xc7.json"
done
.venv/bin/python synth/sta_est.py synth/*_xc7.json
```

## 设计要点

**数值格式（硬件契约）**：激活值 Q16.16；权重 int8 + 每行缩放系数（Q24）；送进矩阵单元的激活值量化为 int8 + 2 的幂移位量（不需要除法）；统一"加半个 LSB 再右移"舍入；1/√x、1/x 用查表 + 2 次牛顿迭代，exp 用 2 的幂 + 查表插值。

**一份模型流程**：`l0/ref_hw.py` 的 `HWModel.forward_steps` 把前向计算拆成 7 种运算（matmul / quant / rmsnorm / rope / attn / silu_gate / resadd）依次交出。L0 用定点函数执行，L2 调度器交给 RTL 执行，两边不会因为代码不同步而出错。这个运算流就是将来片上调度器的指令流。

**接口**：AXI4-Lite 寄存器 + AXI-Stream 数据流，对应 PCIe XDMA 暴露给用户逻辑的接口；仿真时 cocotbext-axi 扮演"电脑 + XDMA"，上板时 RTL 不变。

**验证方式**：每个 RTL 单元都与 L0 对应函数逐比特比对（随机 + 边界值 + 反压），整模型逐 token 比对 logits。

## 后续计划

1. 继续压注意力头 / Softmax 的关键路径；写 Vivado 非工程模式脚本拿真实时序
2. 片上调度器：在 FPGA 上执行运算流，电脑每个 token 只发一条命令
3. 从板载 DDR 读权重的 DMA
4. 预填充用的 GEMM 阵列
5. 选板、XDMA 集成、上板
