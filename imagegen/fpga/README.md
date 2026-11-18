# fpga/ — 32×32 矩阵推理

把 SD-Turbo 的推理换到 `rtl/mvm_core.sv`（N = 32）的 int8 矩阵单元上。

## 硬件契约

```
y = sat32( round( (W · x) · SW >> (24 − s) ) )
```

- **W**：int8 对称量化，每输出行一个缩放 SW（Q24）
- **x**：int8 激活，每输入向量一个移位量 s
- **累加**：int32，按 32 位环绕；32×32 tile 分块累加（整数加法与分块无关）
- **舍入**：加半个 LSB 再算术右移
- **周期模型**：`cycles = B × ⌈M/32⌉ × (⌈K/32⌉ × 32 + 4)` ≈ `MACs / 32`

## 文件

| 文件 | 作用 |
|---|---|
| `mvm32.py` | 32×32 矩阵引擎（含 Q16.16 定点的 SW/移位路径）+ 周期模型 |
| `mvm_tiled.py` | **严格 32×32 tile 引擎**（最终执行路径）：输出按 32 行、K 按 32 元素逐块累加 |
| `q8.py` | INT8 量化原语：per-channel 权重、per-row 激活、int8 GEMM（IMMA） |
| `quant_unet.py` | 主流程：把 diffusers 的 UNet / VAE 换成 INT8，`--tile` 走 32×32 引擎 |
| `test_mvm32.py` | 契约测试：`mvm32` 与 `l0/fixedpoint.int_matmul` 逐比特对比 |
| `make_report.py` | PSNR / SSIM 指标 + 六版对比拼图 |
| `build_report.py` | Markdown → 自包含 HTML → PDF（Edge 无头打印） |
| `INFERENCE_FLOW.md` | 推理流程说明（模型、链路、算子清单、量化方案） |
| `INT8_RESULT.md` | INT8 结果与 tile 引擎数据 |
| `REPORT_32MATRIX.md` | 完整工程报告（替换量、正确性、精度、性能） |
| `REPORT_IMAGE_COMPARISON.md` | 图像对比明细 |

## 用法

```powershell
python test_mvm32.py                      # 契约测试
python quant_unet.py --vae8               # INT8：UNet + VAE 量化（约 50 s）
python quant_unet.py --tile --vae8        # 全链路 32×32 tile 引擎（约 2 min）
python make_report.py                     # 指标 + 对比拼图
python build_report.py                    # 生成 HTML / PDF
```

## 已验证的结论

| 项 | 结果 |
|---|---|
| 引擎 vs `l0/fixedpoint.int_matmul` | 逐比特一致，最大差 0 |
| tile 引擎 vs `torch._int_mm`（4 种真实形状） | 逐比特一致，最大差 0 |
| 全链路出图 vs int8 参考 | 逐像素相同（不同像素 0） |
| 与 FP32 的 PSNR | 22.00 dB（损失全来自 int8 量化） |
| 替换量 | 2596.5 / 2619.9 GMAC = **99.1%** |
| 硬件规模 | 79.7 M tile、81.65 G 周期、710 s @115 MHz（32 路） |

> 早期尝试过完整 Q16.16 定点路线（CLIP/UNet/VAE 全部定点）。结论：CLIP 因 LayerNorm 输出的
> 离群通道（单通道 16–26 vs 典型 0.005）不可整行定点；UNet 定点后精度不足。最终改用
> INT8 W8A8（per-channel 权重 + per-row 激活），相关实验代码已清理，结论保留在本文档与
> `INFERENCE_FLOW.md` 中。
