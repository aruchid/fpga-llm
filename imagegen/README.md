# imagegen — SD-Turbo 生图与 32×32 矩阵推理

在 SD-Turbo（512×512、4 步）上验证「用 32×32 int8 矩阵单元替代 GPU 矩阵运算」的可行性，
包含完整的推理链路、量化方案、替换量统计与对比报告。

## 环境

| 项 | 值 |
|---|---|
| Python | `D:\anaconda3\envs\pt_2026`，torch 2.11.0+cu128 |
| GPU | RTX 5060 8 GB |
| 模型 | SD-Turbo（SD 2.1 架构），diffusers 格式，见下 |
| 基线环境 | ComfyUI 0.45（可选，仅用于 `comfy_gen.py`） |

## 快速开始

```powershell
# 1) 下载 diffusers 格式模型（ModelScope，约 2.6 GB）
python fetch_model.py --dest models/sd-turbo-diffusers

# 2) 契约测试：32×32 引擎与 fpga-llm 的定点契约逐比特一致
python fpga/test_mvm32.py

# 3) FP32 参考 + INT8 量化（UNet + VAE），约 50 s
python fpga/quant_unet.py --vae8

# 4) 全链路 32×32 tile 引擎，约 2 分钟
python fpga/quant_unet.py --tile --vae8

# 5) 生成对比图表与 PDF 报告
python fpga/make_report.py
python fpga/build_report.py
```

产物在 `out/`：`diffusers_fp32_seed42.png`（参考）、`diffusers_int8_seed42.png`（INT8）、
`diffusers_int8_tile_seed42.png`（32×32 tile）、`report_contact_sheet.png`、`report_32matrix.pdf`。

## 目录

```
imagegen/
├── comfy_gen.py            ComfyUI API 生图（基线，可选的对照路径）
├── fetch_model.py          从 ModelScope 下载 diffusers 格式 SD-Turbo
├── models/                 模型权重（不入库，由 fetch_model.py 生成）
├── fpga/
│   ├── mvm32.py            32×32 矩阵引擎（与 rtl/mvm_core.sv、l0/fixedpoint 逐比特一致）
│   ├── mvm_tiled.py        严格 32×32 tile 引擎（最终执行路径）
│   ├── q8.py               INT8 量化原语（per-channel 权重 / per-row 激活 / int8 GEMM）
│   ├── quant_unet.py       主流程：把 diffusers 的 UNet / VAE 换成 INT8 + tile 引擎
│   ├── test_mvm32.py       契约测试：引擎 vs fpga-llm 的定点契约
│   ├── make_report.py      指标计算（PSNR/SSIM）+ 六版对比拼图
│   ├── build_report.py     Markdown → 自包含 HTML → PDF
│   ├── INFERENCE_FLOW.md   推理流程说明
│   ├── INT8_RESULT.md      INT8 方案与结果
│   ├── REPORT_32MATRIX.md  完整工程报告（含替换量与性能分析）
│   └── REPORT_IMAGE_COMPARISON.md  图像对比明细
└── out/                    生成结果与报告（PNG / PDF）
```

## 关键结论

| 项 | 值 |
|---|---|
| 全链算力 | 2619.9 GMAC（CLIP 23.5 + UNet 1356.4 + VAE 1240.1） |
| 走 32×32 矩阵 | **2596.5 GMAC（99.1%）** |
| 与 FP32 的 PSNR | **22.00 dB** |
| tile 引擎正确性 | 与 IMMA 逐比特一致，出图逐像素相同 |
| 硬件规模 | 79.7 M tile / 81.65 G 周期 / 710 s @115 MHz（32 路） |

详见 `fpga/REPORT_32MATRIX.md`。

## 说明

- 早期曾尝试完整 Q16.16 定点路线（CLIP/UNet/VAE 全部定点），验证后判定 CLIP 因离群通道
  不可定点、UNet 定点需要更细的量化粒度，遂转向 INT8 W8A8；相关实验代码已清理，
  结论保留在 `fpga/INFERENCE_FLOW.md` 与 `fpga/INT8_RESULT.md` 中。
- `models/` 与 `out/*.npy`、`out/*.html` 不入库（见 `.gitignore`）。
