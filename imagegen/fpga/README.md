# 用 32 路矩阵单元替换 torch —— 进度

目标：用 fpga-llm 的 `rtl/mvm_core.sv`（N=32）的**逐比特等价软件模拟器**替代 PyTorch 的
矩阵乘，跑通 SD-Turbo 的 CLIP → UNet×N → VAE 全链路，对保留的提示词出单张 512×512 图。

没有 Verilator，所以是行为级模拟（bit-exact），周期数由 `mvm_core.sv` 状态机解析估算，
不是真实 RTL 仿真。

## 硬件契约（来自 `l0/fixedpoint.py` + `rtl/mvm_core.sv`）

    y = sat32(round((W · x) · SW >> (24 - s)))

- W：int8 对称量化，每输出行一个 SW（Q24）
- x：int8 + 每输入向量一个移位量 s
- 累加：int32（按 32 位环绕），32×32 tile 分块，加法精确所以与分块无关
- 舍入：加半个 LSB 再算术右移

## 已完成

### 1. 引擎 `mvm32.py`

- `MVM32.gemm / dot_i8`：按 32 路、32×32 tile、int32 环绕、SW 缩放、饱和输出
- 与仓库 `l0/fixedpoint.int_matmul` **逐比特一致**（`test_mvm32.py`：量化、批量列、s=0、
  分块累加等价、饱和触发、周期公式全部 PASS）
- 加速技巧：numpy 的 int32 矩阵乘是慢速循环，改用 float64 BLAS 累加；只要
  `K·127² < 2^53`，整数在 float64 中精确可表示，求和顺序不影响结果 → 仍然逐比特一致
- 周期模型：`nt·(kb·N + N/8)` 拍，即每拍一行权重（32 个 int8）

### 2. 定点原语 `fx.py`

`l0` 里标量版 `rsqrt/recip` 向量化后与标量版**逐比特一致**；`softmax/sigmoid/silu` 复用；
GroupNorm 按 PyTorch 语义跨「空间位置 × 组内通道」统计，均值保 Q16 精度，误差 1.3e-4。

### 3. 质量闸门：VAE 解码（`sd_vae.py`）

用同一张图的 latent 做对照：

| 实现 | 对照 | PSNR |
|---|---|---|
| numpy 浮点解码 | ComfyUI float 解码 | **51.63 dB** |
| 32 路定点解码（全部卷积走引擎） | ComfyUI float 解码 | **29.22 dB** |

- 定点路径：1257 GMAC、3864 万 tile、0 次饱和
- 软件耗时 302 s；按 115 MHz、32 路估算需 **344.7 s**——两者几乎相同，说明瓶颈就是
  「1 MAC/lane/cycle」的算力，软件模拟没有额外折扣
- 视觉差异集中在雪地/树枝高频区域（见 `out/vae_compare.png`，右图为放大差异）

结论：**int8 权重 + Q16.16 激活 + 每像素一个 s 的方案能把 SD VAE 解码出来**，画质可用。

## 待做

1. `sd_clip.py`：CLIP-L 文本编码器定点实现（12 层、768 维），tokenizer 用
   `E:\ComfyUI\comfy\sd1_tokenizer\{vocab.json,merges.txt}`
2. `sd_unet.py`：SD-Turbo UNet 定点实现（down/mid/up、cross-attn、时间嵌入、skip 连接）
3. Euler 离散 4 步采样 + VAE 解码，端到端出图，与 ComfyUI 的同提示词结果对比

### 算力预算（估算）

UNet 512×512 每步约 0.7–0.9 GMAC，4 步约 3–3.6 GMAC，加 VAE 1.26 GMAC：

- 软件模拟（实测 4.2 GMAC/s）：约 15–20 分钟/张
- 115 MHz × 32 路真硬件：约 20 分钟/张（同一瓶颈）
- 1 步采样可把 UNet 部分压到 1/4

## 文件

| 文件 | 作用 |
|---|---|
| `mvm32.py` | 32 路矩阵单元行为级模拟器 + 周期模型 |
| `fx.py` | 定点原语（Q16.16、LUT、GroupNorm） |
| `sd_vae.py` | VAE 解码器：浮点版 + 定点版 |
| `get_ref.py` | 从 ComfyUI 取 latent / float 解码作为对照 |
| `test_mvm32.py` / `test_fx.py` / `test_vae.py` | 逐比特与质量测试 |
