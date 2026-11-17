# 当前推理流程说明（重构前的基线）

本文档描述**现在这套能出图的 INT8 推理**是怎么跑的，作为接下来「用 32×32 矩阵单元重构」的基准。
所有数字均为本机实测（RTX 5060 8GB / i5-12600KF / 32GB）。

---

## 1. 模型组成

SD-Turbo（Stability AI），架构属于 **SD 2.1**（不是 SD1.5），diffusers 格式：

| 组件 | 参数量 | 作用 | 输入 → 输出 |
|---|---|---|---|
| CLIP 文本编码器（OpenCLIP ViT-H） | 354.0 M | 提示词编码 | token ids (77) → context (77×1024) |
| UNet | 865.9 M | 噪声预测（4 步） | latent 64×64×4 + t + context → 64×64×4 |
| VAE 解码器 | 83.7 M | latent 解码成像素 | 64×64×4 → 512×512×3 |

权重总量 1.30 B 参数：int8 约 **1.21 GB**，fp32 约 4.86 GB。

---

## 2. 总流程

```
提示词 "a cozy cabin in a snowy pine forest at dusk, ..."
   │
   ├─[1] CLIP 分词：BPE → 77 个 token（开头 49406，结尾 49407，其余用 0 填充）
   │        → CLIP 24 层 Transformer（causal）→ 取倒数第二层，做 final LayerNorm
   │        → context (77 × 1024)
   │
   ├─[2] 初始噪声：x ~ N(0,1)（seed 固定，torch.randn）
   │        → x × sqrt(1 + σ₀²)，σ₀ = 14.6146
   │
   ├─[3] 4 步去噪（Euler，eps 预测，guidance = 0 即不做 CFG）
   │        timesteps: 999, 666, 333, 0
   │        sigmas:   14.6146, 2.9183, 0.9324, 0.0292, 0
   │        每步：  eps = UNet(x, t, context)
   │                x₀  = x − σ·eps
   │                d   = (x − x₀)/σ = eps
   │                x   = x + (σ_next − σ)·d
   │
   └─[4] VAE 解码：x / 0.18215 → 解码器 → 512×512×3 RGB
```

**关键点**：模型是 **epsilon 预测**（diffusers 的 `scheduler.config.prediction_type` 就是 `epsilon`）；
之前手写采样器时误按 v-prediction 算，是导致「latent 塌成纯色」的根因。

采样语义（sigma 表、步长、缩放）现在完全交给 diffusers 的 `EulerDiscreteScheduler`。

---

## 3. 各阶段的算子清单

### 3.1 CLIP 文本编码器（354 M）

- 24 层 Transformer，隐藏维 1024，16 头（每头 64），MLP 4096
- 每层：LayerNorm → QKV 线性（1024→3072）→ causal 注意力 → out 投影（1024→1024）
  → LayerNorm → FF（1024→4096→1024）
- 算力约 **23 GMAC**（77 token，只跑一次）

### 3.2 UNet（866 M，主要算力）

结构（latent 64×64 起）：

```
conv_in (4→320, 3×3)
down 0: res(320→320) + attn  ×2, downsample(320)
down 1: res(320→640) + attn  ×2, downsample(640)
down 2: res(640→1280) + attn ×2, downsample(1280)
down 3: res(1280→1280) ×2               [8×8，无注意力]
mid   : res(1280) + self-attn + cross-attn + res(1280)
up 3  : res ×3 (+skip) , upsample
up 2  : res ×3 (+skip) + attn, upsample
up 1  : res ×3 (+skip) + attn, upsample
up 0  : res ×3 (+skip) + attn
out   : GroupNorm + SiLU + conv(320→4, 3×3)
```

每个 **resblock**：GroupNorm → SiLU → conv3×3 → 加时间步嵌入 → GroupNorm → SiLU → conv3×3 → 残差。
时间步嵌入：sinusoidal(320) → Linear(320→1280) → SiLU → Linear(1280→1280)，每个 resblock 再 Linear(1280→C)。

每个 **spatial transformer**：LayerNorm/GroupNorm → proj_in → **self-attn** → **cross-attn**（K/V 来自 context）
→ GEGLU 前馈（C→4C×2→4C）→ proj_out。

**算力实测：4 步合计 1356 GMAC（339 GMAC/步）**，其中卷积占大头、注意力/FF 次之。

### 3.3 VAE 解码器（84 M）

conv_in → mid（res + 单头自注意力 + res）→ 4 个 up 块（每块 3 个 resblock + 上采样卷积）→
GroupNorm + SiLU + conv_out。算力 **1257 GMAC**（512×512 输出的主要开销在 512² 分辨率的卷积上）。

---

## 4. 量化方案（当前 INT8 版本）

| 部分 | 方式 | 说明 |
|---|---|---|
| 权重 | **per-channel int8** | `w_q = round(w / (max|w|/127))`，每个输出通道一个 scale |
| 激活 | **per-row int8** | 每个像素（卷积）或每个 token（线性）一个 scale |
| 矩阵乘 | **int8 × int8 → int32 累加** | `torch._int_mm`（IMMA） |
| 反量化 | `y = acc · (s_w · s_a) + bias` | 在向量单元里做，浮点 |
| GroupNorm / SiLU / softmax / GEGLU | **不量化**，保持浮点 | 这些不是矩阵运算 |

与 fp32 相比：**PSNR 22.01 dB**，平均像素差 0.043（视觉上几乎一致，略软）。

---

## 5. 矩阵乘怎么落到「32×32 矩阵单元」

现在的实现里，所有矩阵乘都归结为一个形状：

```
Y(M×B) = W(M×K) · X(K×B)      W、X 为 int8，累加 int32
```

- **线性层**：M = 输出通道/特征维，K = 输入维，B = 行数（像素数或 token 数）
- **3×3 卷积**：先把输入展开成 im2col 矩阵（K = 9·C，B = H·W），再同上
- **注意力打分**：q·kᵀ，M = 关键数，K = 头维，B = 查询数

对应到 `rtl/mvm_core.sv`（N = 32）：

```
按 32×32 的 tile 切分：ceil(M/32) × ceil(K/32) 个 tile
每个 tile：32 行权重 × 32 个输入元素的点积累加到 int32
权重以 AXI-Stream 逐行送入（每拍 32 个 int8 = 一行）
每行一个缩放系数 SW（Q24），每个输入向量一个移位量 s
输出：y = sat32(round((W·x)·SW >> (24−s)))
```

实测：在真实卷积尺寸（320→320、3×3、64×64，3.77 GMAC）上，
`torch._int_mm` 的结果与 CPU int32 累加、与引擎的 `dot_i8` **逐比特相同（最大差 0）**。

---

## 6. 数据流与资源

| 项 | 数值 |
|---|---|
| 权重（int8） | UNet 0.81 GB + CLIP 0.33 GB + VAE 0.08 GB = 1.21 GB |
| 每步激活峰值（64×64 latent） | 约 84 MB（int64 模拟）/ 42 MB（int32 硬件） |
| VAE 解码峰值张量（512×512×128） | 268 MB（int32） |
| 片上 tile buffer（32×32 int8） | 1 KB/行 × 32 行 = 1 KB（权重）+ 32 B（输入向量） |

---

## 7. 当前实测性能

| 阶段 | 时间（RTX 5060） | 备注 |
|---|---|---|
| CLIP 编码 | 0.05 s | 23 GMAC |
| UNet ×4 步（FP32） | 0.97 s | 含 VAE 解码 |
| UNet ×4 步（INT8，Python 逐层 im2col） | 20.5 s | 瓶颈是 Python 组织数据，不是矩阵乘 |
| VAE 解码（INT8，torch） | 18.9 s | 同上 |

**32 路引擎等效耗时**：1356 GMAC ÷ (32 × 115 MHz = 3.68 GMAC/s) ≈ **369 s**（仅 UNet 4 步）。

---

## 8. 重构计划（下一步）

把这套流程里的每一次矩阵乘，从 `torch._int_mm` 换成**按 32×32 tile 显式调度的引擎**：

1. `mvm_tile.py`：实现 tile 级 GEMM —— 按 `ceil(M/32)×ceil(K/32)` 切块，逐 tile 做
   32×32 的 int32 累加，权重按行流式读取，统计 tile 数与周期。
2. 把 `q8.mm8` 替换成 tile 引擎（接口相同），其余（量化、反量化、norm、softmax）不动。
3. 验证：输出图像与当前 INT8 结果**逐比特相同**（因为累加顺序不影响整数结果），
   同时给出 tile 数、周期数、以及按 115 MHz / 32 路的等效硬件耗时。
4. 报告：FP32 / INT8-torch / INT8-tile引擎 三方对比 + 资源估算。

> 之所以能保证「换引擎不改结果」：int8 乘积是整数，int32 累加满足结合律，
> 32×32 的分块累加与整块累加结果完全相同（已在 `test_mvm32.py` 里验证过）。
