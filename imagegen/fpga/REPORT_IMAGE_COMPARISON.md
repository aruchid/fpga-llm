# 生图结果对比报告

同一提示词、同一随机种子，比较 6 个版本的输出：

> **提示词**：a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic
> **配置**：512×512，4 步 Euler，guidance = 0（SD-Turbo 免 CFG），seed = 42

![六版对比](E:\FPGALLM\imagegen\out\report_contact_sheet.png)

---

## 1. 量化指标

以 **A（diffusers fp32）** 为参考：

| 版本 | PSNR | SSIM | 平均像素差 | 视觉评价 |
|---|---|---|---|---|
| A. diffusers FP32（参考） | — | — | — | 基准 |
| B. diffusers FP16 | **50.78 dB** | 0.9997 | 0.0012 | 与 A 几乎无法区分 |
| C. ComfyUI FP | 9.11 dB | 0.111 | 0.278 | **构图不同**（随机数源不同），但同样是雪夜木屋 |
| D. INT8（UNet 量化、VAE 浮点） | **22.01 dB** | 0.869 | 0.043 | 结构/色调一致，细节略软 |
| E. INT8 + 32×32 tile（UNet+VAE 全量化） | **22.00 dB** | 0.868 | 0.043 | 与 D 几乎无差 |
| F. 早期失败版（采样器 eps/v-pred 误判） | 11.39 dB | 0.215 | 0.230 | 塌成纯色 |

---

## 2. 逐项解读

### A → B：fp32 换 fp16，几乎无损（50.78 dB）

diffusers 默认 fp16，我们把管线转成 fp32 作为量化参考。两者 PSNR 50.78 dB、SSIM 0.9997，
属于**同一次计算的不同数值精度**，不构成质量差异。后续量化对比都以 fp32 为准。

### A → C：ComfyUI 的图，指标很差但不是坏了（9.11 dB）

ComfyUI 那张（第一轮生成），PSNR 只有 9.11 dB、SSIM 0.11，**原因是初始噪声不同**：

- ComfyUI 的 KSampler 用 `torch.Generator`（CUDA 上）生成噪声
- diffusers 用 `torch.Generator("cuda").manual_seed(42)`，序列不同

扩散模型对初始噪声极其敏感，噪声不同 → 构图完全不同（C 里木屋偏右、树更多、多了一条雪径）。
所以这张**不能做像素级比较**，只能作为「同一提示词的另一张合理结果」参考。
这也解释了为什么之前用 ComfyUI 的图做参考时，怎么对都对不上——不是实现对错，是噪声不同。

### A → D：UNet 量化到 int8，损失 22.01 dB

这是**真正的量化损失**：权重 per-channel int8、激活 per-row int8、矩阵乘 int8×int8→int32。

- SSIM 0.869 说明结构高度保持
- 平均像素差 0.043（约 11/255）
- 视觉上：雪面细节变平滑、树枝纹理略糊、天空渐变出现轻微色带

这类损失来自激活量化（每行一个 scale，8 bit），是 W8A8 的典型水平。

### D → E：再把 VAE 也量化，几乎无影响（45.38 dB）

D 的 VAE 是浮点，E 的 VAE 解码器也换成了 int8 + 32×32 tile。两者差 45.38 dB——
**VAE 量化在整链里可以忽略**。原因见下一节：VAE 单独量化本来就有 29 dB，而 UNet 的
量化损失远大于它，两者叠加后 VAE 部分被淹没。

### VAE 单独量化：29.22 dB（同 latent 的隔离实验）

用**同一个 latent** 分别做浮点解码和定点解码：

| 对比 | PSNR | SSIM |
|---|---|---|
| ComfyUI 浮点解码 vs 我们 numpy 定点解码 | **29.22 dB** | 0.949 |
| ComfyUI 浮点解码 vs 我们 torch 定点解码 | 29.22 dB | 0.949 |

这是干净的「只量化 VAE」实验：单看 29 dB 不算高，但在整链里只占很小一部分误差。

### A → F：失败案例（11.39 dB）

早期版本把 SD-Turbo 的预测类型搞错（按 v-prediction 解算，实际 scheduler 是 epsilon），
latent 在第一步之后塌缩，最终输出纯色。

排除过程值得记录：
- 我们的 UNet 与 ComfyUI 的 UNet **直接对拍 cos = 0.99958**（实现是对的）
- 但采样后第 2 步的 latent 与 ComfyUI 差 17 倍，且我们的模型输出与噪声不相关（0.001），
  ComfyUI 反推的 eps 却相关 0.995
- 原因：ComfyUI 的 `apply_model` 会做 `model_sampling` 的 denoise 变换，不是裸 eps

最终改用 diffusers 的 `EulerDiscreteScheduler` 负责采样语义，问题消失。

---

## 3. 误差分解

以 A（fp32）为基准，各环节的贡献：

| 环节 | 单独影响 | 整链占比 |
|---|---|---|
| fp16 精度 | 50.78 dB | 可忽略 |
| UNet int8 量化 | ~22 dB | **主导** |
| VAE int8 量化 | 29.22 dB（隔离） | 被 UNet 淹没（D→E 45.4 dB） |
| 32×32 tile 引擎 | **0（逐比特相同）** | 无 |

关键结论：**32×32 矩阵单元本身不引入任何误差**——D 与 E 的差异全部来自 VAE 量化，
而 `MVM32Tiled` 与 `torch._int_mm` 在多种形状上逐比特一致（最大差 0）。
整链 22 dB 的损失全部来自「把权重/激活压到 int8」这一步。

---

## 4. 速度

| 版本 | 软件耗时（RTX 5060） | 相对 fp16 |
|---|---|---|
| B. diffusers FP16 | 0.8 s | 1× |
| A. diffusers FP32 | 0.96 s | 1.2× |
| D. INT8（`_int_mm`） | 20.5 s | 26× |
| E. INT8 + 32×32 tile | 117 s | 147× |
| 真 FPGA（32 路 @115 MHz，估算） | ~713 s | ~890× |

软件变慢的主因不是 int8，也不是 32×32 分块，而是：
- **im2col / 逐层量化用 Python 做**（约占 70%）
- tile 引擎用 float64 逐块累加保证整数精确（约占 10–20%）
- 纯矩阵乘只要 0.3 s（2596 GMAC ÷ 8.6 TMAC/s）

FPGA 的 713 s 是算力下限：32 路 × 115 MHz = 3.68 GMAC/s，2596 GMAC 只能这么快。

---

## 5. 结论

1. **INT8 全链路可用**：PSNR 22 dB、SSIM 0.87，视觉上与原图高度一致，只是细节略软。
2. **32×32 矩阵引擎零误差**：与 IMMA 逐比特一致，换引擎不改变任何像素。
3. **量化损失主要在 UNet**，VAE 量化几乎无感（45 dB）。
4. **速度对比要分开看**：软件 147× 是 Python 实现开销（可优化到 0.3–0.5 s）；
   硬件 890× 是 32 路算力的物理下限（需加宽阵列解决）。
