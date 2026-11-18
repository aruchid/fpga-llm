# INT8 SD-Turbo 生图结果

## 结论

用 **INT8 量化权重 + INT8 动态激活 + int8×int8→int32 累加** 跑通了 SD-Turbo 的完整推理，
对保留提示词生成了 512×512 图像：

> a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic

| 指标 | 值 |
|---|---|
| 与 FP32 参考的 PSNR | **22.01 dB** |
| 平均绝对像素差 | 0.0427 |
| UNet 算力（4 步） | **1356 GMAC**（339 GMAC/步） |
| 软件耗时（RTX 5060） | FP32 0.94 s / INT8 2.1 s / INT8+32×32 tile 3.0 s |
| 32 路引擎等效耗时 | 1356 GMAC ÷ 3.68 GMAC/s ≈ **369 s** @115 MHz |

图像对比见 `out/diffusers_fp_vs_int8.png`（左 FP32、右 INT8）。

## 量化方案

| 部分 | 处理 |
|---|---|
| 权重 | **per-channel int8**（每个输出通道一个 scale） |
| 激活 | **per-row int8**（每个像素 / 每个 token 一个 scale，与 32 路矩阵单元的「每向量一个移位量」对应） |
| 矩阵乘 | **int8 × int8 → int32 累加**（`torch._int_mm`，即 IMMA 指令） |
| 反量化 | `acc × (s_w × s_a) + bias`，在向量单元里做 |
| GroupNorm / SiLU / softmax | 保持原样（不量化） |

## 与 32 路矩阵单元的一致性

在真实卷积尺寸（320→320、3×3、64×64，3.77 GMAC）上验证：

```
torch._int_mm 结果  ==  CPU int32 累加  ==  引擎 dot_i8 结果     最大差 0
```

即：量化推理里的每一次矩阵乘，**就是** `rtl/mvm_core.sv` 执行的运算。
整条链的位精确性由此传递（相同的 int8 输入 → 相同的 int32 累加 → 相同的反量化结果）。

## 复现

```powershell
# 1) 下载 diffusers 格式的 SD-Turbo（ModelScope，约 2.5 GB）
python fetch_model.py --dest models/sd-turbo-diffusers

# 2) INT8 量化 + 生成 + 与 FP32 对比
python fpga/quant_unet.py
```

## 备注

- 采样语义（scheduler、v-pred/eps、sigma 序列）全部交给 diffusers，不再手写，避免此前
  SD-Turbo 到底是 eps 还是 v-pred 的反复。
- ComfyUI 的 GGUF 路径实测是「反量化成 fp16 再算」，不是 int8 运算，因此只适合做权重精度参考，
  不能作为 int8 矩阵推理的验证环境。
- 当前 INT8 版本的 20.5 s 主要是 Python 逐层 im2col 的开销；同样的算力在 `torch._int_mm`
  上只需约 0.1 s，瓶颈完全在数据组织而不在矩阵乘。

---

## 9. 用 32×32 tile 引擎重构（已完成）

把量化推理里的每一次矩阵乘从 `torch._int_mm` 换成 `mvm_tile.MVM32Tile`：
按 RTL 的方式把 K 切成 32 元素的块逐个累加（`acc += W[:, k0:k0+32] · X[k0:k0+32, :]`），
累加在 float64 里做（整数精确），M 方向按 32 行分 tile 统计周期。

**结果：输出图像与 torch `_int_mm` 版本逐像素完全相同（512×512×3，不同像素数 = 0）。**

| 指标 | 值 |
|---|---|
| tile 总数（4 步 UNet） | 41.6 M |
| 周期数 | 42.55 G |
| 等效硬件耗时 | **370 s @115 MHz, 32 路** |
| 软件耗时 | 31.3 s（tile 版）vs 15.0 s（`_int_mm` 版） |

整张图的硬件耗时估算（32 路 / 115 MHz）：

| 阶段 | GMAC | 等效耗时 |
|---|---|---|
| CLIP 编码 | 23 | ~6 s |
| UNet ×4 步 | 1356 | ~370 s |
| VAE 解码 | 1257 | ~342 s |
| **合计** | **2636** | **~718 s ≈ 12 分钟/张** |

这就是当前 32 路设计的端到端算力下限；换成 128 路约 3–4 分钟，256 路约 2 分钟。
---

## 10. 严格 32×32 tile 引擎（全链路替换）

`mvm_tiled.py` 把切分做成真正的计算而不是统计：输出按 32 行一 tile、K 按 32 个元素一块，
`for 每个 K 块: for 每个输出 tile: acc[t] += W[t, kk] · X[kk]`（每个 tile 是 32×32 乘加），
累加结果按 32 位环绕转 int32，与 RTL 累加器一致。

把 UNet 和 **VAE 解码器**都换成这个引擎后：

| 指标 | 值 |
|---|---|
| 总 MACs（UNet 4 步 + VAE 解码） | **2596.5 GMAC**（UNet 1356 + VAE 1240） |
| tile 总数 | **79.7 M** |
| 周期数 | **81.65 G** |
| 等效硬件耗时 | **710 s @115 MHz / 32 路** |
| 与 FP32 的 PSNR | **22.00 dB** |
| 软件耗时 | 85 s（瓶颈是 Python 逐层 im2col） |

已验证 `MVM32Tiled.gemm_int8` 在多种真实形状上与 `torch._int_mm`（IMMA）**逐比特一致**，
所以整条链的结果与「int8 硬件真算」相同。

剩余未量化的只有 CLIP 文本编码器（23 GMAC，占全链 0.9%），跑在浮点上。

### 全链硬件下限（32 路 / 115 MHz）

| 阶段 | GMAC | 等效耗时 |
|---|---|---|
| CLIP（浮点） | 23 | ~6 s |
| UNet ×4 步（int8 tile） | 1356 | ~370 s |
| VAE 解码（int8 tile） | 1240 | ~337 s |
| **合计** | **2619** | **~713 s ≈ 12 分钟/张** |