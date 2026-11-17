# imagegen — 生图模型跑通记录

复用本机已有的 ComfyUI 环境跑 **SD-Turbo** 文生图，不下载任何权重。

## 环境（实测）

| 项 | 值 |
|---|---|
| 服务 | ComfyUI 0.45.0，`http://127.0.0.1:8188`（已在运行） |
| Python | `D:\anaconda3\envs\pt_2026`，torch 2.11.0+cu128 |
| GPU | RTX 5060 8GB（驱动 576.88，CUDA 12.8） |
| 权重 | `E:\ComfyUI\models\checkpoints\sd_turbo.safetensors`（4.86 GB） |

本机另有完整 Z-Image-Turbo 一套（`diffusion_models\z_image_turbo_bf16` 11.46 GB + `text_encoders\qwen_3_4b` 7.49 GB + `vae\ae`），8GB 显存需 offload，未在本轮验证。

## 用法

```powershell
# 生成一张（4 步，512×512）
python comfy_gen.py --prompt "a red fox sitting in fresh snow, soft morning light" --steps 4 --seed 100

# 1 步极速
python comfy_gen.py --prompt "..." --steps 1 --size 512
```

出图落在 `out/seed<seed>_s<steps>_<size>px.png`。脚本只依赖标准库 `urllib`，用系统 Python 3.14 即可运行，不需要 torch。

## 实测速度（512×512，模型已在显存/内存中）

| 步数 | 端到端耗时 |
|---|---|
| 4 步 | ~1.05 s |
| 1 步 | ~0.54 s |
| 首次生成（含加载 4.86 GB 权重） | ~4.0 s |

## 文件

- `comfy_gen.py` — 主路径：拼 txt2img 图（CheckpointLoaderSimple → CLIPTextEncode×2 → EmptyLatentImage → KSampler → VAEDecode → SaveImage），POST `/prompt`、轮询 `/history`、经 `/view` 取图。
- `fetch_model.py` + `gen.py` — 备选路径：用 diffusers 直连本地权重。需要自建 venv 装 torch/diffusers，本轮因已有 ComfyUI 而未使用。

## SD-Turbo 参数要点

- `steps` 1–4（官方推荐 1–4），`cfg=1.0`（等价关闭 CFG，所以负向提示词基本不起作用）
- 采样器 `euler` + `scheduler=normal`，分辨率 512×512（原生训练分辨率）
- 每条提示词对应一个 `seed`，同 seed 同参数结果可复现
