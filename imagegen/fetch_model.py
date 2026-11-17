"""从 ModelScope 拉取 sd-turbo 的 diffusers 权重（只取 fp16 变体，约 2.5GB）。

用法:
    python fetch_model.py [--model-id AI-ModelScope/sd-turbo] [--dest models/sd-turbo]
"""

import argparse
import sys
from pathlib import Path

from modelscope import snapshot_download

# diffusers 加载只需要这些；两套权重里只要 fp16，省掉一半下载量
ALLOW = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.onnx_data"]

REQUIRED = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="AI-ModelScope/sd-turbo")
    ap.add_argument("--dest", default="models/sd-turbo")
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {args.model_id} -> {dest}", flush=True)

    try:
        snapshot_download(
            args.model_id,
            local_dir=str(dest),
            allow_file_pattern=ALLOW,
            ignore_file_pattern=IGNORE,
        )
    except TypeError:
        # 老版本 modelscope 没有 allow_file_pattern 参数
        print("[fetch] allow_file_pattern unsupported, downloading all", flush=True)
        snapshot_download(args.model_id, local_dir=str(dest))

    missing = [p for p in REQUIRED if not (dest / p).is_file()]
    if missing:
        print("[fetch] MISSING:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    total = sum((dest / p).stat().st_size for p in REQUIRED)
    print(f"[fetch] OK  {len(REQUIRED)} files, {total / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
