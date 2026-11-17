"""sd-turbo 文生图：最小可跑的推理脚本。

用法:
    python gen.py --prompt "a red fox in the snow" --steps 4 --seed 42

默认读 models/sd-turbo（先用 fetch_model.py 拉取），输出到 out/。
"""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

DEFAULT_PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def load_pipe(model_dir: Path, device: str):
    t0 = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        safety_checker=None,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sd-turbo")
    ap.add_argument("--prompt", action="append", help="可重复传入多组提示词")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=0.0, help="sd-turbo 用 0.0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("[gen] CUDA unavailable, falling back to CPU")
        device = "cpu"

    model_dir = Path(args.model).resolve()
    if not (model_dir / "model_index.json").is_file():
        print(f"[gen] model not found at {model_dir}; run: python fetch_model.py")
        return 2

    prompts = args.prompt or [DEFAULT_PROMPT]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe, load_s = load_pipe(model_dir, device)
    print(f"[gen] device={device} dtype={pipe.dtype} load={load_s:.1f}s")

    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        t0 = time.perf_counter()
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.size,
            width=args.size,
            generator=gen,
        ).images[0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        path = out_dir / f"seed{args.seed + i}_s{args.steps}_{args.size}.png"
        image.save(path)
        print(json.dumps({
            "prompt": prompt,
            "image": str(path),
            "steps": args.steps,
            "size": args.size,
            "seconds": round(dt, 2),
        }, ensure_ascii=False))

    if device == "cuda":
        print(f"[gen] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
