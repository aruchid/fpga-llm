"""通过 ComfyUI HTTP API 跑 sd-turbo 文生图（复用本机已装好的权重，不下载任何东西）。

用法:
    python comfy_gen.py --prompt "a red fox in the snow" --steps 4 --seed 42

前置: ComfyUI 在 http://127.0.0.1:8188 运行，checkpoints 里有 sd_turbo.safetensors。
"""

import argparse
import json
import time
import urllib.parse
import uuid
from pathlib import Path

import urllib.request

BASE = "http://127.0.0.1:8188"


def http_json(path: str, payload: dict | None = None, timeout: int = 30):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def build_workflow(ckpt: str, prompt: str, negative: str, steps: int, cfg: float,
                   width: int, height: int, seed: int, sampler: str, scheduler: str):
    """标准 txt2img 图：加载 checkpoint -> 双路编码 -> 空 latent -> 采样 -> 解码 -> 保存。"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "fpga_llm_t2i"}},
    }


def wait_history(prompt_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = http_json(f"/history/{prompt_id}", timeout=15)
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(f"ComfyUI 执行失败: {json.dumps(status, ensure_ascii=False)[:800]}")
            if entry.get("outputs"):
                return entry
        time.sleep(0.5)
    raise TimeoutError(f"等待 {timeout:.0f}s 仍未出图")


def download_image(meta: dict, dest: Path) -> Path:
    q = urllib.parse.urlencode({
        "filename": meta["filename"],
        "subfolder": meta.get("subfolder", ""),
        "type": meta.get("type", "output"),
    })
    with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--ckpt", default="sd_turbo.safetensors")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.0, help="sd-turbo 用 1.0（等价关闭 CFG）")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="normal")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wf = build_workflow(args.ckpt, args.prompt, args.negative, args.steps, args.cfg,
                        args.size, args.size, args.seed, args.sampler, args.scheduler)

    t0 = time.perf_counter()
    resp = http_json("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    if resp.get("node_errors"):
        print("[comfy] node_errors:", json.dumps(resp["node_errors"], ensure_ascii=False))
        return 2
    prompt_id = resp["prompt_id"]
    entry = wait_history(prompt_id, args.timeout)
    wall = time.perf_counter() - t0

    images = []
    for node_out in entry["outputs"].values():
        for img in node_out.get("images", []):
            name = f"seed{args.seed}_s{args.steps}_{args.size}px.png"
            path = download_image(img, out_dir / name)
            images.append(path)

    print(json.dumps({
        "prompt": args.prompt,
        "ckpt": args.ckpt,
        "steps": args.steps,
        "size": args.size,
        "seed": args.seed,
        "wall_seconds": round(wall, 2),
        "images": [str(p) for p in images],
    }, ensure_ascii=False, indent=2))
    return 0 if images else 1


if __name__ == "__main__":
    raise SystemExit(main())
