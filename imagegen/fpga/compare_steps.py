"""逐步对拍：同一个 x0，ComfyUI 的 KSamplerAdvanced 跑 1..4 步，和我们的中间 latent 比。

用法: python compare_steps.py --ours out/latent_f1_s --n 4
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

BASE = "http://127.0.0.1:8188"
COMFY = Path(r"E:\ComfyUI")
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def http_json(path, payload=None, timeout=60):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def run_comfy(x0: np.ndarray, end_step: int, prompt: str = PROMPT, tag="cmp",
              start_step: int = 0) -> np.ndarray:
    """把 x0 喂进去，跑 start_step→end_step 步，返回结果 latent。"""
    lat = COMFY / "input" / f"{tag}_x0.latent"
    save_file({"latent_tensor": x0.astype(np.float32),
               "latent_format_version_0": np.zeros((0,), np.float32)}, str(lat))
    wf = {
        "1": {"class_type": "LoadLatent", "inputs": {"latent": f"{tag}_x0.latent"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_turbo.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 1]}},
        "4": {"class_type": "KSamplerAdvanced",
              "inputs": {"model": ["2", 0], "add_noise": "disable", "noise_seed": 0,
                         "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal",
                         "positive": ["3", 0], "negative": ["3", 0], "latent_image": ["1", 0],
                         "start_at_step": start_step, "end_at_step": end_step,
                         "return_with_leftover_noise": "enable"}},
        "5": {"class_type": "SaveLatent", "inputs": {"samples": ["4", 0], "filename_prefix": f"{tag}_x{end_step}"}},
    }
    resp = http_json("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    if resp.get("node_errors"):
        raise RuntimeError(json.dumps(resp["node_errors"], ensure_ascii=False)[:600])
    pid = resp["prompt_id"]
    for _ in range(1200):
        hist = http_json(f"/history/{pid}")
        if pid in hist and hist[pid].get("outputs"):
            break
        time.sleep(0.2)
    else:
        raise TimeoutError("comfy timeout")
    f = sorted((COMFY / "output").glob(f"{tag}_x{end_step}*.latent"), key=lambda p: p.stat().st_mtime)[-1]
    return load_file(str(f))["latent_tensor"]


def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="out/latent_f1_s")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    # 我们的第 0 步 latent 是 x1；为了对齐，先读 x0（噪声 * sqrt(1+σ²)）
    noise = np.load("out/noise_seed42.npy")[0]
    x0 = noise * np.float32(np.sqrt(1.0 + 14.614640235900879 ** 2))
    print(f"x0 std {x0.std():.3f}")
    ref = {0: run_comfy(x0[None], 0)}
    print(f"comfy x0   std {ref[0].std():.3f}")
    for k in range(1, args.n + 1):
        ref[k] = run_comfy(x0[None], k)
        ours = np.load(f"{args.ours}{k-1}.npy")
        print(f"step {k}: comfy std {ref[k].std():7.3f} | ours std {ours.std():7.3f} | cos {cos(ref[k], ours):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
