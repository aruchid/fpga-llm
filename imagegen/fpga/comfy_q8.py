"""在 ComfyUI 里跑 Q8_0 GGUF 的 UNet，与 FP 版本同种子对比。"""

import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8188"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"
GGUF = "sd-turbo-v2-1-Q8_0.gguf"


def http_json(path, payload=None, timeout=120):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def run(use_gguf: bool, tag: str, seed=42, steps=4):
    clip_src = ["1", 1] if use_gguf else ["1", 1]
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_turbo.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": clip_src}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": tag}},
    }
    model_src = ["1", 0]
    if use_gguf:
        wf["8"] = {"class_type": "UnetLoaderGGUFSD21", "inputs": {"unet_name": GGUF}}
        model_src = ["8", 0]
    wf["5"] = {"class_type": "KSampler",
               "inputs": {"model": model_src, "seed": seed, "steps": steps, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "normal",
                          "positive": ["2", 0], "negative": ["2", 0],
                          "latent_image": ["4", 0], "denoise": 1.0}}
    resp = http_json("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    if resp.get("node_errors"):
        print("node_errors:", json.dumps(resp["node_errors"], ensure_ascii=False)[:900])
        return None
    pid = resp["prompt_id"]
    for _ in range(1200):
        hist = http_json(f"/history/{pid}")
        if pid in hist:
            e = hist[pid]
            if e.get("status", {}).get("status_str") == "error":
                print("执行错误:", json.dumps(e["status"], ensure_ascii=False)[:900])
                return None
            if e.get("outputs"):
                for node_out in e["outputs"].values():
                    for img in node_out.get("images", []):
                        q = urllib.parse.urlencode({"filename": img["filename"],
                                                    "subfolder": img.get("subfolder", ""),
                                                    "type": img.get("type", "output")})
                        with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=60) as r:
                            img = np.asarray(Image.open(__import__("io").BytesIO(r.read())).convert("RGB"))
                        return img
        time.sleep(0.3)
    print("timeout")
    return None


def main() -> int:
    out = Path(r"E:\FPGALLM\imagegen\out")
    t0 = time.perf_counter()
    a = run(False, "fp_check")
    print(f"FP  {time.perf_counter()-t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    b = run(True, "gguf_q8")
    print(f"GGUF {time.perf_counter()-t0:.1f}s", flush=True)
    if a is None or b is None:
        return 1
    Image.fromarray(a).save(out / "comfy_fp.png")
    Image.fromarray(b).save(out / "comfy_gguf_q8.png")
    af = a.astype(np.float32) / 255
    bf = b.astype(np.float32) / 255
    mse = float(np.mean((af - bf) ** 2))
    print(f"PSNR {10*np.log10(1/mse) if mse>0 else 99:.2f} dB")
    canvas = np.concatenate([a, b], axis=1)
    Image.fromarray(canvas).save(out / "comfy_fp_vs_q8.png")
    print("->", out / "comfy_fp_vs_q8.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
