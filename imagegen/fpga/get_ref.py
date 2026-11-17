"""从 ComfyUI 取 VAE 参考数据：同一张图的 latent + float 解码结果。

产出:
    out/vae_ref_float.png    ComfyUI 的 float VAE 解码（对照图）
    out/vae_latent.npy       VAEEncode 后的 latent（已按 0.18215 缩放）

用法:
    python get_ref.py [--image out/seed42_s4_512px.png]
"""

import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8188"
COMFY = Path(r"E:\ComfyUI")


def http_json(path, payload=None, timeout=30):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="out/seed42_s4_512px.png")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    src = Path(args.image).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ComfyUI 只能读 input 目录里的图
    dst = COMFY / "input" / "vae_probe.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "vae_probe.png"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_turbo.safetensors"}},
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["2", 2]}},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["2", 2]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "vae_ref"}},
        "6": {"class_type": "SaveLatent", "inputs": {"samples": ["3", 0], "filename_prefix": "vae_latent"}},
    }
    t0 = time.perf_counter()
    resp = http_json("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    if resp.get("node_errors"):
        print("node_errors:", json.dumps(resp["node_errors"], ensure_ascii=False))
        return 2
    pid = resp["prompt_id"]

    entry = None
    for _ in range(600):
        hist = http_json(f"/history/{pid}")
        if pid in hist and hist[pid].get("outputs"):
            entry = hist[pid]
            break
        time.sleep(0.5)
    if entry is None:
        print("timeout waiting for ComfyUI")
        return 1
    print(f"[ref] comfy wall {time.perf_counter() - t0:.1f}s")

    for node_out in entry["outputs"].values():
        for img in node_out.get("images", []):
            if img["type"] == "output" and img["filename"].endswith(".png"):
                q = urllib.parse.urlencode({"filename": img["filename"],
                                            "subfolder": img.get("subfolder", ""), "type": "output"})
                with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=60) as r:
                    (out_dir / "vae_ref_float.png").write_bytes(r.read())
                print("[ref] image ->", out_dir / "vae_ref_float.png")

    import numpy as np
    lat_dir = COMFY / "output"
    cands = sorted(lat_dir.glob("vae_latent*.latent"), key=lambda p: p.stat().st_mtime)
    if not cands:
        print("no .latent produced")
        return 1
    latest = cands[-1]
    with open(latest, "rb") as f:
        head = f.read(9)
    if head[:1] == b"\x00" or head[8:9] == b"{":          # safetensors
        from safetensors.numpy import load_file
        obj = load_file(str(latest))
        key = next((k for k in ("latent_tensor", "samples") if k in obj),
                   max(obj, key=lambda k: obj[k].size))
        z = obj[key]
    else:                                                 # 老格式 torch pickle
        import torch
        obj = torch.load(latest, map_location="cpu", weights_only=False)
        z = obj["samples"] if isinstance(obj, dict) else obj
        z = z.float().numpy()
    z = np.asarray(z, dtype=np.float32)
    np.save(out_dir / "vae_latent.npy", z)
    print(f"[ref] latent {tuple(z.shape)} <- {latest.name}")
    print(f"[ref] latent range [{z.min():.3f}, {z.max():.3f}] std={z.std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
