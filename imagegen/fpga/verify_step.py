"""单步对照：把同一个含噪 latent 喂给 ComfyUI 和我们的 UNet，比较一步后的结果。

这一步能同时验证：UNet 结构、时间步嵌入、eps/v-pred、Euler 公式。

用法: python verify_step.py [--mode float|fx] [--steps 4]
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

import sd_clip
import sd_unet

BASE = "http://127.0.0.1:8188"
COMFY = Path(r"E:\ComfyUI")
CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"
SIGMAS4 = [14.614640235900879, 2.918307065963745, 0.9323579668998718, 0.029167160391807556, 0.0]
TIMESTEPS4 = [999, 666, 333, 0]


def http_json(path, payload=None, timeout=60):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="float", choices=["float", "fx"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out).resolve()

    noise_path = out / f"noise_seed{args.seed}.npy"
    if not noise_path.exists():
        import torch
        g = torch.Generator().manual_seed(args.seed)
        np.save(noise_path, torch.randn((1, 4, 64, 64), generator=g).numpy().astype(np.float32))
    noise = np.load(noise_path)
    x0 = noise * np.float32(np.sqrt(1.0 + SIGMAS4[0] ** 2))
    print(f"[v] x0 std={x0.std():.3f}")

    # ---- 写 latent 文件给 ComfyUI
    lat_file = COMFY / "input" / "probe_x0.latent"
    save_file({"latent_tensor": x0, "latent_format_version_0": np.zeros((0,), dtype=np.float32)},
              str(lat_file))

    wf = {
        "1": {"class_type": "LoadLatent", "inputs": {"latent": "probe_x0.latent"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_turbo.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 1]}},
        "5": {"class_type": "KSamplerAdvanced",
              "inputs": {"model": ["2", 0], "add_noise": "disable", "noise_seed": 0,
                         "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal",
                         "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["1", 0],
                         "start_at_step": 0, "end_at_step": 1, "return_with_leftover_noise": "enable"}},
        "6": {"class_type": "SaveLatent", "inputs": {"samples": ["5", 0], "filename_prefix": "probe_x1"}},
    }
    resp = http_json("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    if resp.get("node_errors"):
        print("node_errors", json.dumps(resp["node_errors"], ensure_ascii=False)[:800])
        return 2
    pid = resp["prompt_id"]
    for _ in range(600):
        hist = http_json(f"/history/{pid}")
        if pid in hist and hist[pid].get("outputs"):
            break
        time.sleep(0.3)
    else:
        print("timeout")
        return 1

    cands = sorted((COMFY / "output").glob("probe_x1*.latent"), key=lambda p: p.stat().st_mtime)
    from safetensors.numpy import load_file
    obj = load_file(str(cands[-1]))
    x1_comfy = obj["latent_tensor"]
    print(f"[v] comfy x1 std={x1_comfy.std():.3f} range [{x1_comfy.min():.2f},{x1_comfy.max():.2f}]")
    # ComfyUI 刚加载完模型，先把它的显存/内存还回去，否则我们这边开 mmap 都失败
    try:
        http_json("/free", {"unload_models": True, "free_memory": True})
        time.sleep(3)
    except Exception as e:
        print("[v] free failed", e)

    # ---- 我们自己算一步
    cw = sd_clip.ClipWeights(CKPT)
    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode(PROMPT)
    uw = sd_unet.UNetWeights(CKPT, cache=False)
    ctx_f = sd_clip.text_encoder_float(cw, ids)
    t0 = time.perf_counter()
    if args.mode == "float":
        out_y = sd_unet.forward(uw, np.transpose(x0[0], (1, 2, 0)), TIMESTEPS4[0], ctx_f)
        x1 = x0 - (SIGMAS4[0] - SIGMAS4[1]) * np.transpose(out_y, (2, 0, 1))[None]
    else:
        import fx
        import mvm32
        import sd_unet_fx
        eng = mvm32.MVM32()
        ctx_q = sd_clip.text_encoder_fx(cw, ids, eng)
        out_y = sd_unet_fx.forward(uw, fx.to_fx(np.transpose(x0[0], (1, 2, 0)).astype(np.float64)),
                                   TIMESTEPS4[0], ctx_q, eng)
        x1 = x0 - (SIGMAS4[0] - SIGMAS4[1]) * np.transpose(fx.from_fx(out_y), (2, 0, 1))[None]
    print(f"[v] ours {args.mode} x1 std={x1.std():.3f} ({time.perf_counter()-t0:.1f}s)")

    a = x1.ravel().astype(np.float64)
    b = x1_comfy.ravel().astype(np.float64)
    print(f"[v] cos={float(a@b)/np.linalg.norm(a)/np.linalg.norm(b):.4f}  "
          f"rel={np.linalg.norm(a-b)/np.linalg.norm(b):.3f}")
    np.save(out / "x1_comfy.npy", x1_comfy)
    np.save(out / "x1_ours.npy", x1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
