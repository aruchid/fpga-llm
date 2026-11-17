"""用 ComfyUI 自己的 CLIP 导出条件向量，判断差异来自 CLIP 还是 UNet。"""

import sys

import numpy as np
import torch

sys.path.insert(0, r"E:\ComfyUI")

PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"
CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"


def main() -> int:
    import comfy.sd
    import comfy.utils

    out = {}
    res = comfy.sd.load_checkpoint_guess_config(
        CKPT, output_vae=True, output_clip=True, embedding_directory=None)
    model, clip, vae = res[0], res[1], res[2]
    for tag, text in (("full", PROMPT), ("empty", "")):
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens(tokens, return_pooled=True)
        c = cond[0] if isinstance(cond, (tuple, list)) else cond
        arr = c.cpu().numpy().astype(np.float32)
        np.save(rf"E:\FPGALLM\imagegen\out\ctx_comfy_{tag}.npy", arr)
        print(f"{tag}: shape {arr.shape} std {arr.std():.4f}")
        out[tag] = arr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
