"""验证 t=666 的 eps：用 ComfyUI 的 x1 → x2 反推 eps，与我们的 UNet(x1, 666) 比。"""

import sys

import numpy as np

sys.path.insert(0, r"E:\FPGALLM\imagegen\fpga")

import sd_clip  # noqa: E402
import sd_unet  # noqa: E402
from compare_steps import run_comfy  # noqa: E402

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"
S = [14.614640235900879, 2.918307065963745, 0.9323579668998718, 0.029167160391807556, 0.0]
T = [999, 666, 333, 0]


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    noise = np.load(r"E:\FPGALLM\imagegen\out\noise_seed42.npy")[0]
    x0 = noise * np.float32(np.sqrt(1 + S[0] ** 2))
    ctx = sd_clip.text_encoder_float(sd_clip.ClipWeights(CKPT), sd_clip.CLIPTokenizer().encode(PROMPT))
    wt = sd_unet.UNetWeights(CKPT, cache=False)

    x = x0
    for i in range(4):
        x_next = run_comfy(x[None], i + 1, prompt=PROMPT, start_step=i)[0]
        eps_c = (x - x_next) / (S[i] - S[i + 1])
        eps_o = sd_unet.forward(wt, np.transpose(x, (1, 2, 0)), float(T[i]), ctx)
        print(f"step {i} t={T[i]:4d}: comfy eps std {eps_c.std():6.3f} | ours {eps_o.std():6.3f} | "
              f"cos {cos(eps_c, eps_o):.5f} | x_next std comfy {x_next.std():6.3f}", flush=True)
        x = x_next
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
