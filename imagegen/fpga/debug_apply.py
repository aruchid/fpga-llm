"""对比 ComfyUI 的 apply_model（采样器实际调用路径）与我们 UNet 的输出。"""

import sys

import numpy as np
import torch

sys.path.insert(0, r"E:\ComfyUI")
sys.path.insert(0, r"E:\FPGALLM\imagegen\fpga")

import sd_clip  # noqa: E402
import sd_unet  # noqa: E402

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    import comfy.sd
    res = comfy.sd.load_checkpoint_guess_config(CKPT, output_vae=False, output_clip=True)
    base = res[0].model          # ModelPatcher 里包着 BaseModel
    clip = res[1]

    noise = np.load(r"E:\FPGALLM\imagegen\out\noise_seed42.npy")[0]
    s0 = 14.614640235900879
    x0 = noise * np.float32(np.sqrt(1 + s0 ** 2))

    tok = clip.tokenize(PROMPT)
    cond_t, _ = clip.encode_from_tokens(tok, return_pooled=True)
    cond = {"c_crossattn": [cond_t], "c_concat": None}

    x = torch.from_numpy(x0[None])
    sigma = torch.tensor([s0])
    with torch.no_grad():
        d = base.apply_model(x, sigma, cond)
    d = d.float().cpu().numpy()
    eps_c = (x0 - d[0]) / s0
    print(f"apply_model 输出 std {d.std():.3f}（若是 x0 则应 ~1-4）")
    print(f"sigma 变换后 timestep = {base.model_sampling.timestep(sigma)}")

    ctx = sd_clip.text_encoder_float(sd_clip.ClipWeights(CKPT), sd_clip.CLIPTokenizer().encode(PROMPT))
    wt = sd_unet.UNetWeights(CKPT, cache=False)
    eps_o = sd_unet.forward(wt, np.transpose(x0, (1, 2, 0)), 999.0, ctx)
    print(f"eps cos = {cos(eps_c, eps_o):.5f}   comfy std {eps_c.std():.3f}  ours {eps_o.std():.3f}")

    # 也直接调 diffusion_model，看是否与 apply_model 一致
    dm = base.diffusion_model
    with torch.no_grad():
        raw = dm(torch.from_numpy(x0[None]).half(), torch.tensor([999.0]),
                 context=torch.from_numpy(ctx)[None].half())
    raw = raw.float().cpu().numpy()
    raw = np.transpose(raw[0], (1, 2, 0)) if raw.shape[1] == 4 else raw[0]
    print(f"直接调 diffusion_model 与 apply_model eps cos = {cos((x0 - raw) / s0, eps_c):.5f}")
    print(f"直接调 与 我们 eps cos = {cos(raw, eps_o):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
