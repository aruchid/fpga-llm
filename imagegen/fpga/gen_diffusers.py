"""用 diffusers 的调度器 + 我们的 INT8 UNet 生成图像。

采样语义（v-pred / 步长 / sigma）全部交给 diffusers 的 scheduler，避免手写采样器出错；
我们只替换 UNet 的前向：--fp 用 diffusers 自带 UNet，默认用 INT8 版。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = r"E:\FPGALLM\imagegen\models\sd-turbo-diffusers"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp", action="store_true")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler

    out = Path(args.out).resolve()
    out.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[gen] pipeline {time.perf_counter()-t0:.1f}s  "
          f"scheduler={type(pipe.scheduler).__name__}  pred={pipe.scheduler.config.prediction_type}", flush=True)

    # 文本编码（diffusers 的 CLIP，与 ComfyUI 一致）
    ctx = pipe.encode_prompt(args.prompt, device="cuda", num_images_per_prompt=1,
                             do_classifier_free_guidance=False)[0]
    print(f"[gen] ctx {tuple(ctx.shape)}", flush=True)

    sched = pipe.scheduler
    sched.set_timesteps(args.steps)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = (1, 4, 64, 64)
    latents = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float16)
    latents = latents * sched.init_noise_sigma

    unet_q8 = None
    if not args.fp:
        import sd_unet_q8
        unet_q8 = sd_unet_q8.W(MODEL_MODEL_PATH())
        unet_q8 = _patch_weights(unet_q8)

    for i, t in enumerate(sched.timesteps):
        ts = time.perf_counter()
        inp = sched.scale_model_input(latents, t)
        if args.fp:
            with torch.no_grad():
                pred = pipe.unet(inp, t, encoder_hidden_states=ctx).sample
        else:
            pred = sd_unet_q8.forward(unet_q8, inp[0].permute(1, 2, 0).contiguous(),
                                      float(t), ctx[0])
            pred = pred.permute(2, 0, 1)[None].to(torch.float16)
        latents = sched.step(pred, t, latents, return_dict=False)[0]
        print(f"  step {i} t={float(t):7.2f} {time.perf_counter()-ts:5.1f}s", flush=True)

    with torch.no_grad():
        img = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(torch.float16)).sample
    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    path = out / (f"diff_{'fp' if args.fp else 'int8'}{args.tag}_seed{args.seed}.png")
    Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
    print("[gen] ->", path, f"{time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
