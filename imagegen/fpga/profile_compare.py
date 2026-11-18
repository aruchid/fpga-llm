"""对比两条路径的耗时：diffusers 的 pipe(...) 全流程 vs 手工分阶段。"""

import sys
import time

import torch

sys.path.insert(0, r"E:\FPGALLM\imagegen\fpga")

MODEL = r"E:\FPGALLM\imagegen\models\sd-turbo-diffusers"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def main() -> int:
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.to(torch.float32)

    import quant_unet
    quant_unet.quantize_unet(pipe.unet)
    quant_unet.quantize_unet(pipe.vae.decoder)
    torch.cuda.empty_cache()

    g = lambda: torch.Generator("cuda").manual_seed(42)

    # 1) pipe(...) 全流程
    pipe(PROMPT, num_inference_steps=4, guidance_scale=0.0, generator=g())   # 预热
    t0 = time.perf_counter()
    pipe(PROMPT, num_inference_steps=4, guidance_scale=0.0, generator=g())
    t_pipe = time.perf_counter() - t0

    # 2) 手工分阶段
    t0 = time.perf_counter()
    ctx = pipe.encode_prompt(PROMPT, device="cuda", num_images_per_prompt=1,
                             do_classifier_free_guidance=False)[0]
    torch.cuda.synchronize()
    t_clip = time.perf_counter() - t0

    sched = pipe.scheduler
    sched.set_timesteps(4)
    lat = torch.randn((1, 4, 64, 64), generator=g(), device="cuda", dtype=torch.float32) * sched.init_noise_sigma
    t0 = time.perf_counter()
    for t in sched.timesteps:
        with torch.no_grad():
            pred = pipe.unet(sched.scale_model_input(lat, t), t, encoder_hidden_states=ctx).sample
        lat = sched.step(pred, t, lat, return_dict=False)[0]
        torch.cuda.synchronize()
    t_unet = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = pipe.vae.decode((lat / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample
    torch.cuda.synchronize()
    t_vae = time.perf_counter() - t0

    print(f"pipe(...) 全流程     {t_pipe:6.2f} s")
    print(f"手工分阶段            {t_clip + t_unet + t_vae:6.2f} s  (CLIP {t_clip:.2f} / UNet {t_unet:.2f} / VAE {t_vae:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
