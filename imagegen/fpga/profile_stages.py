"""分阶段计时：CLIP 编码 / UNet 4 步 / VAE 解码，分别测 fp32、INT8、INT8+tile。"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"E:\FPGALLM\imagegen")
sys.path.insert(0, str(ROOT / "fpga"))

MODEL = str(ROOT / "models" / "sd-turbo-diffusers")
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"
STEPS = 4


def gen_stages(pipe, tag):
    """跑一次完整生成，分别计时三个阶段。"""
    t0 = time.perf_counter()
    ctx = pipe.encode_prompt(PROMPT, device="cuda", num_images_per_prompt=1,
                             do_classifier_free_guidance=False)[0]
    torch.cuda.synchronize()
    t_clip = time.perf_counter() - t0

    g = torch.Generator("cuda").manual_seed(42)
    lat = torch.randn((1, 4, 64, 64), generator=g, device="cuda", dtype=torch.float16)
    sched = pipe.scheduler
    sched.set_timesteps(STEPS)
    lat = lat.to(pipe.unet.dtype) * sched.init_noise_sigma

    t0 = time.perf_counter()
    for t in sched.timesteps:
        inp = sched.scale_model_input(lat, t)
        with torch.no_grad():
            pred = pipe.unet(inp, t, encoder_hidden_states=ctx).sample
        lat = sched.step(pred, t, lat, return_dict=False)[0]
        torch.cuda.synchronize()
    t_unet = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        img = pipe.vae.decode((lat / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample
    torch.cuda.synchronize()
    t_vae = time.perf_counter() - t0

    total = t_clip + t_unet + t_vae
    print(f"{tag:<22} 总 {total:6.2f} s | CLIP {t_clip:5.2f} s | UNet×4 {t_unet:6.2f} s | VAE {t_vae:6.2f} s")
    return total, t_clip, t_unet, t_vae


def main() -> int:
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    print(f"{'配置':<22} {'阶段耗时'}")
    gen_stages(pipe, "FP16 基线")          # 预热 + 基线

    pipe.to(torch.float32)
    gen_stages(pipe, "FP32 基线")

    import quant_unet
    quant_unet.quantize_unet(pipe.unet)
    torch.cuda.empty_cache()
    gen_stages(pipe, "INT8 UNet（fp VAE）")

    quant_unet.quantize_unet(pipe.vae.decoder)
    torch.cuda.empty_cache()
    gen_stages(pipe, "INT8 UNet+VAE（torch）")

    import mvm_tiled
    quant_unet.TILE[0] = mvm_tiled.MVM32Tiled()
    torch.cuda.empty_cache()
    gen_stages(pipe, "INT8 + 32×32 tile")
    st = quant_unet.TILE[0].stats
    print(f"\ntile 引擎：{st.tiles/1e6:.1f}M tile，{st.cycles/1e9:.2f}G 周期，"
          f"等效 {quant_unet.TILE[0].hw_seconds():.0f} s @115MHz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
