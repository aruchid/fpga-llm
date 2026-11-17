"""逐层对拍 ComfyUI 的 UNet 与我们的 numpy 实现，定位结构性差异。"""

import sys

import numpy as np
import torch

sys.path.insert(0, r"E:\ComfyUI")
sys.path.insert(0, r"E:\FPGALLM\imagegen\fpga")

import fx  # noqa: E402
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
    model, clip = res[0], res[1]
    dm = model.model.diffusion_model

    noise = np.load(r"E:\FPGALLM\imagegen\out\noise_seed42.npy")[0]
    x0 = noise * np.float32(np.sqrt(1 + 14.614640235900879 ** 2))
    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode(PROMPT)
    ctx = sd_clip.text_encoder_float(sd_clip.ClipWeights(CKPT), ids)

    caps = {}

    def hook(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            t = t.detach().float()
            if t.ndim == 4:
                t = t.permute(0, 2, 3, 1)      # ComfyUI 内部是 NCHW，我们统一按 NHWC 比
            caps[name] = t.cpu().numpy()
        return f

    for i, blk in enumerate(dm.input_blocks):
        blk.register_forward_hook(hook(f"in{i}"))
    dm.middle_block.register_forward_hook(hook("mid"))
    for i, blk in enumerate(dm.output_blocks):
        blk.register_forward_hook(hook(f"out{i}"))

    with torch.no_grad():
        o_c = dm(torch.from_numpy(x0[None]).half(),
                 torch.tensor([999.0]), context=torch.from_numpy(ctx)[None].half())
    o_c = o_c.float().cpu().numpy()
    if o_c.ndim == 4 and o_c.shape[1] == 4:
        o_c = np.transpose(o_c, (0, 2, 3, 1))

    # ---- 我们的（复制 forward 逻辑并记录相同的检查点）
    wt = sd_unet.UNetWeights(CKPT, cache=False)
    temb = sd_unet.timestep_embedding(999.0)
    w0, b0 = wt.linear("time_embed.0")
    w2, b2 = wt.linear("time_embed.2")
    emb = sd_unet.linear(sd_unet.silu_f32(sd_unet.linear(temb, w0, b0)), w2, b2)
    hs = []
    hcur = sd_unet.conv_float(np.transpose(x0, (1, 2, 0)), *wt.conv("input_blocks.0.0"), pad=1)
    hs.append(hcur)
    chans = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    heads = lambda c: max(1, c // 64)
    ours = {"in0": hcur}
    for ib in range(1, 12):
        if ib in (3, 6, 9):
            hcur = sd_unet.conv_float(hcur, *wt.conv(f"input_blocks.{ib}.0.op"), pad=1, stride=2)
            hs.append(hcur)
            ours[f"in{ib}"] = hcur
            continue
        hcur = sd_unet.resblock(wt, hcur, emb, f"input_blocks.{ib}.0", hcur.shape[-1], chans[ib])
        if ib in (1, 2, 4, 5, 7, 8):
            hcur = sd_unet.spatial_transformer(wt, hcur, ctx, f"input_blocks.{ib}.1", chans[ib],
                                               n_heads=heads(chans[ib]))
        hs.append(hcur)
        ours[f"in{ib}"] = hcur
    hcur = sd_unet.resblock(wt, hcur, emb, "middle_block.0", 1280, 1280)
    hcur = sd_unet.spatial_transformer(wt, hcur, ctx, "middle_block.1", 1280, n_heads=heads(1280))
    hcur = sd_unet.resblock(wt, hcur, emb, "middle_block.2", 1280, 1280)
    ours["mid"] = hcur
    up = [1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
    for ob in range(12):
        skip = hs.pop()
        hcur = np.concatenate([hcur, skip], axis=-1)
        hcur = sd_unet.resblock(wt, hcur, emb, f"output_blocks.{ob}.0", hcur.shape[-1], up[ob])
        if ob >= 3:
            hcur = sd_unet.spatial_transformer(wt, hcur, ctx, f"output_blocks.{ob}.1", up[ob],
                                               n_heads=heads(up[ob]))
        if ob in (2, 5, 8):
            idx = "2" if ob > 2 else "1"
            hcur = sd_unet.conv_float(sd_unet.upsample(hcur), *wt.conv(f"output_blocks.{ob}.{idx}.conv"), pad=1)
        ours[f"out{ob}"] = hcur
    g, b = wt.raw("out.0.weight"), wt.raw("out.0.bias")
    fin = sd_unet.conv_float(sd_unet.silu_f32(sd_unet.gn_float(hcur, g, b)),
                             *wt.conv("out.2"), pad=1)

    print(f"最终输出 cos = {abs(cos(fin, o_c[0])):.5f}")
    print(f"{'检查点':<8} {'comfy shape':<22} {'ours shape':<22} cos")
    for k in ["in0"] + [f"in{i}" for i in range(1, 12)] + ["mid"] + [f"out{i}" for i in range(12)]:
        if k not in caps or k not in ours:
            continue
        c = caps[k]
        a = ours[k]
        c2 = c[0] if c.ndim == 4 else c
        print(f"{k:<8} {str(c2.shape):<22} {str(a.shape):<22} {cos(a, c2):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
