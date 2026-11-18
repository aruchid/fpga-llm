"""把 diffusers 的 SD UNet 换成 INT8 版本并出图对比。

- 权重：per-channel int8（每个输出通道一个 scale）
- 激活：per-row（每个像素/token）int8
- 矩阵乘：int8 × int8 → int32 累加（torch._int_mm），与 32 路矩阵单元同运算
- 其余算子（GroupNorm / SiLU / softmax）保持原样

用法: python quant_unet.py [--act8]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from q8 import mm8, quant_rows as q_rows, quant_weight as q_weight

MAC_COUNT = [0]
MAC_BY = {"conv": 0, "linear": 0}
TILE = [None]        # 设置后，矩阵乘改走 32×32 tile 引擎
PROFILE = {"on": False, "pad": 0.0, "im2col": 0.0, "quant": 0.0, "gemm": 0.0, "dequant": 0.0}

MODEL = r"E:\FPGALLM\imagegen\models\sd-turbo-diffusers"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


class QConv2d(torch.nn.Module):
    """INT8 卷积：im2col → 逐像素量化 → int8 GEMM → 反量化 + bias。"""

    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        # im2col 的列顺序是 (kh, kw, I)，权重必须按同样顺序展平
        w = conv.weight.detach().float().permute(0, 2, 3, 1).reshape(conv.out_channels, -1)
        self.q, self.s = q_weight(w)
        self.b = conv.bias.detach().float() if conv.bias is not None else None
        self.kh, self.kw = conv.kernel_size
        self.stride = conv.stride
        self.pad = conv.padding
        self.groups = conv.groups
        assert self.groups == 1, "只支持 groups=1"

    def forward(self, x, chunk: int = 8192):
        """按输出像素分块：512² 的 im2col 一次要 1.2 GB，分块后每块仅几十 MB。"""
        x = x.float()
        n, c, h, w = x.shape
        out = []
        for b in range(n):
            t0 = time.perf_counter()
            xh = x[b].permute(1, 2, 0).contiguous()          # NCHW → NHWC
            if self.pad[0] or self.pad[1]:
                ph, pw = self.pad[0], self.pad[1]
                # 只把边框清零，避免 zeros 对整个缓冲区做 memset（512² 每层 135 MB）
                xp = torch.empty((h + 2 * ph, w + 2 * pw, c), dtype=xh.dtype, device=xh.device)
                xp[ph:ph + h, pw:pw + w, :] = xh
                if ph:
                    xp[:ph, :, :] = 0
                    xp[ph + h:, :, :] = 0
                if pw:
                    xp[:, :pw, :] = 0
                    xp[:, pw + w:, :] = 0
            else:
                xp = xh
            ho = (h + 2 * self.pad[0] - self.kh) // self.stride[0] + 1
            wo = (w + 2 * self.pad[1] - self.kw) // self.stride[1] + 1
            B = ho * wo
            if PROFILE["on"]:
                torch.cuda.synchronize()
                PROFILE["pad"] += time.perf_counter() - t0

            y_out = torch.empty((B, self.q.shape[0]), dtype=torch.float32, device=x.device)
            for s0 in range(0, B, chunk):
                s1 = min(s0 + chunk, B)
                idx = torch.arange(s0, s1, device=x.device)
                rr = (idx // wo) * self.stride[0]
                cc = (idx % wo) * self.stride[1]

                t0 = time.perf_counter()
                cols = torch.empty((s1 - s0, self.kh * self.kw * c), dtype=torch.float32, device=x.device)
                t = 0
                for i in range(self.kh):
                    for j in range(self.kw):
                        cols[:, t * c:(t + 1) * c] = xp[rr + i, cc + j, :]
                        t += 1
                if PROFILE["on"]:
                    torch.cuda.synchronize()
                    PROFILE["im2col"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                aq, as_ = q_rows(cols)
                if PROFILE["on"]:
                    torch.cuda.synchronize()
                    PROFILE["quant"] += time.perf_counter() - t0

                MAC_COUNT[0] += self.q.shape[0] * self.q.shape[1] * (s1 - s0)
                MAC_BY["conv"] += self.q.shape[0] * self.q.shape[1] * (s1 - s0)

                t0 = time.perf_counter()
                acc = (TILE[0].gemm_int8(self.q, aq.T) if TILE[0] is not None
                       else mm8(self.q, aq.T))                   # (O, chunk) int32
                if PROFILE["on"]:
                    torch.cuda.synchronize()
                    PROFILE["gemm"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                y = acc.float() * (self.s[:, None] * as_[None, :])
                if self.b is not None:
                    y = y + self.b[:, None]
                y_out[s0:s1] = y.T
                if PROFILE["on"]:
                    torch.cuda.synchronize()
                    PROFILE["dequant"] += time.perf_counter() - t0
            out.append(y_out.reshape(ho, wo, -1).permute(2, 0, 1))
        return torch.stack(out, dim=0).to(x.dtype)

class QLinear(torch.nn.Linear):
    """INT8 线性层（原地替换，保留原 module 的接口）。"""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__(lin.in_features, lin.out_features, bias=lin.bias is not None)
        q, s = q_weight(lin.weight.detach().float())
        self.register_buffer("wq", q)
        self.register_buffer("ws", s)
        if lin.bias is not None:
            self.bias = torch.nn.Parameter(lin.bias.detach().float().clone())
        self.weight = None

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        aq, as_ = q_rows(x2)
        MAC_COUNT[0] += self.wq.shape[0] * self.wq.shape[1] * x2.shape[0]
        MAC_BY["linear"] += self.wq.shape[0] * self.wq.shape[1] * x2.shape[0]
        acc = (TILE[0].gemm_int8(self.wq, aq.T) if TILE[0] is not None
               else mm8(self.wq, aq.T))                          # (O, n) int32
        y = acc.float() * (self.ws[:, None] * as_[None, :])
        if self.bias is not None:
            y = y + self.bias[:, None]
        return y.T.reshape(*shape[:-1], -1).to(x.dtype)


def quantize_unet(unet):
    convs = lins = 0
    for name, mod in list(unet.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Conv2d) and child.groups == 1:
                setattr(mod, child_name, QConv2d(child))
                convs += 1
            elif isinstance(child, torch.nn.Linear):
                setattr(mod, child_name, QLinear(child))
                lins += 1
    print(f"[quant] 替换卷积 {convs} 个、线性层 {lins} 个")
    return convs, lins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    ap.add_argument("--tile", action="store_true", help="矩阵乘改走 32x32 tile 引擎")
    ap.add_argument("--vae8", action="store_true", help="VAE 解码器也量化进 32x32 矩阵")
    ap.add_argument("--strict-tiles", action="store_true", help="逐 32 元素 K 块累加（慢，但显式复现 RTL 块累加）")
    args = ap.parse_args()

    from diffusers import AutoPipelineForText2Image

    if args.tile:
        import mvm_tiled
        TILE[0] = mvm_tiled.MVM32Tiled(strict=args.strict_tiles)
        print("[tile] 启用严格 32×32 tile 引擎", flush=True)

    out = Path(args.out).resolve()
    # 整条管线统一 fp32：量化版和参考版同精度，避免 fp16/fp32 混用
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        local_files_only=True).to("cuda")
    pipe.to(torch.float32)          # 统一 fp32（权重已在本地，无需再找 .bin）
    pipe.set_progress_bar_config(disable=True)

    def gen():
        g = torch.Generator("cuda").manual_seed(args.seed)
        return pipe(PROMPT, num_inference_steps=args.steps, guidance_scale=0.0, generator=g).images[0]

    t0 = time.perf_counter()
    fp_img = gen()
    fp_dt = time.perf_counter() - t0
    torch.cuda.empty_cache()        # fp32 的权重/激活会留在缓存分配器里，挤占量化后的显存
    fp_path = out / f"diffusers_fp32_seed{args.seed}.png"
    fp_img.save(fp_path)
    print(f"[fp32] {fp_dt:.2f}s -> {fp_path}", flush=True)

    t0 = time.perf_counter()
    quantize_unet(pipe.unet)
    if args.vae8:
        # VAE 解码器同样量化：整条链（UNet + VAE）都走 32×32 矩阵
        quantize_unet(pipe.vae.decoder)
    torch.cuda.empty_cache()
    print(f"[quant] {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    q_img = gen()
    q_dt = time.perf_counter() - t0
    q_path = out / (f"diffusers_int8{'_tile' if TILE[0] is not None else ''}_seed{args.seed}.png")
    q_img.save(q_path)
    print(f"[int8] {q_dt:.2f}s -> {q_path}", flush=True)

    a = np.asarray(fp_img.convert("RGB"), np.float32) / 255
    b = np.asarray(q_img.convert("RGB"), np.float32) / 255
    mse = float(np.mean((a - b) ** 2))
    print(f"[int8] MACs {MAC_COUNT[0]/1e9:.1f} GMAC（UNet 4 步 + VAE）"
          f"  卷积 {MAC_BY['conv']/1e9:.1f} / 线性 {MAC_BY['linear']/1e9:.1f}")
    if TILE[0] is not None:
        st = TILE[0].stats
        print(f"[tile] tiles {st.tiles/1e6:.1f}M  cycles {st.cycles/1e9:.2f}G  "
              f"等效硬件 {TILE[0].hw_seconds():.0f}s @{TILE[0].fmax_mhz:g}MHz")
    print(f"INT8 vs FP32: PSNR {10*np.log10(1/mse) if mse>0 else 99:.2f} dB  "
          f"mean|Δ| {np.abs(a-b).mean():.4f}  速度 {fp_dt:.2f}s -> {q_dt:.2f}s")
    canvas = np.concatenate([a, b], axis=1)
    Image.fromarray((canvas * 255).round().astype(np.uint8)).save(out / "diffusers_fp_vs_int8.png")
    print("对比图 ->", out / "diffusers_fp_vs_int8.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())