"""生成对比图与指标表：同一提示词（cozy cabin, seed 42）的各个版本。

用法: python make_report.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(r"E:\FPGALLM\imagegen\out")

# (文件, 标签, 说明)
ITEMS = [
    ("diffusers_fp32_seed42.png", "A. FP32 reference", "diffusers fp32, 0.96s"),
    ("diffusers_fp_seed42.png", "B. FP16 baseline", "diffusers fp16"),
    ("seed42_s4_512px.png", "C. ComfyUI FP", "ComfyUI KSampler, 4 steps"),
    ("diffusers_int8_seed42.png", "D. INT8 weights+acts", "UNet int8, VAE fp, 20.5s"),
    ("diffusers_int8_tile_seed42.png", "E. INT8 + 32x32 tile", "UNet+VAE int8 tile, 85s"),
    ("img_fx_fx_eps4_seed42.png", "F. broken (eps mistake)", "wrong sampler -> flat"),
]


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


def ssim(a, b, win=11, sigma=1.5):
    x = a.mean(axis=2)
    y = b.mean(axis=2)
    g = np.arange(win) - win // 2
    k = np.exp(-(g ** 2) / (2 * sigma ** 2))
    ker = np.outer(k, k) / np.sum(k) ** 2
    pad = win // 2
    from numpy.lib.stride_tricks import sliding_window_view

    def filt(img):
        p = np.pad(img, pad, mode="reflect")
        return np.einsum("ijkl,kl->ij", sliding_window_view(p, (win, win)), ker)

    mu_x, mu_y = filt(x), filt(y)
    sx = filt(x * x) - mu_x ** 2
    sy = filt(y * y) - mu_y ** 2
    sxy = filt(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(np.mean(((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) /
                         ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))))


def main() -> int:
    ref = np.asarray(Image.open(OUT / "diffusers_fp32_seed42.png").convert("RGB"), np.float32) / 255
    rows = []
    for fn, label, note in ITEMS:
        p = OUT / fn
        if not p.exists():
            print(f"缺失: {fn}")
            continue
        img = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
        rows.append({
            "file": fn, "label": label, "note": note,
            "psnr": psnr(ref, img), "ssim": ssim(ref, img),
            "mad": float(np.abs(ref - img).mean()),
        })

    print(f"{'文件':<34}{'PSNR':>8}{'SSIM':>8}{'mean|Δ|':>10}   说明")
    for r in rows:
        print(f"{r['file']:<34}{r['psnr']:>8.2f}{r['ssim']:>8.4f}{r['mad']:>10.4f}   {r['note']}")

    # 拼接 2×3 对比图（带标签）
    cell = 512
    cols, rows_n = 3, 2
    sheet = Image.new("RGB", (cols * cell, rows_n * (cell + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (fn, label, note) in enumerate(ITEMS):
        p = OUT / fn
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB").resize((cell, cell))
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + 28)
        sheet.paste(im, (x, y + 28))
        draw.text((x + 6, y + 8), f"{label}", fill="black")
    out = OUT / "report_contact_sheet.png"
    sheet.save(out)
    print("拼图 ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
