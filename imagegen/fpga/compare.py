"""两张图对比：PSNR / SSIM / 并排图。

用法: python compare.py a.png b.png [--out out/cmp.png]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


def ssim(a, b, win=11, sigma=1.5):
    """灰度 SSIM（高斯窗），范围 [0,1]。"""
    from math import exp
    x = a.mean(axis=2)
    y = b.mean(axis=2)
    g = np.arange(win) - win // 2
    k = np.exp(-(g ** 2) / (2 * sigma ** 2))
    kernel = np.outer(k, k) / np.sum(k) ** 2

    def filt(img):
        pad = win // 2
        p = np.pad(img, pad, mode="reflect")
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(p, (win, win))
        return np.einsum("ijkl,kl->ij", w, kernel)

    mu_x, mu_y = filt(x), filt(y)
    sx = filt(x * x) - mu_x ** 2
    sy = filt(y * y) - mu_y ** 2
    sxy = filt(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(np.mean(((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) /
                         ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ia = np.asarray(Image.open(args.a).convert("RGB"), dtype=np.float32) / 255
    ib = np.asarray(Image.open(args.b).convert("RGB"), dtype=np.float32) / 255
    if ia.shape != ib.shape:
        print(f"尺寸不同 {ia.shape} vs {ib.shape}")
        return 1
    p = psnr(ia, ib)
    s = ssim(ia, ib)
    print(f"PSNR {p:.2f} dB   SSIM {s:.4f}   mean|Δ| {np.abs(ia-ib).mean():.4f}")
    if args.out:
        diff = np.abs(ia - ib)
        diff = diff / max(float(diff.max()), 1e-9)
        canvas = np.concatenate([ia, ib, diff], axis=1)
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((canvas * 255).round().astype(np.uint8)).save(out)
        print("对比图 ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
