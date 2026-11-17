"""把本地 sd_turbo.safetensors 的 UNet 转成 Q8_0 GGUF（只量化 2D 线性层）。

ComfyUI-GGUF 的加载器按 `model.diffusion_model.` 前缀过滤，并要求 proj 等线性层是 2D；
3×3 卷积的行长不是 32 的倍数，ggml 的 Q8_0 装不下，所以卷积与 norm 保持 F16
（这也是 ComfyUI-GGUF 对 SD 类 conv UNet 的既定做法）。

用法: python convert_gguf_q8.py
"""

from __future__ import annotations

import numpy as np

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
OUT = r"E:\ComfyUI\models\unet\sd-turbo-v2-1-Q8lin.gguf"
PREFIX = "model.diffusion_model."


def main() -> int:
    import gguf
    from safetensors import safe_open

    Q8 = gguf.GGMLQuantizationType.Q8_0
    F16 = gguf.GGMLQuantizationType.F16

    f = safe_open(CKPT, framework="numpy")
    keys = [k for k in f.keys() if k.startswith(PREFIX)]
    print(f"UNet 张量 {len(keys)} 个")

    w = gguf.GGUFWriter(OUT, "sd")
    w.add_uint32("general.file_type", 7)          # 与 llama.cpp 的 Q8_0 文件类型对应

    n_q = n_f = 0
    for k in sorted(keys):
        name = k[len(PREFIX):]
        t = f.get_tensor(k)
        if t.dtype == np.float16:
            t = t.astype(np.float32)
        if t.ndim == 2 and t.shape[1] % 32 == 0 and t.shape[1] >= 32:
            packed = gguf.quants.quantize(t.astype(np.float32), Q8)
            w.add_tensor(name, packed, raw_dtype=Q8)   # 形状由打包后的字节形状反推
            n_q += 1
        else:
            w.add_tensor(name, t.astype(np.float16))
            n_f += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"完成: Q8_0 {n_q} 个, F16 {n_f} 个 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
