"""检查 GGUF 载入后的 key 与模型类型识别。"""

import importlib.util
import sys

sys.path.insert(0, r"E:\ComfyUI")

PKG = r"E:\ComfyUI\custom_nodes\ComfyUI-GGUF"
spec = importlib.util.spec_from_file_location("ComfyUI_GGUF", PKG + r"\__init__.py",
                                              submodule_search_locations=[PKG])
mod = importlib.util.module_from_spec(spec)
sys.modules["ComfyUI_GGUF"] = mod
spec.loader.exec_module(mod)

from ComfyUI_GGUF import loader  # noqa: E402

import comfy.model_detection as md  # noqa: E402

sd, extra = loader.gguf_sd_loader(r"E:\ComfyUI\models\unet\sd-turbo-v2-1-Q8_0.gguf")
print("tensors", len(sd))
for k in list(sd.keys())[:6]:
    print(" ", k, type(sd[k]).__name__, getattr(sd[k], "shape", None))

try:
    cfg = md.detect_unet_config(sd, "")
    print("detected:", {k: cfg.get(k) for k in ("in_channels", "model_channels", "context_dim", "adm_in_channels")})
except Exception as e:
    print("detect fail:", type(e).__name__, e)
