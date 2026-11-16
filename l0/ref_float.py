"""浮点参考模型：逐步复刻 llama2.c run.c 的 forward()，float32。"""
import numpy as np

from .config import Config
from .weights import Weights


def rmsnorm(x, g, eps=1e-5):
    ss = np.mean(x * x, dtype=np.float32) + np.float32(eps)
    return (g * (x / np.sqrt(ss))).astype(np.float32)


def softmax(x):
    e = np.exp(x - np.max(x))
    return (e / np.sum(e)).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def rope_cos_sin(pos: int, head_size: int):
    """返回一个头内每对 (2j, 2j+1) 的 cos/sin，与 run.c 的频率公式一致。"""
    head_dim = np.arange(0, head_size, 2, dtype=np.float32)
    freq = np.float32(1.0) / np.power(np.float32(10000.0), head_dim / np.float32(head_size))
    val = np.float32(pos) * freq
    return np.cos(val).astype(np.float32), np.sin(val).astype(np.float32)


def apply_rope(vec, cos, sin, head_size):
    v = vec.reshape(-1, head_size // 2, 2)
    v0, v1 = v[..., 0], v[..., 1]
    out = np.empty_like(v)
    out[..., 0] = v0 * cos - v1 * sin
    out[..., 1] = v0 * sin + v1 * cos
    return out.reshape(-1)


class FloatModel:
    def __init__(self, cfg: Config, w: Weights):
        self.cfg, self.w = cfg, w
        self.reset()

    def reset(self):
        c = self.cfg
        self.key_cache = np.zeros((c.n_layers, c.seq_len, c.kv_dim), np.float32)
        self.value_cache = np.zeros_like(self.key_cache)

    def forward(self, token: int, pos: int) -> np.ndarray:
        # numpy 2.x 在 macOS 上的 BLAS matmul 会误报浮点异常，这里屏蔽，改为检查结果是否有限
        with np.errstate(all="ignore"):
            logits = self._forward(token, pos)
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError(f"logits 出现非有限值（pos={pos}）")
        return logits

    def _forward(self, token: int, pos: int) -> np.ndarray:
        c, w = self.cfg, self.w
        hs = c.head_size
        x = w.token_embedding[token].astype(np.float32).copy()
        cos, sin = rope_cos_sin(pos, hs)

        for l in range(c.n_layers):
            # 注意力
            xb = rmsnorm(x, w.rms_att[l])
            q = w.wq[l] @ xb
            k = w.wk[l] @ xb
            v = w.wv[l] @ xb
            q = apply_rope(q, cos, sin, hs)
            k = apply_rope(k, cos, sin, hs)
            self.key_cache[l, pos] = k
            self.value_cache[l, pos] = v

            att_out = np.empty(c.dim, np.float32)
            for h in range(c.n_heads):
                kvh = h // c.kv_mul
                qh = q[h * hs:(h + 1) * hs]
                K = self.key_cache[l, :pos + 1, kvh * hs:(kvh + 1) * hs]
                V = self.value_cache[l, :pos + 1, kvh * hs:(kvh + 1) * hs]
                scores = (K @ qh) / np.float32(np.sqrt(hs))
                att_out[h * hs:(h + 1) * hs] = softmax(scores) @ V
            x = x + w.wo[l] @ att_out

            # 前馈（SwiGLU）
            xb = rmsnorm(x, w.rms_ffn[l])
            hb = silu(w.w1[l] @ xb) * (w.w3[l] @ xb)
            x = x + w.w2[l] @ hb

        x = rmsnorm(x, w.rms_final)
        return (w.wcls @ x).astype(np.float32)
