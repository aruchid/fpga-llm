"""读取 llama2.c 格式的 .bin 权重，或生成随机权重。

所有矩阵都按 [输出, 输入] 存储，与 run.c 的 matmul(xout, x, w, n, d) 一致：
xout[i] = sum_j w[i, j] * x[j]。
"""
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Weights:
    token_embedding: np.ndarray  # (vocab, dim)
    rms_att: np.ndarray          # (L, dim)
    wq: np.ndarray               # (L, dim, dim)
    wk: np.ndarray               # (L, kv_dim, dim)
    wv: np.ndarray               # (L, kv_dim, dim)
    wo: np.ndarray               # (L, dim, dim)
    rms_ffn: np.ndarray          # (L, dim)
    w1: np.ndarray               # (L, hidden, dim)
    w2: np.ndarray               # (L, dim, hidden)
    w3: np.ndarray               # (L, hidden, dim)
    rms_final: np.ndarray        # (dim,)
    wcls: np.ndarray             # (vocab, dim)
    shared_classifier: bool


def load_llama2c(path: str):
    """读取 llama2.c 导出的 fp32 checkpoint（version 0 格式）。"""
    header = np.fromfile(path, dtype="<i4", count=7)
    dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len = (int(v) for v in header)
    shared = vocab > 0
    cfg = Config(dim, hidden, n_layers, n_heads, n_kv_heads, abs(vocab), seq_len)
    data = np.fromfile(path, dtype="<f4", offset=7 * 4)

    pos = 0

    def take(*shape):
        nonlocal pos
        n = int(np.prod(shape))
        if pos + n > data.size:
            raise ValueError(f"权重文件过短：需要 {pos + n} 个 float，实际 {data.size}")
        arr = data[pos:pos + n].reshape(shape)
        pos += n
        return arr

    L, d, kv, h, V = n_layers, dim, cfg.kv_dim, hidden, cfg.vocab_size
    w = dict(
        token_embedding=take(V, d),
        rms_att=take(L, d),
        wq=take(L, d, d),
        wk=take(L, kv, d),
        wv=take(L, kv, d),
        wo=take(L, d, d),
        rms_ffn=take(L, d),
        w1=take(L, h, d),
        w2=take(L, d, h),
        w3=take(L, h, d),
        rms_final=take(d),
    )
    # 跳过旧格式里遗留的 freq_cis_real / freq_cis_imag
    pos += seq_len * cfg.head_size // 2 * 2
    w["wcls"] = w["token_embedding"] if shared else take(V, d)
    return cfg, Weights(shared_classifier=shared, **w)


def save_llama2c(path: str, cfg: Config, w: Weights):
    """按 llama2.c 格式写出（测试读取器用）。"""
    vocab = cfg.vocab_size if w.shared_classifier else -cfg.vocab_size
    header = np.array([cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads,
                       cfg.n_kv_heads, vocab, cfg.seq_len], dtype="<i4")
    parts = [w.token_embedding, w.rms_att, w.wq, w.wk, w.wv, w.wo, w.rms_ffn,
             w.w1, w.w2, w.w3, w.rms_final,
             np.zeros(cfg.seq_len * cfg.head_size // 2 * 2, dtype=np.float32)]
    if not w.shared_classifier:
        parts.append(w.wcls)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        for p in parts:
            f.write(np.ascontiguousarray(p, dtype="<f4").tobytes())


def random_weights(cfg: Config, seed: int = 0, shared: bool = True) -> Weights:
    rng = np.random.default_rng(seed)
    L, d, kv, h, V = cfg.n_layers, cfg.dim, cfg.kv_dim, cfg.hidden_dim, cfg.vocab_size

    def mat(*shape, fan_in):
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

    def gain(*shape):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)

    emb = (rng.standard_normal((V, d)) * 0.5).astype(np.float32)
    return Weights(
        token_embedding=emb,
        rms_att=gain(L, d),
        wq=mat(L, d, d, fan_in=d),
        wk=mat(L, kv, d, fan_in=d),
        wv=mat(L, kv, d, fan_in=d),
        wo=mat(L, d, d, fan_in=d),
        rms_ffn=gain(L, d),
        w1=mat(L, h, d, fan_in=d),
        w2=mat(L, d, h, fan_in=h),
        w3=mat(L, h, d, fan_in=d),
        rms_final=gain(d),
        wcls=emb if shared else mat(V, d, fan_in=d),
        shared_classifier=shared,
    )
