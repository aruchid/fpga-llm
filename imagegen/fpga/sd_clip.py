"""CLIP ViT-H 文本编码器（SD2.1 用的那套）+ CLIP BPE 分词器。

浮点版用于校验实现，定点版把每个线性层送进 32 路矩阵单元。
权重前缀：conditioner.embedders.0.model.*
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

import fx
import mvm32

CLIP_PREFIX = "conditioner.embedders.0.model"
TOKENIZER_DIR = Path(r"E:\ComfyUI\comfy\sd1_tokenizer")
SOT = 49406
EOT = 49407
PAD = 0
MAX_LEN = 77


# ---------------------------------------------------------------- 分词器
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class CLIPTokenizer:
    """OpenAI CLIP 的 byte-level BPE。词表/合并表来自 ComfyUI 的 sd1_tokenizer。"""

    def __init__(self, tok_dir: Path = TOKENIZER_DIR):
        self.byte_encoder = _bytes_to_unicode()
        with open(tok_dir / "vocab.json", "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(tok_dir / "merges.txt", "r", encoding="utf-8") as f:
            merges = f.read().strip().split("\n")[1:]      # 首行是版本注释
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.cache = {self._unk: self._unk}
        self._pat = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
            r"[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]+", re.IGNORECASE)

    @property
    def _unk(self):
        return self.encoder.get("<|endoftext|>", 0)

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            out = " ".join(word)
            self.cache[token] = out
            return out
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str, pad_with_eot: bool = False) -> list[int]:
        text = unicodedata.normalize("NFC", text).lower().strip()
        text = re.sub(r"\s+", " ", text).strip()
        ids = [SOT]
        for piece in self._pat.findall(text):
            sub = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(sub).split(" "))
        ids.append(EOT)
        if pad_with_eot:
            ids = ids + [EOT] * (MAX_LEN - len(ids))
        else:
            ids = ids + [PAD] * (MAX_LEN - len(ids))
        return ids[:MAX_LEN]


# ---------------------------------------------------------------- 权重
class ClipWeights:
    def __init__(self, path, prefix=CLIP_PREFIX):
        self.path = path
        self.prefix = prefix
        self._h = None
        self._c = {}

    def _open(self):
        if self._h is None:
            from safetensors import safe_open
            self._h = safe_open(self.path, framework="numpy")
        return self._h

    def raw(self, name):
        if name not in self._c:
            self._c[name] = self._open().get_tensor(f"{self.prefix}.{name}")
        return self._c[name]


# ---------------------------------------------------------------- 浮点实现
def _erf(x):
    """Abramowitz-Stegun 7.1.26，最大误差 ~1.5e-7。"""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return s * y


def gelu(x):
    return (0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))).astype(np.float32)


def layernorm_f32(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * g + b


def text_encoder_float(wt: ClipWeights, tokens, layer_idx=-2):
    """tokens: list[int] 长度 77 → context (77, 1024) float32。"""
    tok_emb = wt.raw("token_embedding.weight").astype(np.float32)
    pos_emb = wt.raw("positional_embedding").astype(np.float32)
    x = tok_emb[np.asarray(tokens)] + pos_emb

    n_head, head_dim = 16, 64
    causal = np.triu(np.full((MAX_LEN, MAX_LEN), -1e9, dtype=np.float32), k=1)
    for li in range(24):
        p = f"transformer.resblocks.{li}."
        h = layernorm_f32(x, wt.raw(p + "ln_1.weight"), wt.raw(p + "ln_1.bias"))
        qkv = h @ wt.raw(p + "attn.in_proj_weight").T + wt.raw(p + "attn.in_proj_bias")
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(MAX_LEN, n_head, head_dim).transpose(1, 0, 2)
        k = k.reshape(MAX_LEN, n_head, head_dim).transpose(1, 0, 2)
        v = v.reshape(MAX_LEN, n_head, head_dim).transpose(1, 0, 2)
        sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(np.float32(head_dim)) + causal
        sc -= sc.max(axis=-1, keepdims=True)
        p_att = np.exp(sc)
        p_att /= p_att.sum(axis=-1, keepdims=True)
        o = (p_att @ v).transpose(1, 0, 2).reshape(MAX_LEN, 1024)
        o = o @ wt.raw(p + "attn.out_proj.weight").T + wt.raw(p + "attn.out_proj.bias")
        x = x + o
        h = layernorm_f32(x, wt.raw(p + "ln_2.weight"), wt.raw(p + "ln_2.bias"))
        h = gelu(h @ wt.raw(p + "mlp.c_fc.weight").T + wt.raw(p + "mlp.c_fc.bias"))
        h = h @ wt.raw(p + "mlp.c_proj.weight").T + wt.raw(p + "mlp.c_proj.bias")
        x = x + h
        if li == (24 + layer_idx):
            x_inter = x.copy()
    x = x_inter if layer_idx != -1 else x
    return layernorm_f32(x, wt.raw("ln_final.weight"), wt.raw("ln_final.bias"))


# ---------------------------------------------------------------- 定点实现
def gelu_lut(n=1024, lo=-8.0, hi=8.0):
    xs = np.linspace(lo, hi, n)
    return xs, np.round(gelu(xs.astype(np.float32)).astype(np.float64) * fx.ONE).astype(np.int64)


_GELU_X, _GELU_Y = gelu_lut()


def gelu_fx(x):
    """Q16 → Q16，查表 + 线性插值（硬件里就是一块 ROM）。"""
    x = np.asarray(x, dtype=np.int64)
    step = (_GELU_X[1] - _GELU_X[0])
    pos = np.clip((fx.from_fx(x) - _GELU_X[0]) / step, 0, len(_GELU_X) - 1.0001)
    idx = pos.astype(np.int64)
    frac = np.round((pos - idx) * fx.ONE).astype(np.int64)
    return _GELU_Y[idx] + fx.rshift_round((_GELU_Y[np.minimum(idx + 1, len(_GELU_X) - 1)] - _GELU_Y[idx]) * frac, 16)


def _lin_fx(engine, w, b, x):
    """x: (n, K) Q16 → (n, O) Q16。权重 int8 + 每行 SW；激活按 32 分块量化。"""
    wq, sw = mvm32.quantize_weight_rows(w)
    xq, sb = mvm32.quantize_blocks(x)
    y = engine.gemm_blocks(wq, sw, np.ascontiguousarray(xq.T), sb.T).T
    return y + fx.to_fx(b.astype(np.float64))[None, :]


def _quant_rows(x):
    from sd_vae import quantize_shift_batch
    return quantize_shift_batch(x)


def text_encoder_fx(wt: ClipWeights, tokens, engine, layer_idx=-2):
    """定点版：所有线性层走 32 路引擎。返回 (77,1024) Q16 int64。"""
    tok_emb = fx.to_fx(wt.raw("token_embedding.weight").astype(np.float64))
    pos_emb = fx.to_fx(wt.raw("positional_embedding").astype(np.float64))
    x = tok_emb[np.asarray(tokens)] + pos_emb

    n_head, head_dim = 16, 64
    x_inter = None
    for li in range(24):
        p = f"transformer.resblocks.{li}."
        h = fx.layernorm_fx(x, fx.to_fx(wt.raw(p + "ln_1.weight").astype(np.float64)),
                            fx.to_fx(wt.raw(p + "ln_1.bias").astype(np.float64)))
        qkv = _lin_fx(engine, wt.raw(p + "attn.in_proj_weight"), wt.raw(p + "attn.in_proj_bias"), h)
        q, k, v = np.split(qkv, 3, axis=-1)
        # 把 1/√d 折进 q（与 VAE 注意力一致），否则 softmax 过尖
        q = fx.rshift_round(q * fx.to_fx(1.0 / np.sqrt(float(head_dim))), 16)
        o = np.empty((MAX_LEN, 1024), dtype=np.int64)
        for hd in range(n_head):
            sl = slice(hd * head_dim, (hd + 1) * head_dim)
            qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
            qq, qs = _quant_rows(qh)
            kq, ks = _quant_rows(kh)
            vq, vs = _quant_rows(np.ascontiguousarray(vh.T))
            dots = engine.dot_i8(kq, np.ascontiguousarray(qq.T))            # (77,77)
            sc = fx.rshift_round(dots.astype(np.int64), 32 - qs[None, :] - ks[:, None])
            sc = sc.T                                                       # (77,77) 行=query 列=key
            # 因果掩码：query 只能看 key <= 自己，上三角置成很负的值
            sc = np.where(np.triu(np.ones((MAX_LEN, MAX_LEN), dtype=bool), k=1), -(1 << 26), sc)
            pr = fx.softmax_fx(sc, axis=1)
            pq, ps = _quant_rows(pr)
            pv = engine.dot_i8(pq, np.ascontiguousarray(vq.T))              # (77,64)
            o[:, sl] = fx.rshift_round(pv.astype(np.int64), 16 - ps[:, None] - vs[None, :])
        o = _lin_fx(engine, wt.raw(p + "attn.out_proj.weight"), wt.raw(p + "attn.out_proj.bias"), o)
        x = fx.sat32(x + o)
        h = fx.layernorm_fx(x, fx.to_fx(wt.raw(p + "ln_2.weight").astype(np.float64)),
                            fx.to_fx(wt.raw(p + "ln_2.bias").astype(np.float64)))
        h = gelu_fx(_lin_fx(engine, wt.raw(p + "mlp.c_fc.weight"), wt.raw(p + "mlp.c_fc.bias"), h))
        h = _lin_fx(engine, wt.raw(p + "mlp.c_proj.weight"), wt.raw(p + "mlp.c_proj.bias"), h)
        x = fx.sat32(x + h)
        if li == (24 + layer_idx):
            x_inter = x.copy()
    if layer_idx != -1 and x_inter is not None:
        x = x_inter
    return fx.layernorm_fx(x, fx.to_fx(wt.raw("ln_final.weight").astype(np.float64)),
                           fx.to_fx(wt.raw("ln_final.bias").astype(np.float64)))
