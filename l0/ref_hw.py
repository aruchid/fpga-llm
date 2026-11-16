"""硬件等价模型：全部整数运算，流程与 ref_float.FloatModel 一一对应。

KV Cache 以 int8 存储，每个位置、每个 KV 头一个移位量。
"""
from collections import namedtuple

import numpy as np

from . import fixedpoint as fp
from .config import Config
from .weights import Weights


class QLinear:
    def __init__(self, w):
        self.q, self.SW = fp.quantize_weight_rows(w)

    def __call__(self, xq, s, tag):
        return fp.int_matmul(self.q, self.SW, xq, s, tag)


# 一次矩阵乘请求；Op("matmul", [...]) 里的请求共享同一输入 (xq, s)，可合并成一次任务。
MatmulReq = namedtuple("MatmulReq", "lin xq s tag")
# 一个运算：kind ∈ rmsnorm / quant / matmul / rope / attn / silu_gate / resadd
Op = namedtuple("Op", "kind args")


def exec_op(op):
    """L0 执行者：用 fixedpoint 的定点函数完成一个运算。"""
    k, a = op
    if k == "matmul":
        return [r.lin(r.xq, r.s, r.tag) for r in a]
    if k == "rmsnorm":
        return fp.rmsnorm_fx(a[0], a[1], "rmsnorm")
    if k == "quant":
        return fp.quantize_shift(a[0])
    if k == "rope":
        return fp.rope_fx(*a)
    if k == "attn":
        return fp.attention_head_fx(*a)
    if k == "silu_gate":
        return fp.silu_gate_fx(*a)
    if k == "resadd":
        return fp.resadd_fx(*a)
    raise ValueError(k)


class HWModel:
    def __init__(self, cfg: Config, w: Weights):
        self.cfg = cfg
        L = cfg.n_layers
        self.emb = QLinear(w.token_embedding)
        self.wcls = self.emb if w.shared_classifier else QLinear(w.wcls)
        self.wq = [QLinear(w.wq[l]) for l in range(L)]
        self.wk = [QLinear(w.wk[l]) for l in range(L)]
        self.wv = [QLinear(w.wv[l]) for l in range(L)]
        self.wo = [QLinear(w.wo[l]) for l in range(L)]
        self.w1 = [QLinear(w.w1[l]) for l in range(L)]
        self.w2 = [QLinear(w.w2[l]) for l in range(L)]
        self.w3 = [QLinear(w.w3[l]) for l in range(L)]
        self.g_att = fp.to_fx(w.rms_att)
        self.g_ffn = fp.to_fx(w.rms_ffn)
        self.g_final = fp.to_fx(w.rms_final)
        self.rope_cos, self.rope_sin = fp.rope_tables(cfg.seq_len, cfg.head_size)
        self.inv_sqrt_hs = round(fp.ONE / np.sqrt(cfg.head_size))
        self.reset()

    def reset(self):
        c = self.cfg
        shape = (c.n_layers, c.seq_len, c.kv_dim)
        self.k_q = np.zeros(shape, np.int64)
        self.v_q = np.zeros(shape, np.int64)
        self.k_s = np.zeros((c.n_layers, c.seq_len, c.n_kv_heads), np.int64)
        self.v_s = np.zeros_like(self.k_s)

    def forward_fx(self, token: int, pos: int) -> np.ndarray:
        """返回 Q16 定点 logits（int64）。所有运算在电脑上用 fixedpoint 执行。"""
        gen = self.forward_steps(token, pos)
        res = None
        while True:
            try:
                op = gen.send(res)
            except StopIteration as e:
                return e.value
            res = exec_op(op)

    def forward_steps(self, token: int, pos: int):
        """前向计算的生成器版本：每个运算 yield 一个 Op，由执行者（L0 / RTL）算完 send 回结果。

        这个运算序列就是将来片上调度器要执行的"指令流"。
        """
        c = self.cfg
        hs = c.head_size
        # 词嵌入：int8 行 × 行缩放 → Q16（查表，留在电脑端）
        x = fp.rshift_round(self.emb.q[token].astype(np.int64) * self.emb.SW[token],
                            fp.SW_FRAC - fp.FRAC)
        cos, sin = self.rope_cos[pos], self.rope_sin[pos]

        for l in range(c.n_layers):
            # ---- 注意力
            xb = yield Op("rmsnorm", (x, self.g_att[l]))
            xq, s = yield Op("quant", (xb,))
            q, k, v = yield Op("matmul", [MatmulReq(self.wq[l], xq, s, "qkv"),
                                          MatmulReq(self.wk[l], xq, s, "qkv"),
                                          MatmulReq(self.wv[l], xq, s, "qkv")])
            q = yield Op("rope", (q, cos, sin, hs))
            k = yield Op("rope", (k, cos, sin, hs))
            for h in range(c.n_kv_heads):          # 写 KV Cache：每头量化成 int8 + 移位量
                seg = slice(h * hs, (h + 1) * hs)
                self.k_q[l, pos, seg], self.k_s[l, pos, h] = yield Op("quant", (k[seg],))
                self.v_q[l, pos, seg], self.v_s[l, pos, h] = yield Op("quant", (v[seg],))

            att_out = np.empty(c.dim, np.int64)
            for h in range(c.n_heads):
                kvh = h // c.kv_mul
                seg = slice(kvh * hs, (kvh + 1) * hs)
                qq, qs = yield Op("quant", (q[h * hs:(h + 1) * hs],))
                att_out[h * hs:(h + 1) * hs] = yield Op("attn", (
                    qq, qs, self.k_q[l, :pos + 1, seg], self.k_s[l, :pos + 1, kvh],
                    self.v_q[l, :pos + 1, seg], self.v_s[l, :pos + 1, kvh], self.inv_sqrt_hs))

            xq, s = yield Op("quant", (att_out,))
            (o,) = yield Op("matmul", [MatmulReq(self.wo[l], xq, s, "wo")])
            x = yield Op("resadd", (x, o))

            # ---- 前馈（SwiGLU）
            xb = yield Op("rmsnorm", (x, self.g_ffn[l]))
            xq, s = yield Op("quant", (xb,))
            h1, h3 = yield Op("matmul", [MatmulReq(self.w1[l], xq, s, "w1w3"),
                                         MatmulReq(self.w3[l], xq, s, "w1w3")])
            hb = yield Op("silu_gate", (h1, h3))
            hq, s = yield Op("quant", (hb,))
            (f,) = yield Op("matmul", [MatmulReq(self.w2[l], hq, s, "w2")])
            x = yield Op("resadd", (x, f))

        x = yield Op("rmsnorm", (x, self.g_final))
        xq, s = yield Op("quant", (x,))
        (logits,) = yield Op("matmul", [MatmulReq(self.wcls, xq, s, "wcls")])
        return logits

    def weight_groups(self):
        """按调度顺序列出共享输入的权重组（q/k/v、wo、w1/w3、w2，最后 wcls）。"""
        g = []
        for l in range(self.cfg.n_layers):
            g += [[self.wq[l], self.wk[l], self.wv[l]], [self.wo[l]],
                  [self.w1[l], self.w3[l]], [self.w2[l]]]
        return g + [[self.wcls]]

    def forward(self, token: int, pos: int) -> np.ndarray:
        """与 FloatModel 接口一致：返回浮点 logits。"""
        return (self.forward_fx(token, pos) / fp.ONE).astype(np.float32)
