"""单层对照：float32 vs Q16 精确整数 vs int8 量化，判断误差来源。"""

import numpy as np

import fx
import mvm32
import sd_clip
from sd_vae import quantize_shift_batch

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"
PROMPT = "a cozy cabin in a snowy pine forest at dusk, warm light in the windows, photorealistic"


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode(PROMPT)
    wt = sd_clip.ClipWeights(CKPT)
    eng = mvm32.MVM32()

    def lin_q16_exact(x, w, b):
        """Q16 × Q16 的精确整数矩阵乘（不经 int8 量化）。"""
        w16 = np.round(w.astype(np.float64) * 65536).astype(np.int64)
        acc = np.rint(x.astype(np.float64) @ w16.astype(np.float64).T).astype(np.int64)
        return (fx.rshift_round(acc, 16) + fx.to_fx(b.astype(np.float64))[None, :])

    tok_emb = wt.raw("token_embedding.weight").astype(np.float32)
    pos = wt.raw("positional_embedding").astype(np.float32)
    xf = tok_emb[np.asarray(ids)] + pos
    xq = fx.to_fx(tok_emb.astype(np.float64))[np.asarray(ids)] + fx.to_fx(pos.astype(np.float64))

    n_head, hd, L = 16, 64, 77
    causal = np.triu(np.ones((L, L), dtype=bool), k=1)
    xs = {"float": xf, "q16": xq, "int8": xq.copy()}

    def attn(mode, h, p):
        """h: 当前层 LN 后的激活（float 或 Q16）。返回 out_proj 的输入。"""
        if mode == "float":
            qkv = h @ wt.raw(p + "attn.in_proj_weight").T + wt.raw(p + "attn.in_proj_bias")
            Q, K, V = np.split(qkv, 3, axis=-1)
            out = np.empty((L, 1024), np.float32)
            for hh in range(n_head):
                sl = slice(hh * hd, (hh + 1) * hd)
                sc = (Q[:, sl] @ K[:, sl].T) / np.sqrt(np.float32(hd))
                sc = np.where(causal, -1e9, sc)
                sc -= sc.max(1, keepdims=True)
                pf = np.exp(sc)
                pf /= pf.sum(1, keepdims=True)
                out[:, sl] = pf @ V[:, sl]
            return out
        if mode == "q16":
            qkv = lin_q16_exact(h, wt.raw(p + "attn.in_proj_weight"), wt.raw(p + "attn.in_proj_bias"))
            Q, K, V = np.split(qkv, 3, axis=-1)
            out = np.empty((L, 1024), np.int64)
            scale = fx.to_fx(1.0 / np.sqrt(float(hd)))
            for hh in range(n_head):
                sl = slice(hh * hd, (hh + 1) * hd)
                Qs = fx.rshift_round(Q[:, sl] * scale, 16)
                sc = np.rint(Qs.astype(np.float64) @ K[:, sl].astype(np.float64).T / 65536).astype(np.int64)
                sc = np.where(causal, -(1 << 30), sc)
                sc = np.rint(sc.astype(np.float64) * 142).astype(np.int64)  # 乘 log2e 近似，只为形状
                pf = fx.softmax_fx(sc, axis=1)
                out[:, sl] = fx.rshift_round(
                    np.rint(pf.astype(np.float64) @ V[:, sl].astype(np.float64)).astype(np.int64), 16)
            return out
        # int8
        qkv = sd_clip._lin_fx(eng, wt.raw(p + "attn.in_proj_weight"), wt.raw(p + "attn.in_proj_bias"), h)
        Q, K, V = np.split(qkv, 3, axis=-1)
        Q = fx.rshift_round(Q * fx.to_fx(1.0 / np.sqrt(float(hd))), 16)
        out = np.empty((L, 1024), np.int64)
        for hh in range(n_head):
            sl = slice(hh * hd, (hh + 1) * hd)
            qq, qs = quantize_shift_batch(Q[:, sl])
            kq, ks = quantize_shift_batch(K[:, sl])
            vq, vs = quantize_shift_batch(np.ascontiguousarray(V[:, sl].T))
            dots = eng.dot_i8(kq, np.ascontiguousarray(qq.T))
            sc = fx.rshift_round(dots.astype(np.int64), 32 - qs[None, :] - ks[:, None]).T
            sc = np.where(causal, -(1 << 26), sc)
            pr = fx.softmax_fx(sc, axis=1)
            vacc = np.ascontiguousarray(vq.T).astype(np.int64) << vs[None, :]
            out[:, sl] = fx.rshift_round(pr @ vacc, 16)
        return out

    p = "transformer.resblocks.0."
    for mode in ("float", "q16", "int8"):
        if mode == "float":
            h = sd_clip.layernorm_f32(xf, wt.raw(p + "ln_1.weight"), wt.raw(p + "ln_1.bias"))
        else:
            h = fx.layernorm_fx(xq, fx.to_fx(wt.raw(p + "ln_1.weight").astype(np.float64)),
                                fx.to_fx(wt.raw(p + "ln_1.bias").astype(np.float64)))
        a = attn(mode, h, p)
        if mode == "float":
            o = a @ wt.raw(p + "attn.out_proj.weight").T + wt.raw(p + "attn.out_proj.bias")
        elif mode == "q16":
            o = lin_q16_exact(a, wt.raw(p + "attn.out_proj.weight"), wt.raw(p + "attn.out_proj.bias"))
        else:
            o = sd_clip._lin_fx(eng, wt.raw(p + "attn.out_proj.weight"), wt.raw(p + "attn.out_proj.bias"), a)

        if mode == "float":
            y = xf + o
            h2 = sd_clip.layernorm_f32(y, wt.raw(p + "ln_2.weight"), wt.raw(p + "ln_2.bias"))
            m = sd_clip.gelu(h2 @ wt.raw(p + "mlp.c_fc.weight").T + wt.raw(p + "mlp.c_fc.bias"))
            m = m @ wt.raw(p + "mlp.c_proj.weight").T + wt.raw(p + "mlp.c_proj.bias")
            y = y + m
            xs[mode] = y
            print(f"{mode:6s} x std={y.std():.4f}")
        elif mode == "q16":
            y = fx.sat32(xq + o)
            h2 = fx.layernorm_fx(y, fx.to_fx(wt.raw(p + "ln_2.weight").astype(np.float64)),
                                 fx.to_fx(wt.raw(p + "ln_2.bias").astype(np.float64)))
            m = sd_clip.gelu_fx(lin_q16_exact(h2, wt.raw(p + "mlp.c_fc.weight"), wt.raw(p + "mlp.c_fc.bias")))
            m = lin_q16_exact(m, wt.raw(p + "mlp.c_proj.weight"), wt.raw(p + "mlp.c_proj.bias"))
            y = fx.sat32(y + m)
            xs[mode] = y
            print(f"{mode:6s} x std={fx.from_fx(y).std():.4f}  cos vs float={cos(xs['float'], y):.6f}")
        else:
            y = fx.sat32(xq + o)
            h2 = fx.layernorm_fx(y, fx.to_fx(wt.raw(p + "ln_2.weight").astype(np.float64)),
                                 fx.to_fx(wt.raw(p + "ln_2.bias").astype(np.float64)))
            m = sd_clip.gelu_fx(sd_clip._lin_fx(eng, wt.raw(p + "mlp.c_fc.weight"), wt.raw(p + "mlp.c_fc.bias"), h2))
            m = sd_clip._lin_fx(eng, wt.raw(p + "mlp.c_proj.weight"), wt.raw(p + "mlp.c_proj.bias"), m)
            y = fx.sat32(y + m)
            xs[mode] = y
            print(f"{mode:6s} x std={fx.from_fx(y).std():.4f}  cos vs float={cos(xs['float'], y):.6f}")

    print(f"\nq16 vs float cos={cos(xs['float'], xs['q16']):.6f}")
    print(f"int8 vs float cos={cos(xs['float'], xs['int8']):.6f}")
    print(f"int8 vs q16  cos={cos(xs['q16'], xs['int8']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
