"""逐层对比：24 层跑完，看 cos 是平滑衰减还是某层崩掉。"""

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

    tok_emb = wt.raw("token_embedding.weight").astype(np.float32)
    pos = wt.raw("positional_embedding").astype(np.float32)
    xf = tok_emb[np.asarray(ids)] + pos
    xq = fx.to_fx(tok_emb.astype(np.float64))[np.asarray(ids)] + fx.to_fx(pos.astype(np.float64))

    n_head, hd, L = 16, 64, 77
    causal = np.triu(np.ones((L, L), dtype=bool), k=1)
    for li in range(24):
        p = f"transformer.resblocks.{li}."

        h_f = sd_clip.layernorm_f32(xf, wt.raw(p + "ln_1.weight"), wt.raw(p + "ln_1.bias"))
        h_q = fx.layernorm_fx(xq, fx.to_fx(wt.raw(p + "ln_1.weight").astype(np.float64)),
                              fx.to_fx(wt.raw(p + "ln_1.bias").astype(np.float64)))

        # attention: 浮点
        qkv_f = h_f @ wt.raw(p + "attn.in_proj_weight").T + wt.raw(p + "attn.in_proj_bias")
        Qf, Kf, Vf = np.split(qkv_f, 3, axis=-1)
        att_f = np.empty((L, 1024), dtype=np.float32)
        for h in range(n_head):
            sl = slice(h * hd, (h + 1) * hd)
            sc = (Qf[:, sl] @ Kf[:, sl].T) / np.sqrt(np.float32(hd))
            sc = np.where(causal, -1e9, sc)
            sc -= sc.max(1, keepdims=True)
            pf = np.exp(sc); pf /= pf.sum(1, keepdims=True)
            att_f[:, sl] = pf @ Vf[:, sl]
        o_f = (att_f @ wt.raw(p + "attn.out_proj.weight").T + wt.raw(p + "attn.out_proj.bias"))
        xf = xf + o_f

        # attention: 定点
        qkv_q = sd_clip._lin_fx(eng, wt.raw(p + "attn.in_proj_weight"), wt.raw(p + "attn.in_proj_bias"), h_q)
        Qq, Kq, Vq = np.split(qkv_q, 3, axis=-1)
        att_q = np.empty((L, 1024), dtype=np.int64)
        for h in range(n_head):
            sl = slice(h * hd, (h + 1) * hd)
            QQ = fx.rshift_round(Qq[:, sl] * fx.to_fx(1.0/np.sqrt(float(hd))), 16)
            qq, qs = quantize_shift_batch(QQ)
            kq, ks = quantize_shift_batch(Kq[:, sl])
            vq, vs = quantize_shift_batch(np.ascontiguousarray(Vq[:, sl].T))
            dots = eng.dot_i8(kq, np.ascontiguousarray(qq.T))
            sc = fx.rshift_round(dots.astype(np.int64), 32 - qs[None, :] - ks[:, None]).T
            sc = np.where(causal, -(1 << 26), sc)
            pr = fx.softmax_fx(sc, axis=1)
            vacc = np.ascontiguousarray(vq.T).astype(np.int64) << vs[None, :]     # Q16
            att_q[:, sl] = fx.rshift_round(pr @ vacc, 16)
        o_q = sd_clip._lin_fx(eng, wt.raw(p + "attn.out_proj.weight"), wt.raw(p + "attn.out_proj.bias"), att_q)
        xq = fx.sat32(xq + o_q)

        h2_f = sd_clip.layernorm_f32(xf, wt.raw(p + "ln_2.weight"), wt.raw(p + "ln_2.bias"))
        m_f = sd_clip.gelu(h2_f @ wt.raw(p + "mlp.c_fc.weight").T + wt.raw(p + "mlp.c_fc.bias"))
        m_f = m_f @ wt.raw(p + "mlp.c_proj.weight").T + wt.raw(p + "mlp.c_proj.bias")
        xf = xf + m_f

        h2_q = fx.layernorm_fx(xq, fx.to_fx(wt.raw(p + "ln_2.weight").astype(np.float64)),
                               fx.to_fx(wt.raw(p + "ln_2.bias").astype(np.float64)))
        m_q = sd_clip.gelu_fx(sd_clip._lin_fx(eng, wt.raw(p + "mlp.c_fc.weight"), wt.raw(p + "mlp.c_fc.bias"), h2_q))
        m_q = sd_clip._lin_fx(eng, wt.raw(p + "mlp.c_proj.weight"), wt.raw(p + "mlp.c_proj.bias"), m_q)
        xq = fx.sat32(xq + m_q)

        print(f"layer {li:2d}  cos={cos(xf, xq):.6f}  std f={xf.std():6.2f} q={fx.from_fx(xq).std():6.2f}  "
              f"max|f|={np.abs(xf).max():8.2f}  饱和累计={eng.stats.saturations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
