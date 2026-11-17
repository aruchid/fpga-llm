"""逐层对比 CLIP 浮点与定点实现，定位发散点。"""

import numpy as np

import fx
import mvm32
import sd_clip
from sd_vae import quantize_shift_batch

CKPT = r"E:\ComfyUI\models\checkpoints\sd_turbo.safetensors"


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    tok = sd_clip.CLIPTokenizer()
    ids = tok.encode("a cozy cabin")
    wt = sd_clip.ClipWeights(CKPT)
    eng = mvm32.MVM32()

    tok_emb = wt.raw("token_embedding.weight").astype(np.float32)
    pos = wt.raw("positional_embedding").astype(np.float32)
    xf = tok_emb[np.asarray(ids)] + pos
    xq = fx.to_fx(tok_emb.astype(np.float64))[np.asarray(ids)] + fx.to_fx(pos.astype(np.float64))
    print(f"emb        cos={cos(xf, xq):.6f} mean|Δ|={np.mean(np.abs(xf - fx.from_fx(xq))):.4f}")

    p = "transformer.resblocks.0."
    # ln_1
    h_f = sd_clip.layernorm_f32(xf, wt.raw(p + "ln_1.weight"), wt.raw(p + "ln_1.bias"))
    h_q = fx.layernorm_fx(xq, fx.to_fx(wt.raw(p + "ln_1.weight").astype(np.float64)),
                          fx.to_fx(wt.raw(p + "ln_1.bias").astype(np.float64)))
    print(f"ln_1       cos={cos(h_f, h_q):.6f} std f={h_f.std():.3f} q={fx.from_fx(h_q).std():.3f}")

    # qkv
    qkv_f = h_f @ wt.raw(p + "attn.in_proj_weight").T + wt.raw(p + "attn.in_proj_bias")
    qkv_q = sd_clip._lin_fx(eng, wt.raw(p + "attn.in_proj_weight"), wt.raw(p + "attn.in_proj_bias"), h_q)
    print(f"qkv        cos={cos(qkv_f, qkv_q):.6f} max|Δ|={np.max(np.abs(qkv_f - fx.from_fx(qkv_q))):.3f}")

    # attention
    n_head, hd = 16, 64
    Q = qkv_q[:, :1024]
    Kq = qkv_q[:, 1024:2048]
    Vq = qkv_q[:, 2048:]
    out_q = np.empty((77, 1024), dtype=np.int64)
    Qf, Kf, Vf = np.split(qkv_f, 3, axis=-1)
    for h in range(n_head):
        sl = slice(h * hd, (h + 1) * hd)
        qq, qs = quantize_shift_batch(Q[:, sl])
        kq, ks = quantize_shift_batch(Kq[:, sl])
        vq, vs = quantize_shift_batch(np.ascontiguousarray(Vq[:, sl].T))
        dots = eng.dot_i8(kq, np.ascontiguousarray(qq.T))
        sc = fx.rshift_round(dots.astype(np.int64), 32 - qs[None, :] - ks[:, None]).T
        sc = np.where(np.triu(np.ones((77, 77), bool), k=1), -(1 << 26), sc)
        pr = fx.softmax_fx(sc, axis=1)
        # PV：P 保持 Q16，V 用 int8
        vacc = np.ascontiguousarray(vq.T).astype(np.int64) << vs[None, :]
        acc = pr @ vacc
        out_q[:, sl] = fx.rshift_round(acc, 16)
        if h == 0:
            q = Qf[:, sl].reshape(77, hd)
            k_ = Kf[:, sl].reshape(77, hd)
            v = Vf[:, sl].reshape(77, hd)
            scf = (q @ k_.T) / np.sqrt(np.float32(hd))
            scf = np.where(np.triu(np.ones((77, 77), bool), k=1), -1e9, scf)
            scf -= scf.max(1, keepdims=True)
            pf = np.exp(scf); pf /= pf.sum(1, keepdims=True)
            print(f"  head0 sc cos={cos(scf, sc):.6f}  p cos={cos(pf, pr):.6f}  "
                  f"p range q=[{pr.min()},{pr.max()}]")
            of = pf @ v
            print(f"  head0 out cos={cos(of, fx.from_fx(out_q[:, sl])):.6f}")

    o_f = out_q  # 占位
    # 和浮点 attention 整体比
    of_all = np.empty((77, 1024), dtype=np.float32)
    for h in range(n_head):
        sl = slice(h * hd, (h + 1) * hd)
        q = Qf[:, sl]; k_ = Kf[:, sl]; v = Vf[:, sl]
        scf = (q @ k_.T) / np.sqrt(np.float32(hd))
        scf = np.where(np.triu(np.ones((77, 77), bool), k=1), -1e9, scf)
        scf -= scf.max(1, keepdims=True)
        pf = np.exp(scf); pf /= pf.sum(1, keepdims=True)
        of_all[:, sl] = pf @ v
    print(f"attn out   cos={cos(of_all, fx.from_fx(out_q)):.6f}")

    o_f2 = of_all @ wt.raw(p + "attn.out_proj.weight").T + wt.raw(p + "attn.out_proj.bias")
    o_q2 = sd_clip._lin_fx(eng, wt.raw(p + "attn.out_proj.weight"), wt.raw(p + "attn.out_proj.bias"), out_q)
    print(f"out_proj   cos={cos(o_f2, o_q2):.6f}")

    x1f = xf + o_f2
    x1q = fx.sat32(xq + o_q2)
    print(f"x+attn     cos={cos(x1f, x1q):.6f}")

    h2f = sd_clip.layernorm_f32(x1f, wt.raw(p + "ln_2.weight"), wt.raw(p + "ln_2.bias"))
    h2q = fx.layernorm_fx(x1q, fx.to_fx(wt.raw(p + "ln_2.weight").astype(np.float64)),
                          fx.to_fx(wt.raw(p + "ln_2.bias").astype(np.float64)))
    print(f"ln_2       cos={cos(h2f, h2q):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
