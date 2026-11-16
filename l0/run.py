"""命令行入口。

  python -m l0.run --mode float   --prompt "Once upon a time"
  python -m l0.run --mode hw
  python -m l0.run --mode compare  # 浮点 vs 硬件等价，并打印位宽需求
  python -m l0.run --random --mode compare   # 不需要下载模型
"""
import argparse
import time

import numpy as np

from . import fixedpoint as fp
from .config import TINY
from .ref_float import FloatModel
from .ref_hw import HWModel
from .tokenizer import Tokenizer
from .weights import load_llama2c, random_weights


def generate(model, prompt_tokens, steps):
    """贪心解码（温度 0，结果确定）。提示词逐个喂入，之后取 argmax。"""
    model.reset()
    tokens = [prompt_tokens[0]]
    t0 = time.time()
    for pos in range(steps):
        logits = model.forward(tokens[-1], pos)
        if pos + 1 < len(prompt_tokens):
            nxt = prompt_tokens[pos + 1]
        else:
            nxt = int(np.argmax(logits))
        if nxt == Tokenizer.BOS:
            break
        tokens.append(nxt)
    return tokens, len(tokens) / (time.time() - t0)


def compare(fm, hm, tokens):
    """教师强制：两模型喂同一序列，逐位置比较 logits。"""
    fm.reset()
    hm.reset()
    top1, cos, maxerr = [], [], []
    for pos, tok in enumerate(tokens[:-1]):
        a = fm.forward(tok, pos)
        b = hm.forward(tok, pos)
        top1.append(int(np.argmax(a)) == int(np.argmax(b)))
        with np.errstate(all="ignore"):  # macOS BLAS 误报，见 pytest.ini
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        maxerr.append(float(np.max(np.abs(a - b))))
    return np.mean(top1), np.min(cos), np.mean(cos), np.max(maxerr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stories15M.bin")
    ap.add_argument("--tokenizer", default="models/tokenizer.bin")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--mode", choices=["float", "hw", "compare"], default="compare")
    ap.add_argument("--random", action="store_true", help="用随机权重的小模型（无需下载）")
    args = ap.parse_args()

    if args.random:
        cfg, w, tok = TINY, random_weights(TINY), None
        prompt = [1, 5, 17, 42]
    else:
        cfg, w = load_llama2c(args.model)
        tok = Tokenizer.load(args.tokenizer, cfg.vocab_size)
        prompt = tok.encode(args.prompt)
    steps = min(args.steps, cfg.seq_len)
    show = (lambda t: tok.decode(t)) if tok else (lambda t: str(t))
    print(f"配置：{cfg}")

    if args.mode in ("float", "compare"):
        fm = FloatModel(cfg, w)
        f_tokens, f_tps = generate(fm, prompt, steps)
        print(f"\n[浮点] {f_tps:.1f} tok/s\n{show(f_tokens)}")
    if args.mode in ("hw", "compare"):
        hm = HWModel(cfg, w)
        h_tokens, h_tps = generate(hm, prompt, steps)
        print(f"\n[硬件等价] {h_tps:.1f} tok/s\n{show(h_tokens)}")
    if args.mode == "compare":
        same = next((i for i, (a, b) in enumerate(zip(f_tokens, h_tokens)) if a != b), None)
        print("\n生成序列：" + ("完全一致" if same is None and len(f_tokens) == len(h_tokens)
                            else f"在第 {same} 个 token 处开始不同"))
        t1, cmin, cmean, merr = compare(fm, hm, f_tokens)
        print(f"教师强制逐位置对比（{len(f_tokens) - 1} 个位置）：")
        print(f"  top-1 一致率 {t1:.1%}，logits 余弦相似度 平均 {cmean:.5f} / 最低 {cmin:.5f}，"
              f"最大绝对误差 {merr:.4f}")
        print("\n硬件位宽需求（有符号，含符号位）：")
        print(fp.MON.report())


if __name__ == "__main__":
    main()
