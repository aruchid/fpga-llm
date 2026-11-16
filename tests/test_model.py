import numpy as np

from l0.config import TINY
from l0.ref_float import FloatModel
from l0.ref_hw import HWModel
from l0.run import compare, generate
from l0.tokenizer import Tokenizer
from l0.weights import load_llama2c, random_weights, save_llama2c


def test_weights_roundtrip(tmp_path):
    for shared in (True, False):
        w = random_weights(TINY, seed=1, shared=shared)
        p = tmp_path / f"m{shared}.bin"
        save_llama2c(str(p), TINY, w)
        cfg, w2 = load_llama2c(str(p))
        assert cfg == TINY and w2.shared_classifier == shared
        for name in ("token_embedding", "wq", "wk", "w2", "rms_final", "wcls"):
            assert np.array_equal(getattr(w, name), getattr(w2, name))


def test_hw_tracks_float():
    w = random_weights(TINY, seed=2)
    fm, hm = FloatModel(TINY, w), HWModel(TINY, w)
    tokens, _ = generate(fm, [1, 7, 9], 24)
    top1, cmin, cmean, _ = compare(fm, hm, tokens)
    assert cmean > 0.995 and cmin > 0.98 and top1 > 0.85


def test_hw_is_deterministic():
    w = random_weights(TINY, seed=3)
    a, b = HWModel(TINY, w), HWModel(TINY, w)
    for pos, tok in enumerate([1, 4, 8, 15, 16, 23, 42]):
        assert np.array_equal(a.forward_fx(tok, pos), b.forward_fx(tok, pos))


def _tiny_tokenizer():
    vocab = [b"<unk>", b"<s>", b"</s>"] + [f"<0x{i:02X}>".encode() for i in range(256)]
    extra = [b" ", b"h", b"e", b"l", b"o", b"he", b"ll", b"hell", b"hello", b" hello"]
    vocab += extra
    scores = [0.0] * 259 + [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    return Tokenizer(vocab, scores)


def test_tokenizer():
    t = _tiny_tokenizer()
    ids = t.encode("hello")
    assert ids == [1, t.lookup[b" hello"]]
    assert t.decode(ids) == "hello"
    ids = t.encode("hé")                    # é 不在词表 → 字节回退
    assert t.decode(ids) == "hé"
