import math

import numpy as np
import pytest

from l0 import fixedpoint as fp
from l0.ref_float import rmsnorm, silu, softmax

rng = np.random.default_rng(0)


def test_rshift_round():
    assert list(fp.rshift_round([5, -5, 6, -6, 7], 1)) == [3, -2, 3, -3, 4]
    assert int(fp.rshift_round(3, -2)) == 12
    assert list(fp.rshift_round([8, 8, 8], [0, 2, -1])) == [8, 2, 16]


@pytest.mark.parametrize("val", [1e-4, 0.01, 0.37, 1.0, 2.5, 3.99, 100.0, 12345.6])
def test_rsqrt(val):
    r = fp.rsqrt_fx(round(val * fp.ONE)) / fp.ONE
    # 相对误差 2e-4，或输出 Q16 的 2 个 LSB（结果很小时受输出精度限制）
    assert r == pytest.approx(1 / math.sqrt(round(val * fp.ONE) / fp.ONE),
                              rel=2e-4, abs=2 / fp.ONE)


@pytest.mark.parametrize("val", [1.0, 1.5, 1.999, 7.3, 288.0, 1e5])
def test_recip(val):
    m = round(val * fp.ONE)
    r = fp.recip_fx(m, 16, 32) / 2 ** 32
    assert r == pytest.approx(fp.ONE / m, rel=1e-4)


def test_exp2():
    t = np.linspace(-20, 0, 2001)
    got = fp.exp2_neg_fx(fp.to_fx(t)) / fp.ONE
    assert np.max(np.abs(got - 2 ** t)) < 5e-5


def test_softmax():
    for _ in range(20):
        x = rng.standard_normal(rng.integers(1, 200)) * 4
        got = fp.softmax_fx(fp.to_fx(x)) / fp.ONE
        assert np.max(np.abs(got - softmax(x.astype(np.float32)))) < 2e-4
        assert abs(got.sum() - 1) < 5e-3


def test_sigmoid_silu():
    x = np.linspace(-12, 12, 4001)
    got = fp.silu_fx(fp.to_fx(x)) / fp.ONE
    assert np.max(np.abs(got - silu(x.astype(np.float32)))) < 2e-3


def test_rmsnorm():
    for scale in (0.05, 1.0, 30.0):
        x = rng.standard_normal(288) * scale
        g = 1 + 0.1 * rng.standard_normal(288)
        got = fp.rmsnorm_fx(fp.to_fx(x), fp.to_fx(g)) / fp.ONE
        ref = rmsnorm(x.astype(np.float32), g.astype(np.float32))
        assert np.max(np.abs(got - ref)) < 2e-2 * np.max(np.abs(ref))


def test_quantize_shift():
    for scale in (1e-3, 0.5, 10, 5000):
        x = fp.to_fx(rng.standard_normal(100) * scale)
        q, s = fp.quantize_shift(x)
        assert np.max(np.abs(q)) <= 127
        assert s == 0 or np.max(np.abs(q)) >= 64   # 移位量最小
        assert np.allclose(q * 2.0 ** s, x, atol=2.0 ** s)


def test_int_matmul():
    w = rng.standard_normal((40, 96)) * 0.1
    x = rng.standard_normal(96)
    wq, SW = fp.quantize_weight_rows(w)
    xq, s = fp.quantize_shift(fp.to_fx(x))
    got = fp.int_matmul(wq, SW, xq, s) / fp.ONE
    ref = w @ x
    assert np.max(np.abs(got - ref)) < 0.02 * np.max(np.abs(ref))


def test_rope_matches_float():
    from l0.ref_float import apply_rope, rope_cos_sin
    hs = 48
    cos_t, sin_t = fp.rope_tables(16, hs)
    v = rng.standard_normal(hs * 3)
    for pos in (0, 1, 15):
        got = fp.rope_fx(fp.to_fx(v), cos_t[pos], sin_t[pos], hs) / fp.ONE
        c, s = rope_cos_sin(pos, hs)
        assert np.max(np.abs(got - apply_rope(v.astype(np.float32), c, s, hs))) < 1e-3
