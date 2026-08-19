"""E5 anchors: MLX discriminators are now single shared definitions.

- sovits/rvc re-export the SAME class objects from usvc.nn.mlx._gan
  (identity assert below)
- MultiPeriodDiscriminatorV2 (periods=8) produces bit-identical outputs to
  the shared MultiPeriodDiscriminator(periods=(2,3,5,7,11,17,23,37)) with
  copied weights (deterministic ops, MLX interpreter mode)
- use_spectral_norm attribute surface preserved

Run with MLX_DISABLE_COMPILE=1 (CPU JIT needs gcc<13 on this box; the env
var falls back to the interpreter, which is fine for numeric anchors).
"""
from __future__ import annotations

import os

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from usvc.nn.mlx import _gan as G  # noqa: E402
from usvc.nn.mlx.backends import rvc as R  # noqa: E402
from usvc.nn.mlx.backends import sovits as S  # noqa: E402


def test_shared_class_identity():
    assert S.DiscriminatorP is G.DiscriminatorP
    assert S.DiscriminatorS is G.DiscriminatorS
    assert S.MultiPeriodDiscriminator is G.MultiPeriodDiscriminator
    assert issubclass(R.MultiPeriodDiscriminatorV2, G.MultiPeriodDiscriminator)


def test_period_lists_preserved():
    assert [d.period for d in S.MultiPeriodDiscriminator().discriminators[1:]] == [2, 3, 5, 7, 11]
    rvc_discs = R.MultiPeriodDiscriminatorV2().discriminators
    assert [d.period for d in rvc_discs[1:]] == [2, 3, 5, 7, 11, 17, 23, 37]
    assert len(rvc_discs) == 9  # DiscriminatorS + 8x DiscriminatorP


def test_attribute_surface():
    for cls in (S.MultiPeriodDiscriminator, R.MultiPeriodDiscriminatorV2):
        d = cls(use_spectral_norm=False)
        assert d.discriminators  # frozen attr
    p = S.DiscriminatorP(7, use_spectral_norm=False)
    assert p.period == 7 and p.use_spectral_norm is False


def _copied_weights(src, dst):
    dst.load_weights(dict(src.parameters()), strict=False)


def _list_equal(la, lb):
    for ta, tb in zip(la, lb):
        if isinstance(ta, list):
            _list_equal(ta, tb)  # fmap: list per-discriminator of per-layer
        else:
            assert np.array_equal(np.asarray(ta), np.asarray(tb))


def test_mpdv2_bit_equal_to_shared_periods():
    mx.random.seed(0)
    a = R.MultiPeriodDiscriminatorV2()
    b = G.MultiPeriodDiscriminator(periods=(2, 3, 5, 7, 11, 17, 23, 37))
    assert len(a.discriminators) == len(b.discriminators)
    _copied_weights(a, b)
    y = mx.random.normal((1, 8192, 1))     # 8192 % 17 != 0 -> reflect-pad path
    y_hat = mx.random.normal((1, 8192, 1))
    oa, ob = a(y, y_hat), b(y, y_hat)
    for ra, rb in zip(oa, ob):  # y_d_rs, y_d_gs, fmap_rs, fmap_gs
        _list_equal(ra, rb)


def test_mpd_sovits_smoke():
    mx.random.seed(1)
    d = S.MultiPeriodDiscriminator()
    y = mx.random.normal((1, 8192, 1))
    y_hat = mx.random.normal((1, 8192, 1))
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = d(y, y_hat)
    assert len(y_d_rs) == 6 and len(fmap_rs) == 6
    assert all(x.ndim == 2 for x in y_d_rs)  # [B, -1] per discriminator
    # DiscriminatorS has 6 convs+conv_post (7 fmaps); each DiscriminatorP 5+1 (6)
    assert [len(f) for f in fmap_rs] == [7] + [6] * 5