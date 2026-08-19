"""E3 anchors: the six MLX GAN loss/slice helpers (now single copies in
usvc/train/mlx/losses.py) match hand-computed references, and both training
modules re-export the SAME function objects (identity, no hidden forks).
"""
from __future__ import annotations

import os

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from usvc.train.mlx import losses as L  # noqa: E402
from usvc.train.mlx.modules import rvc as R  # noqa: E402
from usvc.train.mlx.modules import sovits as S  # noqa: E402


def _arr(a):
    return np.asarray(a)


def test_identity_single_copy():
    for n in ("slice_wave_real", "slice_mel", "lsgan_gen_loss", "lsgan_disc_loss",
              "feature_match_loss", "kl_loss"):
        assert getattr(R, n) is getattr(L, n) is getattr(S, n), n


def test_slice_wave_real():
    mx.random.seed(0)
    wave = mx.random.normal((2, 4000, 1))
    ids = mx.array([10, 20])
    got = _arr(L.slice_wave_real(wave, ids, 64, 48))
    w, idxs = _arr(wave), _arr(ids)
    want = np.stack([
        w[i, idxs[i] * 48: idxs[i] * 48 + 64, 0] for i in range(2)
    ])[..., None]
    assert np.array_equal(got, want)


def test_slice_mel():
    mx.random.seed(0)
    mel = mx.random.normal((2, 4, 100))
    ids = mx.array([3, 9])
    got = _arr(L.slice_mel(mel, ids, 12))
    m, idxs = _arr(mel), _arr(ids)
    want = np.stack([m[i, :, idxs[i]: idxs[i] + 12] for i in range(2)])
    assert np.array_equal(got, want)


def test_lsgan_losses():
    mx.random.seed(1)
    rl = [mx.random.normal((3, 8)), mx.random.normal((3, 5))]
    fl = [mx.random.normal((3, 8)), mx.random.normal((3, 5))]
    f32 = lambda t: np.asarray(t, dtype=np.float32)
    g = float(_arr(L.lsgan_gen_loss(fl)))
    want_g = np.sum([np.mean((f32(f) - np.float32(1.0)) ** 2) for f in fl])
    assert np.isclose(g, want_g, rtol=0, atol=1e-5)
    d = float(_arr(L.lsgan_disc_loss(rl, fl)))
    want_d = np.sum([np.mean((f32(r) - np.float32(1.0)) ** 2) + np.mean(f32(f) ** 2)
                     for r, f in zip(rl, fl)])
    assert np.isclose(d, want_d, rtol=0, atol=1e-5)


def test_feature_match_loss():
    mx.random.seed(2)
    rf = [[mx.random.normal((2, 4)), mx.random.normal((2, 4))], [mx.random.normal((2, 6))]]
    ff = [[mx.random.normal((2, 4)), mx.random.normal((2, 4))], [mx.random.normal((2, 6))]]
    got = float(_arr(L.feature_match_loss(rf, ff, weight=2.0)))
    want = 2.0 * np.sum([np.mean(np.abs(_arr(f2) - _arr(r2)))
                         for rl2, fl2 in zip(rf, ff) for r2, f2 in zip(rl2, fl2)])
    assert np.isclose(got, want, rtol=0, atol=1e-7)


def test_kl_loss_masked_and_plain():
    mx.random.seed(3)
    B, T, C = 2, 6, 4
    z_p = mx.random.normal((B, T, C))
    logs_q = mx.random.normal((B, T, C))
    m_p = mx.random.normal((B, T, C))
    logs_p = mx.random.normal((B, T, C))
    mask = mx.array([[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])[:, :, None]  # [B, T, 1]
    t32 = lambda t: np.asarray(t, dtype=np.float32)
    kl = (t32(logs_p) - t32(logs_q) - np.float32(0.5)
          + np.float32(0.5) * (t32(z_p) - t32(m_p)) ** 2 * np.exp(np.float32(-2.0) * t32(logs_p)))
    got = float(_arr(L.kl_loss(z_p, logs_q, m_p, logs_p, mask)))
    mask_np = _arr(mask).astype(np.float32)
    want = float((kl * mask_np).sum() / (mask_np.sum() + np.float32(1e-8)))
    assert np.isclose(got, want, rtol=0, atol=1e-5)
    got_plain = float(_arr(L.kl_loss(z_p, logs_q, m_p, logs_p, None)))
    assert np.isclose(got_plain, kl.mean(), rtol=0, atol=1e-5)