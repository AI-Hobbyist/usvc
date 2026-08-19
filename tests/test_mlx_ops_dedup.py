"""E4 anchors: shared _ops kernels (norm_spec/denorm_spec, ode_euler/rk4,
reflow_loss layouts, sequence_mask, stft_center_complex/istft) are bit-equal
to the ORIGINAL per-backend bodies (embedded verbatim below), and the backend
class methods delegate to them (same results on same inputs).

f0_to_coarse: rvc/sovits formulas are mathematically equal but NOT bit-equal
(1/220k round-boundary token flip) — the test guards that both stay separate.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from usvc.nn.mlx import _ops  # noqa: E402


def _eq(a, b):
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# old per-backend bodies (verbatim from pre-E4 source)
# ---------------------------------------------------------------------------

def _ref_norm(x, smin, smax, cast):
    if cast:
        x = x.astype(mx.float32)
    return (x - smin) / (smax - smin) * 2 - 1


def _ref_denorm(x, smin, smax):
    return (x + 1) / 2 * (smax - smin) + smin


def _ref_euler(v, x, t, dt, cond):
    x = x + v(x, 1000 * t, cond) * dt
    return x, t + dt


def _ref_rk4(v, x, t, dt, cond):
    k1 = v(x, 1000 * t, cond)
    k2 = v(x + 0.5 * k1 * dt, 1000 * (t + 0.5 * dt), cond)
    k3 = v(x + 0.5 * k2 * dt, 1000 * (t + 0.5 * dt), cond)
    k4 = v(x + k3 * dt, 1000 * (t + dt), cond)
    x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return x, t + dt


def _ref_reflow_b1mt(v, x_1, t, cond, loss_type="l2_lognorm", mask=None, x_0=None):
    """ddsp6 RectifiedFlow.reflow_loss body (layout [B,1,M,T])."""
    if x_0 is None:
        x_0 = mx.random.normal(x_1.shape, dtype=x_1.dtype)
    x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
    v_pred = v(x_t, 1000 * t, cond)
    if mask is not None:
        m = mask.astype(x_1.dtype)[:, None, None, :]
        elem = mx.square(x_1 - x_0 - v_pred)
        if loss_type == "l2_lognorm":
            weights = 0.398942 / t / (1 - t) * mx.exp(-0.5 * mx.square(mx.log(t / (1 - t))))
            elem = weights[:, None, None, None] * elem
        loss = (elem * m).sum() / (m.sum() * elem.shape[2] * elem.shape[3] + 1e-8)
    elif loss_type == "l1":
        loss = mx.abs(x_1 - x_0 - v_pred).mean()
    elif loss_type == "l2":
        loss = mx.mean(mx.square(x_1 - x_0 - v_pred))
    elif loss_type == "l2_lognorm":
        weights = 0.398942 / t / (1 - t) * mx.exp(-0.5 * mx.square(mx.log(t / (1 - t))))
        loss = mx.mean(weights[:, None, None, None] * mx.square(x_1 - x_0 - v_pred))
    else:
        raise NotImplementedError(loss_type)
    return loss


def _ref_reflow_btm(v, x_1, t, cond, loss_type="l2_lognorm", mask=None, x_0=None):
    """reflow_shallow/reflow_vae reflow_loss body (layout [B,T,M])."""
    if x_0 is None:
        x_0 = mx.random.normal(x_1.shape, dtype=x_1.dtype)
    x_t = x_0 + t[:, None, None] * (x_1 - x_0)
    v_pred = v(x_t, 1000 * t, cond)
    if mask is not None:
        m = mask.astype(x_1.dtype)[:, :, None]
        elem = mx.square(x_1 - x_0 - v_pred)
        if loss_type == "l2_lognorm":
            weights = 0.398942 / t / (1 - t) * mx.exp(-0.5 * mx.square(mx.log(t / (1 - t))))
            elem = weights[:, None, None] * elem
        loss = (elem * m).sum() / (m.sum() * x_1.shape[2] + 1e-8)
    elif loss_type == "l1":
        loss = mx.abs(x_1 - x_0 - v_pred).mean()
    elif loss_type == "l2":
        loss = mx.mean(mx.square(x_1 - x_0 - v_pred))
    elif loss_type == "l2_lognorm":
        weights = 0.398942 / t / (1 - t) * mx.exp(-0.5 * mx.square(mx.log(t / (1 - t))))
        loss = mx.mean(weights[:, None, None] * mx.square(x_1 - x_0 - v_pred))
    else:
        raise NotImplementedError(loss_type)
    return loss


def _ref_pad_reflect_2d(x, left, right):
    """ddsp6 pre-E4 local reflect pad (mirrors F.pad reflect)."""
    if left == 0 and right == 0:
        return x
    parts = [x]
    if left > 0:
        parts.insert(0, x[:, left:0:-1])
    if right > 0:
        parts.append(x[:, -2:-right - 2:-1])
    return mx.concatenate(parts, axis=-1)


def _ref_stft_center_complex(wav, n_fft, hop, window, pad_mode="reflect"):
    """ddsp6 pre-E4 _stft_center_complex body."""
    B, L = wav.shape
    pad = n_fft // 2
    if pad_mode == "reflect" and pad < L:
        padded = _ref_pad_reflect_2d(wav, pad, pad)
    else:
        padded = mx.pad(wav, [(0, 0), (pad, pad)])
    n_frames = (padded.shape[1] - n_fft) // hop + 1
    idx = mx.arange(n_frames)[:, None] * hop + mx.arange(n_fft)[None, :]
    frames = mx.take(padded, idx, axis=1)
    frames = frames * window
    spec = mx.fft.rfft(frames, n=n_fft, axis=-1)
    return spec.transpose(0, 2, 1)


def _ref_istft_center_complex(spec, n_fft, hop, window):
    """ddsp6 pre-E4 _istft_center_complex body."""
    B, F, n_frames = spec.shape
    frames = mx.fft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1)
    frames = frames * window
    expected = n_fft + hop * (n_frames - 1)
    y = mx.zeros((B, expected))
    win_env = mx.zeros((B, expected))
    w2 = window * window
    for i in range(n_frames):
        sl = slice(i * hop, i * hop + n_fft)
        y = y.at[:, sl].add(frames[:, i])
        win_env = win_env.at[:, sl].add(w2)
    start = n_fft // 2
    end = expected - n_fft // 2
    return y[:, start:end] / win_env[:, start:end]


# ---------------------------------------------------------------------------
# norm / denorm
# ---------------------------------------------------------------------------

def _norm_data():
    mx.random.seed(10)
    return mx.random.normal((2, 12, 8)) * 6 - 2


def test_norm_denorm_bit_equal():
    x = _norm_data()
    for cast in (True, False):
        _eq(_ops.norm_spec(x, -12.0, 2.0, cast=cast), _ref_norm(x, -12.0, 2.0, cast))
    _eq(_ops.denorm_spec(x, -12.0, 2.0), _ref_denorm(x, -12.0, 2.0))
    # roundtrip on the [-1,1] domain (float32 ulp-level, not bit-exact)
    for smin, smax in ((-12.0, 2.0), (0.0, 1.0)):
        y = mx.random.normal((2, 4)) * 2  # within [-1,1] mostly
        rt = _ops.denorm_spec(_ops.norm_spec(y, smin, smax, cast=False), smin, smax)
        assert np.allclose(np.asarray(rt), np.asarray(y), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# ode steps
# ---------------------------------------------------------------------------

def _v(x, t, cond):
    return (x * 0.05).astype(mx.float32)  # t/cond unused: shapes differ across layouts


def test_ode_euler_rk4_bit_equal_and_delegates():
    mx.random.seed(11)
    x = mx.random.normal((2, 12, 8))
    t = mx.array(0.3, dtype=mx.float32)
    dt = 0.125
    cond = mx.random.normal((2, 12, 4))
    for got, ref in [
        (_ops.ode_euler(_v, x, t, dt, cond), _ref_euler(_v, x, t, dt, cond)),
        (_ops.ode_rk4(_v, x, t, dt, cond), _ref_rk4(_v, x, t, dt, cond)),
    ]:
        x2, t2 = got
        xr, tr = ref
        _eq(x2, xr)
        _eq(t2, tr)

    # backend class delegates call the same kernel
    from usvc.nn.mlx.backends.ddsp6 import RectifiedFlow

    rf = RectifiedFlow(_v)
    _eq(rf.sample_euler(x, t, dt, cond)[0], _ops.ode_euler(_v, x, t, dt, cond)[0])
    _eq(rf.sample_rk4(x, t, dt, cond)[0], _ops.ode_rk4(_v, x, t, dt, cond)[0])


def test_reflow_loss_layouts_bit_equal():
    mx.random.seed(12)
    x1_b1mt = mx.random.normal((2, 1, 8, 12))
    x1_btm = mx.random.normal((2, 12, 8))
    t = mx.random.uniform(shape=(2,)) * 0.9 + 0.05
    cond = mx.random.normal((2, 12, 4))
    mask_b1mt = mx.random.uniform(shape=(2, 1, 1, 12)) > 0.4
    mask_btm = mx.random.uniform(shape=(2, 12)) > 0.4
    x0_b = mx.random.normal((2, 12, 8))
    x0_b1mt = mx.random.normal((2, 1, 8, 12))

    # b1mt (ddsp6), explicit x_0 for both sides
    _eq(_ops.reflow_loss(_v, x1_b1mt, t, cond, "l2_lognorm", mask_b1mt, x_0=x0_b1mt,
                         layout="b1mt"),
        _ref_reflow_b1mt(_v, x1_b1mt, t, cond, "l2_lognorm", mask_b1mt, x0_b1mt))
    # b1mt no-mask variants (explicit x_0 both sides)
    for lt in ("l1", "l2", "l2_lognorm"):
        _eq(_ops.reflow_loss(_v, x1_b1mt, t, cond, lt, None, x_0=x0_b1mt, layout="b1mt"),
            _ref_reflow_b1mt(_v, x1_b1mt, t, cond, lt, None, x0_b1mt))
    # btm (shallow), internal noise draw: same seed -> same draw -> bit-equal
    mx.random.seed(13)
    want = _ref_reflow_btm(_v, x1_btm, t, cond, "l2_lognorm", mask_btm, None)
    mx.random.seed(13)
    got = _ops.reflow_loss(_v, x1_btm, t, cond, "l2_lognorm", mask_btm, x_0=None, layout="btm")
    _eq(got, want)
    # btm with explicit x_0 (vae) + no-mask variants
    _eq(_ops.reflow_loss(_v, x1_btm, t, cond, "l2_lognorm", mask_btm, x_0=x0_b, layout="btm"),
        _ref_reflow_btm(_v, x1_btm, t, cond, "l2_lognorm", mask_btm, x0_b))
    for lt in ("l1", "l2", "l2_lognorm"):
        _eq(_ops.reflow_loss(_v, x1_btm, t, cond, lt, None, x_0=x0_b, layout="btm"),
            _ref_reflow_btm(_v, x1_btm, t, cond, lt, None, x0_b))


def test_sequence_mask():
    lengths32 = mx.array([3, 0, 7])
    _eq(_ops.sequence_mask(lengths32),
        mx.arange(7, dtype=mx.int32)[None, :] < lengths32[:, None])
    _eq(_ops.sequence_mask(lengths32, 4),
        mx.arange(4, dtype=mx.int32)[None, :] < lengths32[:, None])
    lengths64 = mx.array([3, 0, 7]).astype(mx.int64)  # int64 input stays bool-out
    assert np.asarray(_ops.sequence_mask(lengths64)).dtype == np.bool_


def test_stft_complex_bit_equal_and_roundtrip():
    mx.random.seed(14)
    wav = mx.random.normal((2, 1600))
    win = mx.array(np.hanning(64).astype(np.float32))
    spec = _ops.stft_center_complex(wav, 64, 32, win)
    _eq(spec, _ref_stft_center_complex(wav, 64, 32, win))
    _eq(_ops.istft_center_complex(spec, 64, 32, win),
        _ref_istft_center_complex(spec, 64, 32, win))
    # zero-pad fallback path (short clip: pad >= L)
    short = mx.random.normal((2, 20))
    spec2 = _ops.stft_center_complex(short, 64, 32, win)
    _eq(spec2, _ref_stft_center_complex(short, 64, 32, win))
    assert np.asarray(spec2).shape == (2, 33, 1)  # L=20 -> 1 frame


# ---------------------------------------------------------------------------
# f0_to_coarse: two formulas must STAY separate (rule d)
# ---------------------------------------------------------------------------

def test_f0_to_coarse_formulas_diverge_on_boundaries():
    """Both stay in [1, 255]; rvc/sovits formulas are NOT bit-identical."""
    from usvc.nn.mlx.backends.rvc import f0_to_coarse as rvc_f2c
    from usvc.nn.mlx.backends.sovits import f0_to_coarse as sovits_f2c

    # deterministic grid: reproduces the documented divergence scan
    # (geomspace 20k + 200k uniform + degenerate boundaries, 1 flip found)
    f0s = np.concatenate([
        np.array([0.0, 1e-8, 1e-4, 0.01, 1.0, 5.0, 49.999, 50.0, 50.001]),
        np.geomspace(1.0, 5000.0, 20000),
        np.random.default_rng(0).uniform(0.0, 2000.0, 200000),
        np.array([1099.999, 1100.0, 1100.001, 5000.0, 1e6]),
    ]).astype(np.float32)
    a = np.asarray(rvc_f2c(mx.array(f0s)))
    b = np.asarray(sovits_f2c(mx.array(f0s)))
    assert (a >= 1).all() and (a <= 255).all()
    assert (b >= 1).all() and (b <= 255).all()
    ndiff = int((a != b).sum())
    assert ndiff >= 1, (
        "formulas became bit-equal on the documented grid - re-run the scan "
        "and re-verify before any merge")
    assert ndiff / len(f0s) < 1e-3, f"divergence rate too high: {ndiff}"