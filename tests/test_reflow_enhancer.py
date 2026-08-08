"""Tests for the ReflowEnhancer (DDSP6 RectifiedFlow shallow diffusion).

Covers:
  - enhancer construction from a DDSP6 base model
  - enhance() on synthetic inputs (CPU + optional CUDA)
  - t_start semantics (0 ≈ full generation, →1 ≈ keep gt)
  - sample-rate bridge via the InferenceEngine helper functions

These tests use the DDSP6 base models under the pretrained dir (USVC_PRETRAINED_DIR
or <repo>/pretrained) if available; otherwise they are skipped (environment-dependent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from usvc.hub import pretrained_dir  # noqa: E402

DDSP6_CKPT = pretrained_dir() / "base_models" / "ddsp6" / "6.3_6x512_6x768" / "model.pt"
DDSP6_CFG = pretrained_dir() / "base_models" / "ddsp6" / "6.3_6x512_6x768" / "config.yaml"
VOCODER_DIR = pretrained_dir() / "nsf_hifigan"

_SKIP = not (DDSP6_CKPT.is_file() and (VOCODER_DIR / "model.ckpt").is_file())

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
env = pytest.mark.skipif(_SKIP, reason="DDSP6 base model / vocoder not present")


@env
def test_reflow_enhancer_constructs_cpu():
    """ReflowEnhancer loads a DDSP6 base model on CPU."""
    from usvc.infer.reflow_enhancer import ReflowEnhancer

    enh = ReflowEnhancer(
        reflow_ckpt=DDSP6_CKPT, reflow_config=DDSP6_CFG,
        vocoder_dir=VOCODER_DIR, device="cpu", half=False,
    )
    assert enh.n_units == 768
    assert enh.sr == 44100
    assert enh.hop == 512


@env
@cuda
def test_reflow_enhance_basic_cuda():
    """enhance() returns a finite float32 wav array of the right shape."""
    from usvc.infer.reflow_enhancer import ReflowEnhancer

    enh = ReflowEnhancer(
        reflow_ckpt=DDSP6_CKPT, reflow_config=DDSP6_CFG,
        vocoder_dir=VOCODER_DIR, device="cuda", half=False,
    )
    T = 80
    units = torch.randn(T, 768, device="cuda")
    f0 = torch.full((T,), 220.0, device="cuda")
    volume = torch.full((T,), 0.3, device="cuda")
    gen_wav = torch.randn(T * 512, device="cuda") * 0.05

    out = enh.enhance(units, f0, volume, spk_id=0, gen_wav=gen_wav,
                      t_start=0.5, infer_step=10)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    # output length ≈ T * hop (may differ by a few frames from STFT rounding)
    assert abs(out.shape[0] - T * 512) < 3 * 512


@env
@cuda
def test_t_start_semantics():
    """t_start=0 (full gen) and t_start→1 (keep gt) produce different output."""
    from usvc.infer.reflow_enhancer import ReflowEnhancer

    enh = ReflowEnhancer(
        reflow_ckpt=DDSP6_CKPT, reflow_config=DDSP6_CFG,
        vocoder_dir=VOCODER_DIR, device="cuda", half=False,
    )
    torch.manual_seed(42)
    T = 60
    units = torch.randn(T, 768, device="cuda")
    f0 = torch.full((T,), 220.0, device="cuda")
    volume = torch.full((T,), 0.3, device="cuda")
    gen_wav = torch.randn(T * 512, device="cuda") * 0.05

    torch.manual_seed(0)
    out_gen = enh.enhance(units, f0, volume, spk_id=0, gen_wav=gen_wav, t_start=0.0)
    torch.manual_seed(0)
    out_gt = enh.enhance(units, f0, volume, spk_id=0, gen_wav=gen_wav, t_start=0.9)
    # the two regimes should differ (full generation vs. near-identity)
    assert np.abs(out_gen - out_gt).max() > 1e-3


@env
@cuda
def test_gen_wav_affects_output():
    """The enhancer output must depend on gen_wav (regression for the
    'gt_spec ignored when mel_extractor present' bug)."""
    from usvc.infer.reflow_enhancer import ReflowEnhancer

    enh = ReflowEnhancer(
        reflow_ckpt=DDSP6_CKPT, reflow_config=DDSP6_CFG,
        vocoder_dir=VOCODER_DIR, device="cuda", half=False,
    )
    T = 80
    units = torch.randn(T, 768, device="cuda")
    f0 = torch.full((T,), 220.0, device="cuda")
    volume = torch.full((T,), 0.3, device="cuda")

    torch.manual_seed(0)
    out_noise = enh.enhance(units, f0, volume, spk_id=0,
                            gen_wav=torch.randn(T * 512, device="cuda") * 0.1,
                            t_start=0.5)
    torch.manual_seed(0)
    out_silent = enh.enhance(units, f0, volume, spk_id=0,
                             gen_wav=torch.zeros(T * 512, device="cuda"),
                             t_start=0.5)
    # the two must differ: gen_wav genuinely reaches the reflow target
    assert np.abs(out_noise - out_silent).max() > 0.01


def test_resample_helpers():
    """_resample_wav and _resample_f0 are deterministic and shape-preserving."""
    from usvc.infer.engine import _resample_f0, _resample_wav

    # wav: 48k -> 44.1k round-trip preserves length within ~1 sample
    wav = np.random.randn(48000).astype(np.float32) * 0.1
    down = _resample_wav(wav, 48000, 44100)
    assert down.shape[0] == int(48000 * 44100 / 48000)
    up = _resample_wav(down, 44100, 48000)
    assert abs(up.shape[0] - 48000) < 3

    # f0: [T] -> [T'] linear interpolation
    f0 = torch.linspace(100, 200, 100)
    out = _resample_f0(f0, 200)
    assert out.shape[0] == 200
    assert out[0].item() == pytest.approx(100, abs=1)
    assert out[-1].item() == pytest.approx(200, abs=1)

    # units: [T, D] -> [T', D] preserves feature dimension
    from usvc.infer.engine import _resample_units
    units = torch.randn(100, 768)
    out_u = _resample_units(units, 200)
    assert out_u.shape == (200, 768)
    # endpoints match the source trajectory
    assert torch.allclose(out_u[0], units[0], atol=1e-4)
    assert torch.allclose(out_u[-1], units[-1], atol=1e-4)
