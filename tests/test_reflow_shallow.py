"""Tests for the reflow_shallow backend v2 (DDPM-equivalent shallow enhancer).

v2: condition = units/f0/vol embeddings (mel-independent), so gt_spec
(sovits_mel) genuinely participates in the ODE trajectory at inference.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


def test_model_forward_loss_has_gradient():
    """forward_loss returns a scalar with a gradient (mel-independent cond)."""
    from usvc.backends.reflow_shallow.model import ReflowShallowModel

    model = ReflowShallowModel(out_dims=128, n_layers=2, n_chans=128, n_units=768, n_spk=1, n_hidden=64)
    B, T = 2, 40
    units = torch.randn(B, T, 768)
    f0 = torch.full((B, T), 200.0)
    rms = torch.full((B, T), 0.3)
    real = torch.randn(B, T, 128)
    loss, sub = model.forward_loss(units, f0, rms, real)
    assert loss.dim() == 0
    assert "reflow_loss" in sub
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


def test_model_sample_finite():
    """sample returns a finite [B, T, M] mel."""
    from usvc.backends.reflow_shallow.model import ReflowShallowModel

    model = ReflowShallowModel(out_dims=128, n_layers=2, n_chans=128, n_units=768, n_spk=1, n_hidden=64).eval()
    B, T = 1, 40
    units = torch.randn(B, T, 768)
    f0 = torch.full((B, T), 220.0)
    rms = torch.full((B, T), 0.3)
    mel = torch.randn(B, T, 128) * 3
    with torch.no_grad():
        out = model.sample(units, f0, rms, mel, t_start=0.5, infer_step=5)
    assert out.shape == (B, T, 128)
    assert torch.isfinite(out).all()


def test_condition_is_mel_independent():
    """At t_start≈0 (pure generation from noise), changing gt_spec should
    have limited effect on the output, since the ODE starts from pure noise
    and only the mel-independent condition guides it.

    Note: with random weights the absolute difference is not meaningful, so
    we only check the output shape and finiteness here.  The real
    mel-independence check is an integration test after training."""
    from usvc.backends.reflow_shallow.model import ReflowShallowModel

    model = ReflowShallowModel(out_dims=128, n_layers=2, n_chans=128, n_units=768, n_spk=1, n_hidden=64).eval()
    B, T = 1, 40
    units = torch.randn(B, T, 768)
    f0 = torch.full((B, T), 220.0)
    rms = torch.full((B, T), 0.3)
    mel = torch.randn(B, T, 128) * 3
    with torch.no_grad():
        out = model.sample(units, f0, rms, mel, t_start=0.0, infer_step=5)
    assert out.shape == (B, T, 128)
    assert torch.isfinite(out).all()


ddsp6 = "/home/waas/pretrained/base_models/ddsp6/6.3_6x512_6x768/model.pt"
vocoder_dir = "pretrained/nsf_hifigan"
_skip_env = not (
    __import__("os").path.isfile(ddsp6)
    and __import__("os").path.isfile(__import__("os").path.join(vocoder_dir, "model.ckpt"))
)


@pytest.mark.skipif(_skip_env, reason="DDSP6 base / vocoder not present")
def test_ddsp6_migration():
    """DDSP6 velocity_fn residual weights load (conditioner_projection random)."""
    from usvc.backends.reflow_shallow.model import ReflowShallowModel

    model = ReflowShallowModel(out_dims=128, n_layers=6, n_chans=768, n_units=768, n_spk=1, n_hidden=256)
    report = model.load_ddsp6_reflow(ddsp6)
    assert report["missing_velocity"] == 0


@pytest.mark.skipif(_skip_env or not torch.cuda.is_available(),
                    reason="DDSP6 base + CUDA required")
def test_shallow_enhancer_gen_wav_dependent():
    """The v2 enhancer output must depend on gen_wav (sovits mel = gt_spec)."""
    from usvc.infer.reflow_enhancer import ReflowEnhancer

    enh = ReflowEnhancer(
        reflow_ckpt=ddsp6, reflow_config="configs/nahida_reflow_shallow.yaml",
        vocoder_dir=vocoder_dir, device="cuda", half=False,
    )
    assert enh._mode == "shallow"
    T = 80
    units = torch.randn(T, 768, device="cuda")
    f0 = torch.full((T,), 220.0, device="cuda")
    vol = torch.full((T,), 0.3, device="cuda")

    torch.manual_seed(0)
    a = enh.enhance(units, f0, vol, 0,
                    gen_wav=torch.randn(T * 512, device="cuda") * 0.1, t_start=0.5)
    torch.manual_seed(0)
    b = enh.enhance(units, f0, vol, 0,
                    gen_wav=torch.zeros(T * 512, device="cuda"), t_start=0.5)
    assert np.abs(a - b).max() > 0.01
