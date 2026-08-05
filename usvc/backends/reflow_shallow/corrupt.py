"""Mel corruption utilities for reflow_shallow training.

Simulates the artefacts present in sovits/rvc output mels (electronic timbre,
spectral dropout, high-frequency structural distortion) so that the reflow
shallow-diffusion model trains on (corrupted_mel → clean_mel) pairs that match
the inference-time distribution (sovits/rvc output mel → clean mel).

All transforms operate on the normalised log-mel domain (spec_min=-12,
spec_max=2 → normalised [-1, 1], typical mel values in [-12, 2]).
"""
from __future__ import annotations

import torch

__all__ = ["corrupt_mel"]


def corrupt_mel(
    mel: torch.Tensor,
    strength_range: tuple[float, float] = (0.3, 1.0),
) -> torch.Tensor:
    """Apply random corruption to a log-mel spectrogram.

    Args:
        mel: [B, T, M] or [T, M] log-mel (dB domain, spec_min=-12..spec_max=2).
        strength_range: random strength factor sampled per-batch; 0 = identity,
            1 = strong corruption.

    Returns:
        Corrupted mel, same shape as input.

    Corruption strategies (each applied with ~0.5 probability, composed):
      1. Additive Gaussian noise (background hiss / quantisation).
      2. Random frequency-band masking (spectral dropout → electronic timbre).
      3. High-frequency harmonic jitter (structural HF distortion).
    """
    squeeze = mel.dim() == 2
    if squeeze:
        mel = mel.unsqueeze(0)  # [1, T, M]
    B, T, M = mel.shape
    device = mel.device

    # per-batch strength
    s_lo, s_hi = strength_range
    strength = torch.empty(B, device=device).uniform_(s_lo, s_hi)
    s = strength.view(B, 1, 1)

    out = mel.clone()
    # mel is in dB domain (~[-12, 2]); noise magnitude scales with strength.
    noise_scale_db = 0.15  # max ~0.15 dB noise at full strength

    # 1) Gaussian noise (applied always, scaled by strength)
    noise = torch.randn_like(out) * (noise_scale_db * s)
    out = out + noise

    # 2) Random frequency band masking (per-sample, ~0.5 prob)
    #    Zero out 1-3 random mel bands of width 2-8 bins (push toward mel.min()-0.5).
    mask_floor = float(mel.min().item()) - 0.5
    do_mask = torch.rand(B, device=device) < 0.5
    if do_mask.any():
        idx = torch.where(do_mask)[0]
        for i in idx.tolist():
            n_bands = int(torch.randint(1, 4, (1,)).item())  # 1-3 bands
            for _ in range(n_bands):
                bw = int(torch.randint(2, 9, (1,)).item())   # width 2-8
                start = int(torch.randint(0, max(M - bw, 1), (1,)).item())
                out[i, :, start:start + bw] = mask_floor

    # 3) High-frequency harmonic jitter (per-sample, ~0.5 prob)
    #    Add structured noise to the upper half of the spectrum.
    hf_start = M // 2
    do_jitter = torch.rand(B, device=device) < 0.5
    if do_jitter.any() and hf_start < M:
        idx = torch.where(do_jitter)[0]
        hf = out[idx, :, hf_start:]
        # structured noise: smooth random pattern (diffuse, not pointwise)
        raw = torch.randn(hf.shape, device=device)
        # light temporal smoothing via 1D avg pool on time axis
        if hf.shape[1] > 4:
            raw = torch.nn.functional.avg_pool1d(
                raw.transpose(1, 2), kernel_size=3, stride=1, padding=1
            ).transpose(1, 2)
        jitter = raw * (0.3 * s[idx].view(-1, 1, 1))
        out[idx, :, hf_start:] = hf + jitter

    if squeeze:
        out = out.squeeze(0)
    return out
