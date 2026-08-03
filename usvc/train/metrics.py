"""Validation metrics shared across all backends.

Implements the standard SVC evaluation metrics used by RIFT, DDSP, Diffusion-SVC:
MCD (Mel Cepstral Distortion), SI-SNR, PSNR, MSE.
All operate on mel spectrograms [B, T, M] or [T, M].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["mcd", "si_snr", "psnr", "mse", "compute_all_metrics"]


def mcd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mel Cepstral Distortion (per-sample scalar).

    Args:
        pred, target: [T, M] or [B, T, M] mel spectrograms.
    Returns:
        scalar (mean over batch) if batched, else scalar.
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    # MCD in dB, using first 25 cepstral coeffs is standard, but all-mel is
    # also common. We use all mel bins (matches RIFT-SVC's mcd implementation).
    diff = pred - target
    # (10*sqrt(2))/ln10 ≈ 6.1411 — standard MCD scaling factor
    mcd_val = 6.1411 * torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=-1))
    if squeeze:
        return mcd_val.squeeze(0)
    return mcd_val.mean()


def si_snr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant Signal-to-Noise Ratio (higher is better).

    Args:
        pred, target: [T, M] or [B, T, M].
    Returns:
        SI-SNR in dB (mean over batch).
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    # Flatten time × mel for SI-SNR computation
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    # Zero-mean
    pred_flat = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
    target_flat = target_flat - target_flat.mean(dim=-1, keepdim=True)

    # SI-SNR
    eps = 1e-8
    dot = (pred_flat * target_flat).sum(dim=-1, keepdim=True)
    energy = (target_flat ** 2).sum(dim=-1, keepdim=True) + eps
    scale = dot / energy
    s_target = scale * target_flat
    e_noise = pred_flat - s_target

    si_snr_val = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    if squeeze:
        return si_snr_val.squeeze(0)
    return si_snr_val.mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio for mel spectrograms (higher is better).

    Uses max(target) as peak (standard for mel PSNR in SVC).
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    mse_val = (pred - target) ** 2
    mse_per_sample = mse_val.mean(dim=(-2, -1))
    max_val = target.amax(dim=(-2, -1))
    # Avoid log(0)
    eps = 1e-8
    psnr_val = 10 * torch.log10((max_val + eps) ** 2 / (mse_per_sample + eps))
    if squeeze:
        return psnr_val.squeeze(0)
    return psnr_val.mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE."""
    return F.mse_loss(pred, target)


def compute_all_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute all four metrics at once. Returns dict of scalars."""
    return {
        "mcd": mcd(pred, target),
        "si_snr": si_snr(pred, target),
        "psnr": psnr(pred, target),
        "mse": mse(pred, target),
    }
