"""GAN and VITS loss primitives.

Shared by so-vits and RVC backends (both use VITS-style GAN training).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "lsgan_disc_loss",
    "lsgan_gen_loss",
    "feature_match_loss",
    "kl_loss",
]


def lsgan_disc_loss(
    real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]
) -> torch.Tensor:
    """LSGAN discriminator loss.

    LSGAN uses least-squares: D wants Dr→1, Df→0.
        loss = mean((Dr - 1)^2) + mean(Df^2)

    Args:
        real_logits: list of [B, ...] discriminator outputs on real samples.
        fake_logits: list of [B, ...] discriminator outputs on fake samples.
    """
    loss = 0.0
    for dr, df in zip(real_logits, fake_logits):
        loss = loss + F.mse_loss(dr, torch.ones_like(dr))
        loss = loss + F.mse_loss(df, torch.zeros_like(df))
    return loss / len(real_logits)


def lsgan_gen_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator loss. G wants Df→1.

        loss = mean((Df - 1)^2)
    """
    loss = 0.0
    for df in fake_logits:
        loss = loss + F.mse_loss(df, torch.ones_like(df))
    return loss / len(fake_logits)


def feature_match_loss(
    real_fmaps: list[list[torch.Tensor]],
    fake_fmaps: list[list[torch.Tensor]],
    weight: float = 2.0,
) -> torch.Tensor:
    """Feature matching loss (L1 between intermediate D features).

    Args:
        real_fmaps: list (per discriminator) of list (per layer) of [B, ...].
        fake_fmaps: same structure.
        weight: scalar multiplier (sovits/RVC use 2.0).
    """
    loss = 0.0
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + F.l1_loss(f, r.detach())
    return weight * loss


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL divergence loss for VITS posterior → prior.

    Uses the so-vits-svc / VITS simplified KL formula:
        kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p)^2 * exp(-2*logs_p)
        loss = sum(kl * mask) / sum(mask)   [normalized by frame count]

    This differs slightly from the full VITS KL in the treatment of the
    posterior variance term (so-vits drops the (logs_p - logs_q)^2 component
    and normalizes by mask sum instead of mean).

    Args:
        z_p: [B, T, C] flow-transformed posterior sample.
        logs_q: [B, T, C] posterior log-std.
        m_p: [B, T, C] prior mean.
        logs_p: [B, T, C] prior log-std.
        z_mask: [B, 1, T] optional mask.
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2 * logs_p)
    if z_mask is not None:
        kl = kl * z_mask
        return kl.sum() / (z_mask.sum() * kl.shape[-1] + 1e-8)
    return kl.mean()
