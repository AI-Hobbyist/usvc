"""Diffusion beta schedules and alpha computation.

Used by DDPMObjective. Shared by Diffusion-SVC and so-vits diffusion branch
(both use linear schedule: linspace(1e-4, 0.02, 1000)).
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "betas_for_alpha_bar",
    "compute_alphas",
]


def linear_beta_schedule(timesteps: int, max_beta: float = 0.02) -> torch.Tensor:
    """Linear beta schedule: linspace(1e-4, max_beta, timesteps).

    This is the schedule used by Diffusion-SVC and so-vits-svc.
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return torch.from_numpy(betas).float()


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal 2021). Not used by default."""
    x = np.arange(timesteps + 1)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.from_numpy(np.clip(betas, 0, 0.999)).float()


def betas_for_alpha_bar(
    num_diffusion_timesteps: int, alpha_bar: float, max_beta: float = 0.999
) -> torch.Tensor:
    """Create a beta schedule that approximates a given cumulative alpha."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def compute_alphas(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    """Precompute all alpha-related buffers for DDPM.

    Returns dict with:
        alphas:          1 - betas
        alphas_cumprod:  cumulative product of alphas
        alphas_cumprod_prev: shifted by one (prepended 1.0)
        sqrt_*
        posterior_*
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0]), alphas_cumprod[:-1]]
    )

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.rsqrt(alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.rsqrt(1.0 - alphas_cumprod)

    # Posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # Clip for numerical stability (first step)
    posterior_log_variance_clipped = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    )
    posterior_mean_coef1 = (
        betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas)
        / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }
