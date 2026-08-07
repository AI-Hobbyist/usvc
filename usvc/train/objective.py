"""Shared mathematical training objectives.

FlowMatchingObjective — shared by rift_dit and ddsp6 backends.
    Both use rectified flow: x0=randn, xt=(1-t)*x0+t*x1, target=x1-x0.
    Mathematical identity confirmed between RIFT-SVC (rf.py:210-233) and
    DDSP-SVC-6.3 (reflow.py:20-35).

DDPMObjective — shared by diffusion and sovits (diffusion branch) backends.
    Both use DDPM epsilon-prediction with linear beta schedule.
    so-vits diffusion/ is derived from Diffusion-SVC (near-identical code).
"""
from __future__ import annotations

import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from ..nn.normalization import MelNormalizer
from ..nn.schedule import linear_beta_schedule, compute_alphas

__all__ = ["FlowMatchingObjective", "DDPMObjective", "sample_time_lognorm", "lens_to_mask"]


def lens_to_mask(t: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """[b] lengths -> [b, n] boolean mask (True = valid frame).

    Shared by all backends that mask padding frames in the loss (RIFT-SVC
    rf.py convention ``loss[mask]``). Kept here so flow / DDPM / reflow
    objectives can all consume the same mask without each backend rolling
    its own.
    """
    if length is None:
        length = int(t.amax().item()) if t.numel() else 0
    seq = torch.arange(length, device=t.device)
    return seq < t[..., None]


def sample_time_lognorm(size: int, device: torch.device) -> torch.Tensor:
    """Stratified lognorm time sampling (RIFT-SVC rf.py:20-27).

    Stratified sampling of normals → sigmoid → t ∈ (0,1).
    Concentrates samples near t≈0.5 where the flow is hardest.
    """
    quantiles = torch.linspace(0, 1, size + 1, device=device)
    z = quantiles[:-1] + torch.rand((size,), device=device) / size
    z = torch.erfinv(2 * z - 1) * math.sqrt(2)
    return torch.sigmoid(z)


def sample_time_uniform(size: int, device: torch.device) -> torch.Tensor:
    """Uniform time sampling on [0, 1)."""
    return torch.rand((size,), device=device)


class FlowMatchingObjective:
    """Rectified Flow / Flow Matching velocity loss.

    Shared by:
      - rift_dit (DiT velocity network): t passed as-is, mel is [B,T,M]
      - ddsp6    (LYNXNet2 velocity network): t scaled ×1000, mel is [B,1,M,T]

    The velocity_fn signature differs between the two backends, so the caller
    (backend module) wraps their model into a uniform velocity_fn(t, x, cond).

    Mathematical core (identical to both RIFT rf.py and DDSP reflow.py):
        x1 = norm(mel)
        x0 = randn_like(x1)
        xt = (1-t)*x0 + t*x1
        target = x1 - x0
        loss = MSE(velocity_fn(xt, t, cond), target)

    Args:
        normalizer: MelNormalizer (spec_min=-12, spec_max=2)
        time_schedule: 'lognorm' (RIFT default, stratified) or 'uniform'
        loss_type: 'l2' (plain MSE) or 'l2_lognorm' (DDSP default, lognorm weight)
    """

    def __init__(
        self,
        normalizer: MelNormalizer,
        time_schedule: Literal["lognorm", "uniform"] = "lognorm",
        loss_type: Literal["l2", "l2_lognorm"] = "l2",
    ):
        self.norm = normalizer
        self.time_schedule = time_schedule
        self.loss_type = loss_type

    def sample_time(self, size: int, device: torch.device) -> torch.Tensor:
        if self.time_schedule == "lognorm":
            return sample_time_lognorm(size, device)
        return sample_time_uniform(size, device)

    def compute_loss(
        self,
        x1_mel: torch.Tensor,
        velocity_fn: Callable,
        cond: dict | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow-matching velocity MSE loss.

        Args:
            x1_mel: [B, T, M] or [B, 1, M, T] target mel (raw, un-normalized).
            velocity_fn: callable(xt, t, cond) → velocity prediction, same shape as xt.
            cond: conditioning passed to velocity_fn (dict or tensor).
            mask: [B, T] boolean mask for valid frames (None = all valid).
                  Only applied to [B, T, M] layout (rift_dit).

        Returns:
            scalar loss tensor.
        """
        x1 = self.norm.norm(x1_mel)
        x0 = torch.randn_like(x1)
        batch = x1.shape[0]
        t = self.sample_time(batch, x1.device)

        # Broadcast t to match x1's shape
        while t.ndim < x1.ndim:
            t = t.unsqueeze(-1)
        xt = (1 - t) * x0 + t * x1
        target = x1 - x0

        pred = velocity_fn(xt, t.squeeze(), cond)

        if self.loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        else:
            # l2_lognorm: weight = 0.398942 / t / (1-t) * exp(-0.5 * log(t/(1-t))^2)
            # The weight MUST use the SAME t used to build xt and feed the
            # velocity net, otherwise E_{t'~p}[w(t')]·E_t[L(t)] = E_t[L(t)]
            # (since ∫₀¹ w dt = 1 for the logit-normal density) and the lognorm
            # time emphasis collapses to plain L2. This matches ReFlow-VAE-SVC /
            # DDSP-SVC-6.3 / the usvc ddsp6 RectifiedFlow.reflow_loss verbatim port.
            t_1d = t.flatten()
            w = (
                0.398942
                / t_1d
                / (1 - t_1d)
                * torch.exp(-0.5 * torch.log(t_1d / (1 - t_1d)) ** 2)
            )
            # Broadcast w to match the elementwise MSE shape, then weight it.
            mse = F.mse_loss(pred, target, reduction="none")
            while w.ndim < mse.ndim:
                w = w.unsqueeze(-1)
            loss = w * mse

        if mask is not None and loss.ndim >= 2 and mask.shape[1] == loss.shape[1]:
            # Apply per-frame mask (rift_dit [B,T,M] layout)
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)

        return loss.mean()


class DDPMObjective:
    """DDPM epsilon-prediction MSE loss.

    Shared by:
      - diffusion backend (Diffusion-SVC style): WaveNet/ConvNext denoiser
      - sovits diffusion branch: same code, near-identical

    Mathematical core (Diffusion-SVC diffusion.py:194-210, so-vits same):
        betas = linspace(1e-4, 0.02, 1000)  # linear
        x0 = norm(mel)
        t ~ randint(0, 1000)
        noise = randn_like(x0)
        xt = sqrt(acp[t]) * x0 + sqrt(1-acp[t]) * noise
        eps_pred = denoise_fn(xt, t, cond)
        loss = MSE(eps_pred, noise)

    Args:
        normalizer: MelNormalizer
        timesteps: 1000 (standard)
        max_beta: 0.02 (linear schedule upper bound)
    """

    def __init__(
        self,
        normalizer: MelNormalizer,
        timesteps: int = 1000,
        max_beta: float = 0.02,
        loss_type: Literal["l2", "l1"] = "l2",
    ):
        self.norm = normalizer
        self.timesteps = timesteps
        self.loss_type = loss_type
        betas = linear_beta_schedule(timesteps, max_beta=max_beta)
        alphas = compute_alphas(betas)
        # Store as buffers (caller should register them on a module if needed)
        self.sqrt_alphas_cumprod = alphas["sqrt_alphas_cumprod"]
        self.sqrt_one_minus_alphas_cumprod = alphas["sqrt_one_minus_alphas_cumprod"]

    def register_buffers(self, module: torch.nn.Module) -> None:
        """Register alpha buffers onto a nn.Module so they move with .to(device).

        Call this from a LightningModule.__init__ if you want device-aware buffers.
        """
        module.register_buffer(
            "sqrt_alphas_cumprod", self.sqrt_alphas_cumprod, persistent=False
        )
        module.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            self.sqrt_one_minus_alphas_cumprod,
            persistent=False,
        )

    def compute_loss(
        self,
        x0_mel: torch.Tensor,
        denoise_fn: Callable,
        cond: dict | torch.Tensor | None = None,
        t_max: int | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute epsilon-prediction MSE loss.

        Args:
            x0_mel: [B, T, M] target mel (raw).
            denoise_fn: callable(xt, t, cond) → eps prediction.
            cond: conditioning.
            t_max: upper bound (exclusive) for the random timestep. None →
                self.timesteps. so-vits shallow diffusion trains with
                t = randint(0, k_step_max) where k_step_max < 1000.
            mask: optional [B, T] boolean/float frame mask (1 = real frame,
                0 = padding). When given, the loss is averaged only over real
                frames (padding has mel=0 → norm(+0.714), a loud non-silence
                target that pollutes the denoiser gradient). Matches RIFT-SVC
                rf.py ``loss[mask]``.

        Returns:
            scalar loss tensor.
        """
        x0 = self.norm.norm(x0_mel)
        batch = x0.shape[0]
        device = x0.device

        t_hi = t_max if t_max is not None else self.timesteps
        t = torch.randint(0, t_hi, (batch,), device=device)
        noise = torch.randn_like(x0)

        sqrt_acp = self.sqrt_alphas_cumprod.to(device)[t]
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod.to(device)[t]

        # Broadcast to x0's shape
        while sqrt_acp.ndim < x0.ndim:
            sqrt_acp = sqrt_acp.unsqueeze(-1)
            sqrt_omacp = sqrt_omacp.unsqueeze(-1)

        xt = sqrt_acp * x0 + sqrt_omacp * noise
        eps_pred = denoise_fn(xt, t, cond)

        if mask is not None and mask.shape[1] == x0.shape[1]:
            elem = F.mse_loss(eps_pred, noise, reduction="none")
            elem = elem * mask.unsqueeze(-1)
            return elem.sum() / (mask.sum() * elem.shape[-1] + 1e-8)
        if self.loss_type == "l1":
            return F.l1_loss(eps_pred, noise)
        return F.mse_loss(eps_pred, noise)
