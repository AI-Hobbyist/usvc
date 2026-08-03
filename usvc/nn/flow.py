"""Rectified Flow / Flow Matching samplers.

Shared by RIFT (DiT velocity network) and DDSP6.3 (LYNXNet2 velocity network).
"""
from __future__ import annotations

import torch

__all__ = ["euler_sample", "rk4_sample"]


def euler_sample(
    velocity_fn,
    x0: torch.Tensor,
    cond: dict | torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """Euler ODE integration from t=0 (noise) to t=1 (clean).

    velocity_fn(x, t, cond) -> v, where dx/dt = v.
    t ∈ [0, 1]: x(0) = x0 (noise), x(1) ≈ clean sample.
    """
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        v = velocity_fn(x, t, cond)
        x = x + dt * v
    return x


def rk4_sample(
    velocity_fn,
    x0: torch.Tensor,
    cond: dict | torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """RK4 ODE integration from t=0 to t=1. More accurate than Euler."""
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t_val = i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)

        k1 = velocity_fn(x, t, cond)
        k2 = velocity_fn(
            x + 0.5 * dt * k1,
            torch.full((x.shape[0],), t_val + 0.5 * dt, device=x.device, dtype=x.dtype),
            cond,
        )
        k3 = velocity_fn(
            x + 0.5 * dt * k2,
            torch.full((x.shape[0],), t_val + 0.5 * dt, device=x.device, dtype=x.dtype),
            cond,
        )
        k4 = velocity_fn(
            x + dt * k3,
            torch.full((x.shape[0],), t_val + dt, device=x.device, dtype=x.dtype),
            cond,
        )
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
