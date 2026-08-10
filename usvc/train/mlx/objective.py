"""MLX 训练目标函数:FlowMatchingObjective / DDPMObjective。

镜像 usvc/train/objective.py 的数学逻辑,张量层用 mlx 实现,
供 mlx 训练循环使用。输入均为 mlx array(或可转 mlx 的 numpy)。
"""
from __future__ import annotations

import math
from typing import Callable, Literal

import mlx.core as mx

__all__ = [
    "FlowMatchingObjective",
    "DDPMObjective",
    "sample_time_lognorm",
    "sample_time_uniform",
    "lens_to_mask",
]


def lens_to_mask(t: mx.array, length: int | None = None) -> mx.array:
    """[b] lengths -> [b, n] boolean mask (True = valid frame)。"""
    if length is None:
        length = int(t.max().item()) if t.size else 0
    seq = mx.arange(length, dtype=mx.int32)
    t = t.astype(mx.int32)
    return seq[None, :] < t[:, None]


def sample_time_lognorm(size: int, dtype=mx.float32) -> mx.array:
    """Stratified lognorm time sampling (RIFT-SVC rf.py:20-27)。

    z = erfinv(2*(quantile + rand/size) - 1) * sqrt(2); t = sigmoid(z)
    """
    quantiles = mx.linspace(0, 1, size + 1, dtype=dtype)
    z = quantiles[:-1] + mx.random.uniform(shape=(size,), dtype=dtype) / size
    z = mx.erfinv(2 * z - 1) * math.sqrt(2)
    return mx.sigmoid(z)


def sample_time_uniform(size: int, dtype=mx.float32) -> mx.array:
    return mx.random.uniform(shape=(size,), dtype=dtype)


class FlowMatchingObjective:
    """Rectified Flow / Flow Matching velocity loss (MLX 版)。

    数学核心(与 torch 版完全一致):
        x1 = norm(mel)
        x0 = randn_like(x1)
        xt = (1-t)*x0 + t*x1
        target = x1 - x0
        loss = MSE(velocity_fn(xt, t, cond), target)
    """

    def __init__(
        self,
        spec_min: float = -12.0,
        spec_max: float = 2.0,
        time_schedule: Literal["lognorm", "uniform"] = "lognorm",
        loss_type: Literal["l2", "l2_lognorm"] = "l2",
        dtype=mx.float32,
    ):
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.time_schedule = time_schedule
        self.loss_type = loss_type
        self.dtype = dtype

    def norm(self, x: mx.array) -> mx.array:
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def sample_time(self, size: int) -> mx.array:
        if self.time_schedule == "lognorm":
            return sample_time_lognorm(size, self.dtype)
        return sample_time_uniform(size, self.dtype)

    def compute_loss(
        self,
        x1_mel: mx.array,
        velocity_fn: Callable,
        cond=None,
        mask: mx.array | None = None,
    ) -> mx.array:
        x1 = self.norm(x1_mel.astype(self.dtype))
        x0 = mx.random.normal(x1.shape, dtype=self.dtype)
        batch = x1.shape[0]
        t = self.sample_time(batch)

        while t.ndim < x1.ndim:
            t = t[..., None]
        xt = (1 - t) * x0 + t * x1
        target = x1 - x0

        pred = velocity_fn(xt, t.squeeze(), cond)

        if self.loss_type == "l2":
            loss = (pred - target) ** 2
        else:
            # l2_lognorm: weight = 0.398942 / t / (1-t) * exp(-0.5 * log(t/(1-t))^2)
            t_1d = t.reshape(-1)
            w = (
                0.398942
                / t_1d
                / (1 - t_1d)
                * mx.exp(-0.5 * mx.log(t_1d / (1 - t_1d)) ** 2)
            )
            mse = (pred - target) ** 2
            while w.ndim < mse.ndim:
                w = w[..., None]
            loss = w * mse

        if mask is not None and loss.ndim >= 2 and mask.shape[1] == loss.shape[1]:
            loss = loss * mask[..., None].astype(self.dtype)
            return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)

        return loss.mean()


class DDPMObjective:
    """DDPM epsilon-prediction MSE loss (MLX 版)。

    数学核心(与 torch 版一致):
        betas = linspace(1e-4, 0.02, 1000)
        x0 = norm(mel)
        t ~ randint(0, 1000)
        noise = randn_like(x0)
        xt = sqrt(acp[t]) * x0 + sqrt(1-acp[t]) * noise
        loss = MSE(eps_pred, noise)
    """

    def __init__(
        self,
        spec_min: float = -12.0,
        spec_max: float = 2.0,
        timesteps: int = 1000,
        max_beta: float = 0.02,
        loss_type: Literal["l2", "l1"] = "l2",
        dtype=mx.float32,
    ):
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.dtype = dtype
        # 与 torch 版一致的线性 beta schedule
        import numpy as np
        betas = np.linspace(1e-4, max_beta, timesteps).astype(np.float32)
        alphas = 1.0 - betas
        acp = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = mx.array(np.sqrt(acp), dtype=dtype)
        self.sqrt_one_minus_alphas_cumprod = mx.array(np.sqrt(1.0 - acp), dtype=dtype)

    def norm(self, x: mx.array) -> mx.array:
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def compute_loss(
        self,
        x0_mel: mx.array,
        denoise_fn: Callable,
        cond=None,
        t_max: int | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        x0 = self.norm(x0_mel.astype(self.dtype))
        batch = x0.shape[0]

        t_hi = t_max if t_max is not None else self.timesteps
        t = mx.random.randint(0, t_hi, shape=(batch,))
        noise = mx.random.normal(x0.shape, dtype=self.dtype)

        sqrt_acp = self.sqrt_alphas_cumprod[t]
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_acp.ndim < x0.ndim:
            sqrt_acp = sqrt_acp[..., None]
            sqrt_omacp = sqrt_omacp[..., None]

        xt = sqrt_acp * x0 + sqrt_omacp * noise
        eps_pred = denoise_fn(xt, t, cond)

        if mask is not None and mask.shape[1] == x0.shape[1]:
            elem = (eps_pred - noise) ** 2
            elem = elem * mask[..., None].astype(self.dtype)
            return elem.sum() / (mask.sum() * elem.shape[-1] + 1e-8)
        if self.loss_type == "l1":
            return mx.abs(eps_pred - noise).mean()
        return ((eps_pred - noise) ** 2).mean()
