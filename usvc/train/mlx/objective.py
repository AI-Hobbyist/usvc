"""MLX 训练目标:FlowMatchingObjective / DDPMObjective(镜像 usvc/train/objective.py)。"""
from __future__ import annotations
import math
from typing import Callable, Literal
import mlx.core as mx
__all__ = ["FlowMatchingObjective", "DDPMObjective", "sample_time_lognorm",
           "sample_time_uniform", "lens_to_mask"]
def lens_to_mask(t: mx.array, length: int | None = None) -> mx.array:  # [b]→[b,n] bool(True=有效帧)
    if length is None:
        length = int(t.max().item()) if t.size else 0
    return mx.arange(length, dtype=mx.int32)[None, :] < t.astype(mx.int32)[:, None]
def sample_time_lognorm(size: int, dtype=mx.float32) -> mx.array:
    """分层 lognorm 时间采样(RIFT rf.py:20-27):z=erfinv(2*(q+r/size)-1)*√2;t=sigmoid(z)。"""
    quantiles = mx.linspace(0, 1, size + 1, dtype=dtype)
    z = quantiles[:-1] + mx.random.uniform(shape=(size,), dtype=dtype) / size
    return mx.sigmoid(mx.erfinv(2 * z - 1) * math.sqrt(2))
def sample_time_uniform(size: int, dtype=mx.float32) -> mx.array:
    return mx.random.uniform(shape=(size,), dtype=dtype)
def _expand(t, ndim):  # t [B] → [B, 1, ...] 广播到 ndim 维
    while t.ndim < ndim:
        t = t[..., None]
    return t
class FlowMatchingObjective:
    """Rectified Flow velocity loss:x1=norm(mel), x0=randn, xt=(1-t)x0+t x1, target=x1-x0。"""
    def __init__(self, spec_min: float = -12.0, spec_max: float = 2.0,
                 time_schedule: Literal["lognorm", "uniform"] = "lognorm",
                 loss_type: Literal["l2", "l2_lognorm"] = "l2",
                 dtype=mx.float32):
        self.spec_min, self.spec_max = spec_min, spec_max
        self.time_schedule, self.loss_type, self.dtype = time_schedule, loss_type, dtype
    def norm(self, x: mx.array) -> mx.array:
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1
    def sample_time(self, size: int) -> mx.array:
        if self.time_schedule == "lognorm":
            return sample_time_lognorm(size, self.dtype)
        return sample_time_uniform(size, self.dtype)
    def compute_loss(self, x1_mel: mx.array, velocity_fn: Callable, cond=None,
                     mask: mx.array | None = None) -> mx.array:
        x1 = self.norm(x1_mel.astype(self.dtype))
        x0 = mx.random.normal(x1.shape, dtype=self.dtype)
        t = _expand(self.sample_time(x1.shape[0]), x1.ndim)
        xt = (1 - t) * x0 + t * x1
        pred = velocity_fn(xt, t.squeeze(), cond)
        if self.loss_type == "l2":
            loss = (pred - (x1 - x0)) ** 2
        else:
            # l2_lognorm: weight = 0.398942 / t / (1-t) * exp(-0.5 * log(t/(1-t))^2)
            tv = t.reshape(-1)
            w = _expand(0.398942 / tv / (1 - tv)
                        * mx.exp(-0.5 * mx.log(tv / (1 - tv)) ** 2), x1.ndim)
            loss = w * (pred - (x1 - x0)) ** 2
        if mask is not None and loss.ndim >= 2 and mask.shape[1] == loss.shape[1]:
            loss = loss * mask[..., None].astype(self.dtype)
            return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)
        return loss.mean()
class DDPMObjective:
    """DDPM epsilon-prediction MSE:betas=linspace(1e-4,0.02,1000), xt=√acp·x0+√(1-acp)·noise。"""
    def __init__(self, spec_min: float = -12.0, spec_max: float = 2.0,
                 timesteps: int = 1000, max_beta: float = 0.02,
                 loss_type: Literal["l2", "l1"] = "l2", dtype=mx.float32):
        self.spec_min, self.spec_max = spec_min, spec_max
        self.timesteps, self.loss_type, self.dtype = timesteps, loss_type, dtype
        # 与 torch 版一致的线性 beta schedule
        import numpy as np
        betas = np.linspace(1e-4, max_beta, timesteps).astype(np.float32)
        acp = np.cumprod(1.0 - betas, axis=0)
        self.sqrt_alphas_cumprod = mx.array(np.sqrt(acp), dtype=dtype)
        self.sqrt_one_minus_alphas_cumprod = mx.array(np.sqrt(1.0 - acp), dtype=dtype)
    def norm(self, x: mx.array) -> mx.array:
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1
    def compute_loss(self, x0_mel: mx.array, denoise_fn: Callable, cond=None,
                     t_max: int | None = None, mask: mx.array | None = None) -> mx.array:
        x0 = self.norm(x0_mel.astype(self.dtype))
        t = mx.random.randint(0, t_max if t_max is not None else self.timesteps,
                              shape=(x0.shape[0],))
        noise = mx.random.normal(x0.shape, dtype=self.dtype)
        sqrt_acp = _expand(self.sqrt_alphas_cumprod[t], x0.ndim)
        sqrt_omacp = _expand(self.sqrt_one_minus_alphas_cumprod[t], x0.ndim)
        eps_pred = denoise_fn(sqrt_acp * x0 + sqrt_omacp * noise, t, cond)
        if mask is not None and mask.shape[1] == x0.shape[1]:
            elem = (eps_pred - noise) ** 2 * mask[..., None].astype(self.dtype)
            return elem.sum() / (mask.sum() * elem.shape[-1] + 1e-8)
        if self.loss_type == "l1":
            return mx.abs(eps_pred - noise).mean()
        return ((eps_pred - noise) ** 2).mean()
