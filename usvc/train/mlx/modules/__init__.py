"""MLX backend 训练 module 注册表。

build_mlx_module(cfg) -> (model_g, model_d, loss_fns)
    model_g: 生成器/主模型(mlx.nn.Module)
    model_d: 判别器(GAN backend)或 None
    loss_fns: dict
        - 非 GAN:{"loss": loss_fn(model, batch) -> loss}
        - GAN:{"g": g_loss_fn(g_model, d_model, batch) -> loss,
               "d": d_loss_fn(d_model, g_model, batch) -> loss}
"""
from __future__ import annotations

__all__ = ["build_mlx_module"]


def build_mlx_module(cfg):
    backend = cfg.backend
    if backend == "rift_dit":
        from . import rift_dit as m
    elif backend == "ddsp6":
        from . import ddsp6 as m
    elif backend == "diffusion":
        from . import diffusion as m
    elif backend == "reflow_shallow":
        from . import reflow_shallow as m
    elif backend == "reflow_vae":
        from . import reflow_vae as m
    elif backend == "sovits":
        from . import sovits as m
    elif backend == "rvc":
        from . import rvc as m
    else:
        raise ValueError(f"no mlx training module for backend: {backend}")
    return m.build(cfg)
