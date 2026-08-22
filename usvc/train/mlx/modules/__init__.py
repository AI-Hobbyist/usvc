"""MLX backend 训练 module 注册表。
build_mlx_module(cfg) -> (model_g, model_d, loss_fns)
    model_d: 判别器(GAN backend)或 None
    loss_fns: 非 GAN {"loss": fn(model, batch) -> loss};
              GAN {"g": fn(g_model, d_model, batch) -> loss,
                   "d": fn(d_model, g_model, batch) -> loss}
"""
from __future__ import annotations
__all__ = ["build_mlx_module"]
_BACKENDS = frozenset(
    {"rift_dit", "ddsp6", "diffusion", "reflow_shallow", "reflow_vae", "sovits", "rvc"}
)
def build_mlx_module(cfg):
    from importlib import import_module
    if cfg.backend not in _BACKENDS:
        raise ValueError(f"no mlx training module for backend: {cfg.backend}")
    return import_module(f".{cfg.backend}", __name__).build(cfg)
