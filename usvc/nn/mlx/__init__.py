"""usvc MLX 后端(镜像 torch 实现,Metal 加速;``USVC_BACKEND=mlx`` 选择)。"""
from __future__ import annotations

__all__ = ["_HAS_MLX"]

try:
    import mlx.core as mx  # noqa: F401

    _HAS_MLX = True
except Exception:  # pragma: no cover
    _HAS_MLX = False
