"""usvc Triton kernel 模块（M2+）。

每个 fused 算子提供统一接口，自动选择 triton / fallback。
数值等价守卫：tests/test_triton_fusion.py。
import 失败自动 fallback，跨平台（Linux triton / Windows triton-windows）。
"""
from __future__ import annotations

import importlib

from ._compat import _HAS_TRITON, _PLATFORM

__all__ = ["_HAS_TRITON", "_PLATFORM", "fused_attention", "fused_sinegen"]


def __getattr__(name: str):
    # 按需加载子模块，避免无 triton 环境下 import 整个包失败，也避免顶层 import 的递归。
    _LAZY = {
        "fused_attention": "usvc._triton.fused_attention",
        "fused_norm": "usvc._triton.fused_norm",
        "fused_sinegen": "usvc._triton.fused_sinegen",
    }
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name])
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
