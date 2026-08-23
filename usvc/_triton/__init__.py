"""usvc Triton kernels：统一接口自动选 triton/fallback，import 失败自动 fallback，跨平台（Linux triton / Windows triton-windows）；等价守卫 tests/test_triton_fusion.py。"""
import importlib

from ._compat import _HAS_TRITON, _PLATFORM

_LAZY = {"fused_attention": "usvc._triton.fused_attention", "fused_norm": "usvc._triton.fused_norm",
         "fused_sinegen": "usvc._triton.fused_sinegen"}


def __getattr__(name):
    # 按需加载子模块：无 triton 环境下包仍可 import，也避免顶层 import 的递归。
    if name in _LAZY:
        return importlib.import_module(_LAZY[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
