"""usvc Triton kernel 模块（M2+）。

每个 fused 算子提供统一接口：
    def fused_xxx(*args, **kwargs):
        '''数值等价于 PyTorch 原生路径。自动选择 triton / fallback。'''
        if _HAS_TRITON:
            return _triton_impl(*args, **kwargs)
        return _pytorch_impl(*args, **kwargs)

数值等价守卫：tests/test_triton_fusion.py 对每个算子 assert_close。
import 失败自动 fallback，跨平台（Linux triton / Windows triton-windows）。
"""
from __future__ import annotations

from ._compat import _HAS_TRITON, _PLATFORM

__all__ = ["_HAS_TRITON", "_PLATFORM"]

# 延迟导入具体 kernel，避免无 triton 环境下 import 整个模块失败
def __getattr__(name: str):
    if name == "fused_mel_post":
        from . import fused_mel
        return fused_mel.fused_mel_post
    if name == "fused_spec_mag":
        from . import fused_mel
        return fused_mel.fused_spec_mag
    if name == "fused_rms":
        from . import fused_rms
        return fused_rms.fused_rms
    raise AttributeError(name)
