"""平台探测 + triton 可用性。"""
from __future__ import annotations

import platform

_PLATFORM = platform.system()  # 'Linux' / 'Windows' / 'Darwin'

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False
