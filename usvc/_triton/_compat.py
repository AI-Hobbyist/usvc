"""平台探测 + triton 可用性。"""
import platform

try:
    import triton, triton.language as tl  # noqa: F401
    _PLATFORM, _HAS_TRITON = platform.system(), True
except Exception:
    _PLATFORM, _HAS_TRITON = platform.system(), False
