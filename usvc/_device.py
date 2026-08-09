"""Unified device abstraction for the torch / MLX dual-backend.

usvc 原生以 torch 为后端(见 ``_envcheck.py`` 的 CUDA/triton 探测),本模块
在框架最外层建立 **impl 层** 抽象:``resolve_impl`` 决定当前用 ``torch``
还是 ``mlx``,``resolve_device`` 把抽象设备名解析为各后端自己的设备字符串。

设计原则(与 usvc 现有的 ``device`` 字符串透传风格一致,见
``infer/engine.py`` / ``data/engine.py``):

- ``impl`` 是"后端选择":``torch``(CUDA/MPS/CPU)或 ``mlx``(Apple Metal)。
- ``device`` 是"设备名",语义随 impl 变化:
  - torch: ``cuda`` / ``mps`` / ``cpu``(或带编号 ``cuda:0``)
  - mlx: ``gpu`` / ``cpu``
- MLX 不可用(非 Apple Silicon、未安装、Metal 失败)时,**自动回退 torch**,
  保证 usvc 的 "graceful degradation" 原则(见 README 与 _envcheck.py)。

优先级(从高到低):
  1. ``USVC_BACKEND`` 环境变量(``torch`` | ``mlx``)
  2. 调用方显式传入的 ``pref`` 参数
  3. 平台默认:Apple Silicon(Darwin + arm64)默认 mlx,其余平台 torch
"""
from __future__ import annotations

import os
import platform

__all__ = [
    "resolve_impl",
    "mlx_available",
    "resolve_device",
    "is_apple_silicon",
    "default_device",
]


def is_apple_silicon() -> bool:
    """是否为 Apple Silicon(Darwin + arm64)。

    注意:Intel Mac(Darwin + x86_64)也能装 mlx,但无 GPU 加速价值,
    且 MLX 官方 wheel 主要面向 arm64;统一按 arm64 判断。
    """
    return platform.system() == "Darwin" and platform.machine().lower() in (
        "arm64",
        "aarch64",
    )


def mlx_available() -> bool:
    """探测 MLX 是否可用(安装 + Metal 可计算)。

    触发一次真实计算(``mx.eval(mx.zeros(1))``),确认 Metal 后端就绪,
    避免"装了但 Metal 不可用"的假阳性。探测失败(ImportError / RuntimeError)
    一律返回 False,由调用方回退 torch。
    """
    try:
        import mlx.core as mx  # noqa: F401

        mx.eval(mx.zeros(1))
        return True
    except Exception:
        return False


def resolve_impl(pref: str | None = None) -> str:
    """解析后端实现:``torch`` 或 ``mlx``。

    优先级:环境变量 ``USVC_BACKEND`` > 参数 ``pref`` > 平台默认。
    - ``USVC_BACKEND=mlx`` 但 mlx 不可用 → 回退 ``torch`` 并打印一次提示。
    - 显式 ``pref="torch"`` / ``"mlx"`` 尊重(mlx 不可用时回退 torch)。
    """
    env = os.environ.get("USVC_BACKEND", "").strip().lower()
    if env:
        if env in ("torch", "mlx"):
            if env == "mlx" and not mlx_available():
                _warn_fallback("env USVC_BACKEND=mlx 但 MLX 不可用")
                return "torch"
            return env
        # 非法环境变量值:忽略,继续走参数/平台默认
        _warn_fallback(f"忽略非法 USVC_BACKEND={env!r}")
    if pref:
        pref = pref.strip().lower()
        if pref in ("torch", "mlx"):
            if pref == "mlx" and not mlx_available():
                _warn_fallback("显式 pref=mlx 但 MLX 不可用")
                return "torch"
            return pref
    # 平台默认:Apple Silicon 默认 mlx,其余 torch
    if is_apple_silicon() and mlx_available():
        return "mlx"
    return "torch"


def resolve_device(impl: str, pref: str | None = None) -> str:
    """把抽象设备名解析为后端自己的设备字符串。

    Args:
        impl: ``torch`` 或 ``mlx``。
        pref: 可选的显式设备字符串。
            - torch: ``cuda`` / ``cuda:N`` / ``mps`` / ``cpu``(透传)
            - mlx: ``gpu`` / ``cpu``

    Returns:
        torch: ``pref`` 若给;否则 cuda > mps > cpu 自动探测。
        mlx: ``pref`` 若给(仅接受 gpu/cpu,非法回退 gpu);否则 gpu
            (Apple Silicon 必有 Metal;mlx_available 已保证)。
    """
    if impl == "mlx":
        if pref:
            p = pref.strip().lower()
            if p in ("gpu", "cpu"):
                return p
        return "gpu"
    # torch 分支
    if pref:
        return pref
    return _torch_default_device()


def default_device() -> str:
    """便捷入口:按当前平台/后端返回默认设备字符串。"""
    impl = resolve_impl()
    return resolve_device(impl)


def _torch_default_device() -> str:
    """torch 默认设备:自动探测 cuda > mps > cpu。

    与 ``_envcheck.full_report`` 的探测顺序一致;任何探测失败回退 cpu。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _warn_fallback(reason: str) -> None:
    """回退 torch 时打印一次提示(不刷屏)。"""
    import sys

    sys.stderr.write(f"[usvc] {reason},回退到 torch 后端。\n")
