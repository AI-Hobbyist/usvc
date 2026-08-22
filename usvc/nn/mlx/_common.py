"""MLX 共享类型:任何有 ``kind: str`` + ``params: dict`` 的对象(含
usvc.data.feature_config.FeatureConfig)都满足此协议;独立定义避免循环依赖。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["FeatureConfigLike"]


@runtime_checkable
class FeatureConfigLike(Protocol):
    kind: str
    params: dict
