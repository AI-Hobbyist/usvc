"""MLX 模块共享类型与 helper。

FeatureConfigLike:usvc 的 FeatureConfig(dataclass)定义在
``usvc/data/feature_config.py``,用于特征提取器构造。MLX 侧提取器
(MelExtractor 等)需要同样的配置访问接口,为避免循环依赖和保持
torch 版独立,这里定义一个轻量 Protocol(结构化子类型)。

任何具有 ``kind: str`` 和 ``params: dict`` 属性的对象都满足此协议,
包括 usvc.data.feature_config.FeatureConfig。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["FeatureConfigLike"]


@runtime_checkable
class FeatureConfigLike(Protocol):
    """特征配置协议(结构化子类型,duck typing)。

    usvc.data.feature_config.FeatureConfig 满足此协议,MLX 侧提取器
    只依赖这两个属性,不耦合具体 dataclass 定义。
    """

    kind: str
    params: dict
