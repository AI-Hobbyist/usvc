"""Backend protocol (DSP-002 §2)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.feature_config import FeatureSpec

__all__ = ["Backend"]


class Backend(ABC):
    #: Default feature requirements. Override in subclass.
    DEFAULT_FEATURES: FeatureSpec = FeatureSpec()

    @staticmethod
    @abstractmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        """Resolve the actual FeatureSpec from training config (allow user override)."""
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self, dataset_dir: str, split: str, feature_hashes: dict[str, str]):
        raise NotImplementedError
