"""Backend protocol (DSP-002 §2, extended for DSP-003 training)."""
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
    def build_dataset(self, dataset_dir: str, split: str, feature_hashes: dict):
        """Build and return the dataset for this backend.

        Args:
            dataset_dir: preprocessed dataset root (has meta.json + features/).
            split: 'train' | 'val'.
            feature_hashes: {kind: hash} for locating feature files.
        """
        raise NotImplementedError

    @abstractmethod
    def build_module(self, train_cfg):
        """Build and return the LightningModule for this backend."""
        raise NotImplementedError
