"""reflow_shallow dataset: reads mel + units + f0 + rms.

mel/units/f0/rms use the 44.1k/512 contentvec768 config family, shared
with sovits/ddsp6 so preprocessed features are reused. No pitch aug.
Shared implementation lives in usvc/backends/_dataset.py (MelUnitsDataset).
"""
from __future__ import annotations

from .._dataset import MelUnitsDataset

__all__ = ["ReflowShallowDataset"]


class ReflowShallowDataset(MelUnitsDataset):
    """Dataset for the reflow_shallow backend (v2, needs units)."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 128,
        augment: bool = True,
        encoder_kind: str = "contentvec",
    ):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment,
                         encoder_kind, pitch_aug=False)