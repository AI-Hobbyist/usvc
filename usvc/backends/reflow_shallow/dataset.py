"""reflow_shallow dataset: mel/units/f0/rms .pt (44.1k/512 contentvec768 family).

Shares the sovits/ddsp6 preprocessing family; no pitch aug. Shared impl:
MelUnitsDataset.
"""
from __future__ import annotations

from .._dataset import MelUnitsDataset

__all__ = ["ReflowShallowDataset"]


class ReflowShallowDataset(MelUnitsDataset):
    def __init__(self, dataset_dir, split, feature_hashes, max_frames=128,
                 augment=True, encoder_kind="contentvec"):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment,
                         encoder_kind, pitch_aug=False)