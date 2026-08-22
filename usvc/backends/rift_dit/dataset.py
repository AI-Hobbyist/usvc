"""rift_dit dataset: preprocessed .pt features (mel/f0/rms/units), no pitch aug.

Layout (DSP-001): {mel,a}/(hash)/(spk)/(file).pt [T,128], f0/rms [T],
units/{kind}/{hash}/{spk}/{file}.pt [T,D]. Shared impl: MelUnitsDataset.
"""
from __future__ import annotations

from .._dataset import MelUnitsDataset

__all__ = ["RIFTDataset"]


class RIFTDataset(MelUnitsDataset):
    def __init__(self, dataset_dir, split, feature_hashes, max_frames=256,
                 augment=True, encoder_kind="contentvec_tta2x"):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment,
                         encoder_kind, pitch_aug=False)