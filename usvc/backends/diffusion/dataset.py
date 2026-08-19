"""diffusion dataset: reads preprocessed features for DDPM training.

Same data pattern as ddsp6 (mel/f0/volume/units/spk). Pitch aug optional
(precomputed aug_mel/aug_shift, 50%; Diffusion-SVC default OFF at spec
level, enabled via train_cfg use_pitch_aug).
Shared implementation lives in usvc/backends/_dataset.py (MelUnitsDataset).
"""
from __future__ import annotations

from .._dataset import MelUnitsDataset

__all__ = ["DiffusionDataset"]


class DiffusionDataset(MelUnitsDataset):
    """Dataset for diffusion backend (Diffusion-SVC style)."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 172,
        augment: bool = True,
        encoder_kind: str = "contentvec",
    ):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment,
                         encoder_kind, pitch_aug=True)