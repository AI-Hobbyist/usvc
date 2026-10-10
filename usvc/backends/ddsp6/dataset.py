"""ddsp6 dataset: mel(128), f0, volume(=rms), units(contentvec), spk_id, aug_shift.

Reads preprocessed .pt features. Pitch augmentation (offline aug_mel/aug_vol)
is optional; when present, used 50% of the time (DDSP-SVC default ON). Shared
implementation lives in usvc/backends/_dataset.py (MelUnitsDataset); this file
pins the ddsp6 parameter defaults (2s @ 44.1k/512, tta2x encoder).
"""
from __future__ import annotations

from .._dataset import MelUnitsDataset

__all__ = ["DDSPDataset"]


class DDSPDataset(MelUnitsDataset):
    strict_speakers = True
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 172,  # DDSP default: 2s @ 44.1k/512
        augment: bool = True,
        encoder_kind: str = "contentvec_tta2x",
    ):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment,
                         encoder_kind, pitch_aug=True)
