"""rift_dit dataset: reads preprocessed .pt features.

Layout (from DSP-001 preprocessing):
    mel/{mel_hash}/{speaker}/{file}.pt        [T, 128]
    f0/{f0_hash}/{speaker}/{file}.pt          [T]
    rms/{rms_hash}/{speaker}/{file}.pt        [T]
    units/{kind}/{units_hash}/{speaker}/{file}.pt  [T, D]

No pitch shift augmentation (RIFT uses drop_speaker CFG instead).
Shared implementation lives in usvc/backends/_dataset.py (MelUnitsDataset).
"""
from __future__ import annotations

from .._dataset import MelUnitsDataset

__all__ = ["RIFTDataset"]


class RIFTDataset(MelUnitsDataset):
    """Dataset for rift_dit backend.

    Reads .pt features (like RIFT-SVC dataset.py). Random temporal crop is
    provided by BaseSVCDataset._crop().
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 256,
        augment: bool = True,
        encoder_kind: str = "contentvec_tta2x",
    ):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment,
                         encoder_kind, pitch_aug=False)