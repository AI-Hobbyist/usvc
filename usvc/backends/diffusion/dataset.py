"""diffusion dataset: preprocessed mel/f0/volume/units (+pitch aug) for DDPM.

Same data pattern as ddsp6；共享实现 usvc/backends/_dataset.py (MelUnitsDataset)。
Pitch aug 默认开（预计算 aug_mel/aug_shift，50%；Diffusion-SVC spec 层默认 OFF，
由 train_cfg use_pitch_aug 启用）。
"""
from __future__ import annotations

from .._dataset import MelUnitsDataset


class DiffusionDataset(MelUnitsDataset):
    def __init__(self, dataset_dir, split, feature_hashes, max_frames=172,
                 augment=True, encoder_kind="contentvec"):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment,
                         encoder_kind, pitch_aug=True)