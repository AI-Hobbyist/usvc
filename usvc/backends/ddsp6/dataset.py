"""ddsp6 dataset: reads preprocessed features for DDSP training.

DDSP6.3 uses: mel(128), f0, volume (=rms), units(contentvec), spk_id, aug_shift.
Pitch augmentation (offline aug_mel/aug_vol) is optional; if present, used 50%.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ...data.meta import load_meta
from ...train.dataset import BaseSVCDataset

__all__ = ["DDSPDataset"]


class DDSPDataset(BaseSVCDataset):
    """Dataset for ddsp6 backend.

    Reads: mel/{hash}/{spk}/{fn}.pt, f0/, rms/ (as volume), units/.
    Force-aligns all features to mel's frame count.
    Optional pitch augmentation reads aug_mel/aug_vol if present.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 172,  # DDSP default: 2s @ 44.1k/512
        augment: bool = True,
        encoder_kind: str = "contentvec_tta2x",
    ):
        super().__init__(dataset_dir, split, max_frames, augment)
        self.dataset_dir = Path(dataset_dir)
        self.feature_hashes = feature_hashes
        self.encoder_kind = encoder_kind

        self.meta = load_meta(str(dataset_dir))
        self.spk2idx = {s: i for i, s in enumerate(self.meta.speakers)}

        if self.meta.splits:
            split_files = set(self.meta.splits.get(split, []))
            self.keys = [
                (a.speaker, a.file_name)
                for a in self.meta.audios
                if a.file_name in split_files
            ]
        else:
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios]

        self._mel_dir = self.dataset_dir / "mel" / feature_hashes.get("mel", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
        self._units_dir = (
            self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        )
        # Pitch augmentation (present only when use_pitch_aug preprocessed them).
        self._aug_mel_dir = self.dataset_dir / "aug_mel" / feature_hashes.get("aug_mel", "")
        self._aug_shift_dir = self.dataset_dir / "aug_shift" / feature_hashes.get("aug_shift", "")

        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0] + ".pt" if not fn.endswith(".pt") else fn

        mel = self._load_pt(self._mel_dir / spk / stem)
        f0 = self._load_pt(self._f0_dir / spk / stem)
        rms = self._load_pt(self._rms_dir / spk / stem)
        units = self._load_pt(self._units_dir / spk / stem)

        # Force-align to mel frame count (units may be off-by-one)
        T = mel.shape[0]
        f0 = f0[:T]
        rms = rms[:T]
        units = units[:T]

        return {
            "mel": mel,
            "f0": f0,
            "rms": rms,  # DDSP calls this volume
            "units": units,
            "spk_id": self.spk2idx.get(spk, 0),
        }

    def _augment(self, item: dict) -> dict:
        """DDSP pitch augmentation: 50% chance use precomputed aug_mel/aug_vol
        + shift f0 by 2^(shift/12) (original DDSP-SVC behaviour).
        """
        return self._pitch_augment(item)
