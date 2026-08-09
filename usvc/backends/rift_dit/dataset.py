"""rift_dit dataset: reads preprocessed .pt features via mmap.

Layout (from DSP-001 preprocessing):
    mel/{mel_hash}/{speaker}/{file}.pt        [T, 128]
    f0/{f0_hash}/{speaker}/{file}.pt          [T]
    rms/{rms_hash}/{speaker}/{file}.pt        [T]
    units/{kind}/{units_hash}/{speaker}/{file}.pt  [T, D]

This mirrors RIFT-SVC's SVCDataset but uses usvc's path_tag scheme.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ...data.meta import load_meta
from .model import RIFTDiTModel  # noqa: F401 (avoid circular at runtime)
from ...train.dataset import BaseSVCDataset

__all__ = ["RIFTDataset"]


class RIFTDataset(BaseSVCDataset):
    """Dataset for rift_dit backend.

    Reads .pt features via mmap (like RIFT-SVC dataset.py).
    Random temporal crop is provided by BaseSVCDataset._crop().
    No pitch shift augmentation (RIFT uses drop_speaker CFG instead).
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
        super().__init__(dataset_dir, split, max_frames, augment)
        self.dataset_dir = Path(dataset_dir)
        self.feature_hashes = feature_hashes
        self.encoder_kind = encoder_kind

        # Load meta to get audio list and speaker map
        self.meta = load_meta(str(dataset_dir))
        self.spk2idx = {s: i for i, s in enumerate(self.meta.speakers)}

        # Build key list from audios
        all_audios = self.meta.audios
        # Simple split: if no explicit splits, use all for both train/val
        # (for real training, configure splits in meta.json)
        if self.meta.splits:
            split_files = set(self.meta.splits.get(split, []))
            self.keys = [
                (a.speaker, a.file_name)
                for a in all_audios
                if a.file_name in split_files
            ]
        else:
            self.keys = [(a.speaker, a.file_name) for a in all_audios]

        # Resolve feature paths based on hashes
        self._mel_dir = self.dataset_dir / "mel" / feature_hashes.get("mel", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")

        # Units have encoder_kind subdirectory
        units_hash = feature_hashes.get("units", "")
        self._units_dir = (
            self.dataset_dir / "units" / encoder_kind / units_hash
        )

        # mmap loader
        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0] + ".pt" if not fn.endswith(".pt") else fn

        mel = self._load_pt(self._mel_dir / spk / stem)
        f0 = self._load_pt(self._f0_dir / spk / stem)
        rms = self._load_pt(self._rms_dir / spk / stem)
        units = self._load_pt(self._units_dir / spk / stem)

        # Force-align all features to mel's frame count (units may be off-by-one
        # due to contentvec gather-align boundary effects).
        T = mel.shape[0]
        f0 = f0[:T]
        rms = rms[:T]
        units = units[:T]

        return {
            "mel": mel,
            "f0": f0,
            "rms": rms,
            "units": units,
            "spk_id": self.spk2idx.get(spk, 0),
        }
