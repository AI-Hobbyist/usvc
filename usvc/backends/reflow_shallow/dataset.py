"""reflow_shallow dataset: reads preprocessed features for shallow-diffusion training.

Unlike DDSP6, this backend needs only mel (the corruption target/source),
f0 and rms (kept for inference-feature parity).  No units required — the
reflow velocity network is conditioned on mel alone.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ...data.meta import load_meta
from ...train.dataset import BaseSVCDataset

__all__ = ["ReflowShallowDataset"]


class ReflowShallowDataset(BaseSVCDataset):
    """Dataset for the reflow_shallow backend.

    Reads: mel/{hash}/{spk}/{fn}.pt, f0/, rms/.  No units.
    Force-aligns all features to mel's frame count.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 128,
        augment: bool = True,
        encoder_kind: str = "contentvec",
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

        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0] + ".pt" if not fn.endswith(".pt") else fn

        mel = self._load_pt(self._mel_dir / spk / stem)
        f0 = self._load_pt(self._f0_dir / spk / stem)
        rms = self._load_pt(self._rms_dir / spk / stem)

        T = mel.shape[0]
        f0 = f0[:T]
        rms = rms[:T]

        # The shared collate (BaseSVCDataset) requires a "units" key for
        # frame_len bookkeeping; reflow_shallow does not use it, so emit a
        # 1-D placeholder aligned to T frames.
        units_dummy = mel.new_zeros(T, 1)

        return {
            "mel": mel,
            "f0": f0,
            "rms": rms,
            "units": units_dummy,
            "spk_id": self.spk2idx.get(spk, 0),
        }

    def _augment(self, item: dict) -> dict:
        return item
