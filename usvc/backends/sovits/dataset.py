"""sovits dataset: reads preprocessed features for GAN training.

sovits needs: units(hubert-soft), f0, spec(linear spectrogram), wav(waveform), spk_id, uv.
Uses filelist-based loading. Variable-length, sliced at train time.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ...data.meta import load_meta
from ...train.dataset import BaseSVCDataset

__all__ = ["SoVITSDataset"]


class SoVITSDataset(BaseSVCDataset):
    """Dataset for sovits backend (VITS + NSF-HiFiGAN)."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 790,  # sovits default segment
        augment: bool = True,
        encoder_kind: str = "hubertsoft",
    ):
        super().__init__(dataset_dir, split, max_frames, augment)
        self.dataset_dir = Path(dataset_dir)
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

        self._units_dir = (
            self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        )
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")

        # sovits needs spec (linear spectrogram) and wav
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")
        self._audio_dir = self.dataset_dir / "audio"

        self.sr = self.meta.sr
        self.hop = self.meta.hop

        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0]
        pt_stem = stem + ".pt"
        wav_stem = stem + ".wav"

        units = self._load_pt(self._units_dir / spk / pt_stem)
        f0 = self._load_pt(self._f0_dir / spk / pt_stem)

        # spec (linear spectrogram) — may not exist in all datasets
        spec_path = self._spec_dir / spk / pt_stem
        spec = self._load_pt(spec_path) if spec_path.exists() else None

        # wav
        wav_path = self._audio_dir / spk / wav_stem
        wav = None
        if wav_path.exists():
            import soundfile as sf

            wav, _ = sf.read(str(wav_path), dtype="float32")
            wav = torch.from_numpy(wav)

        # uv from f0
        uv = (f0 > 0).float()

        # Align frame-level features
        T = units.shape[0]
        f0 = f0[:T]
        uv = uv[:T]
        if spec is not None:
            spec = spec[:, :T] if spec.ndim == 2 else spec[:T]

        return {
            "units": units,
            "f0": f0,
            "uv": uv,
            "mel": spec,  # store spec in mel field (SVCBatch.mel is generic target)
            "wav": wav,
            "spk_id": self.spk2idx.get(spk, 0),
        }
