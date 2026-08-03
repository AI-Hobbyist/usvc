"""rvc dataset: reads preprocessed features for RVC VITS training.

RVC needs: units(contentvec 768d), pitch(quantized token 1-255), pitchf(continuous f0),
spec(linear spectrogram), wav(waveform), sid(speaker id).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ...data.meta import load_meta
from ...train.dataset import BaseSVCDataset

__all__ = ["RVCDataset"]


class RVCDataset(BaseSVCDataset):
    """Dataset for rvc backend (VITS + NSF-HiFiGAN, v2)."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 790,
        augment: bool = True,
        encoder_kind: str = "contentvec",
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
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")
        self._audio_dir = self.dataset_dir / "audio"

        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0]
        pt_stem = stem + ".pt"
        wav_stem = stem + ".wav"

        units = self._load_pt(self._units_dir / spk / pt_stem)
        f0 = self._load_pt(self._f0_dir / spk / pt_stem)

        # spec (linear spectrogram)
        spec_path = self._spec_dir / spk / pt_stem
        spec = self._load_pt(spec_path) if spec_path.exists() else None

        # wav
        wav_path = self._audio_dir / spk / wav_stem
        wav = None
        if wav_path.exists():
            import soundfile as sf

            wav, _ = sf.read(str(wav_path), dtype="float32")
            wav = torch.from_numpy(wav)

        # RVC f0: quantize to token (1-255) + keep continuous pitchf
        # RVC formula: pitch_token = f0_to_coarse(f0), pitchf = log2(f0/10) clamped
        pitchf = f0.clone()
        pitch_token = self._f0_to_coarse(f0)

        T = units.shape[0]
        units = units[:T]
        pitch_token = pitch_token[:T]
        pitchf = pitchf[:T]
        if spec is not None:
            spec = spec[:, :T] if spec.ndim == 2 else spec[:T]

        return {
            "units": units,
            "f0": pitchf,  # continuous f0 (SVCBatch.f0)
            "pitch_token": pitch_token,  # quantized f0 token [T]
            "mel": spec,
            "wav": wav,
            "spk_id": self.spk2idx.get(spk, 0),
        }

    @staticmethod
    def _f0_to_coarse(f0: torch.Tensor) -> torch.Tensor:
        """Quantize f0 to RVC pitch token (1-255).

        RVC uses a log-scale mapping: f0=0 (unvoiced) → 255, else 1-254.
        """
        # RVC coarse f0: mapping based on log2 scale, 256 bins
        f0 = f0.float()
        uv = f0 == 0
        # log2(f0), clamp to RVC range
        import numpy as np

        f0_log = torch.where(uv, torch.ones_like(f0) * 1e-5, f0)
        f0_mel = 1127 * torch.log(1 + f0_log / 700)  # mel scale
        f0_mel_min = 0
        f0_mel_max = 1107
        f0_coarse = (
            (f0_mel - f0_mel_min)
            * 254
            / (f0_mel_max - f0_mel_min)
        ).round().long()
        f0_coarse = f0_coarse.clamp(1, 254)
        f0_coarse[uv] = 255  # unvoiced
        return f0_coarse
