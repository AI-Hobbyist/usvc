"""sovits dataset: reads preprocessed features for GAN training.

sovits needs: units(hubert-soft), f0, spec(linear spectrogram), wav(waveform), spk_id, uv.
Uses filelist-based loading. Variable-length, sliced at train time.
Shared skeleton lives in usvc/backends/_dataset.py (BaseDirDataset + crop/pad
helpers). The random crop keeps so-vits' own random source: the per-dataset
self._rng (seed 42, NOT the global torch generator — see RVCDataset).
"""
from __future__ import annotations

import numpy as np
import torch

from .._dataset import BaseDirDataset, align_spec_to_frames, align_wav, slice_or_pad

__all__ = ["SoVITSDataset", "interp_unvoiced_f0"]


def interp_unvoiced_f0(f0: torch.Tensor) -> torch.Tensor:
    """Linear-interpolate unvoiced (f0==0) frames (so-vits/RVC convention).

    Callers must capture the raw voicing mask (uv = f0 > 0) BEFORE calling
    this: sovits keeps emb_uv conditioned on the true unvoiced frames while
    the NSF source receives the fully-voiced interpolated f0.
    """
    voiced = f0 > 0
    if not voiced.any():
        return f0
    device, dtype = f0.device, f0.dtype
    f0_np = f0.detach().float().cpu().numpy()
    unvoiced_idx = np.where(~(f0_np > 0))[0]
    voiced_idx = np.where(f0_np > 0)[0]
    f0_np[unvoiced_idx] = np.interp(unvoiced_idx, voiced_idx, f0_np[voiced_idx])
    return torch.from_numpy(f0_np).to(device=device, dtype=dtype)


class SoVITSDataset(BaseDirDataset):
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
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment, encoder_kind)

        self._units_dir = (
            self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        )
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")

        # sovits needs spec (linear spectrogram) and wav
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")
        # rms (volume) for vol_embedding; may not exist in all datasets
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
        self._audio_dir = self.dataset_dir / "audio"

        self.sr = self.meta.sr
        self.hop = self.meta.hop

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

        # rms (volume) — may not exist in all datasets
        rms_path = self._rms_dir / spk / pt_stem
        rms = self._load_pt(rms_path) if rms_path.exists() else None

        # wav
        wav_path = self._audio_dir / spk / wav_stem
        wav = None
        if wav_path.exists():
            import soundfile as sf

            wav, _ = sf.read(str(wav_path), dtype="float32")
            wav = torch.from_numpy(wav)

        # uv from raw f0 (must be captured before interpolation: emb_uv keeps
        # the true voicing while the NSF source gets fully-voiced f0)
        uv = (f0 > 0).float()
        f0 = interp_unvoiced_f0(f0)

        # Align frame-level features; apply fixed-length crop/pad to spec so
        # batching is consistent. NOTE: spec is [T, C] here (frames first);
        # after transpose it becomes [C, T] and the base-class _crop() can no
        # longer align it with units (shape[0] != T), so crop before transpose.
        mf = self.max_frames
        T = units.shape[0]
        if T > mf:
            start = int(torch.randint(0, T - mf + 1, (1,), generator=self._rng).item())
            units = slice_or_pad(units, mf, T, start)
            f0 = slice_or_pad(f0, mf, T, start)
            uv = slice_or_pad(uv, mf, T, start)
            if rms is not None:
                rms = slice_or_pad(rms, mf, T, start)
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = slice_or_pad(spec, mf, T, start)
            T = mf
        elif T < mf:
            units = slice_or_pad(units, mf, T)
            f0 = slice_or_pad(f0, mf, T)
            uv = slice_or_pad(uv, mf, T)
            if rms is not None:
                rms = slice_or_pad(rms, mf, T)
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = slice_or_pad(spec, mf, T)
            T = mf
        else:
            if spec is not None and spec.ndim == 3:
                spec = spec[0]
        f0 = f0[:T]
        uv = uv[:T]
        # Align wav to T*hop (frame count may exceed wav samples by one
        # frame due to len//hop + 1 convention); GAN wave slicing needs
        # ids_slice*hop + segment <= T*hop.
        if wav is not None:
            wav = align_wav(wav, T * self.meta.hop)
        if spec is not None:
            # Force-align spec frames to exactly T (units frame count).
            spec = align_spec_to_frames(spec, T)

        return {
            "units": units,
            "f0": f0,
            "uv": uv,
            "mel": spec,  # store spec in mel field (SVCBatch.mel is generic target)
            "wav": wav,
            "rms": rms,  # [T] volume (None if not preprocessed)
            "spk_id": self.spk2idx.get(spk, 0),
        }