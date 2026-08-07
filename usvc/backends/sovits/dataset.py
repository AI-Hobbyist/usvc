"""sovits dataset: reads preprocessed features for GAN training.

sovits needs: units(hubert-soft), f0, spec(linear spectrogram), wav(waveform), spk_id, uv.
Uses filelist-based loading. Variable-length, sliced at train time.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch

from ...data.meta import load_meta
from ...train.dataset import BaseSVCDataset

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
        # rms (volume) for vol_embedding; may not exist in all datasets
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
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
            units = units[start : start + mf]
            f0 = f0[start : start + mf]
            uv = uv[start : start + mf]
            if rms is not None:
                rms = rms[start : start + mf]
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = spec[start : start + mf]
            T = mf
        elif T < mf:
            pad_n = mf - T
            units = torch.nn.functional.pad(units, (0, 0, 0, pad_n))
            f0 = torch.nn.functional.pad(f0, (0, pad_n))
            uv = torch.nn.functional.pad(uv, (0, pad_n))
            if rms is not None:
                rms = torch.nn.functional.pad(rms, (0, pad_n))
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_n))
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
            hop = self.meta.hop
            target_len = T * hop
            if wav.shape[0] > target_len:
                wav = wav[:target_len]
            elif wav.shape[0] < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        if spec is not None:
            # Force-align spec frames to exactly T (units frame count).
            # spec and units may differ by ±1 frame due to differing STFT
            # conventions (spec uses center=True len//hop+1; units align via
            # gather). Without this, collate sees [C, 47] vs [C, 48].
            if spec.ndim == 3:
                spec = spec[0]  # [T_spec, C]
            spec_T = spec.shape[0]
            if spec_T > T:
                spec = spec[:T]
            elif spec_T < T:
                spec = torch.nn.functional.pad(spec, (0, 0, 0, T - spec_T))
            spec = spec.transpose(0, 1)  # [C, T]

        return {
            "units": units,
            "f0": f0,
            "uv": uv,
            "mel": spec,  # store spec in mel field (SVCBatch.mel is generic target)
            "wav": wav,
            "rms": rms,  # [T] volume (None if not preprocessed)
            "spk_id": self.spk2idx.get(spk, 0),
        }
