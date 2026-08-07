"""Inference-time feature extraction for a single audio clip.

Reuses the same extractors as preprocessing (UnitsExtractor / F0Extractor /
RmsExtractor), driven by the backend's FeatureSpec so the inference features
match training exactly (same encoder, same hop alignment).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.feature_config import FeatureConfig, FeatureSpec
from ..features.f0 import F0Extractor
from ..features.rms import RmsExtractor
from ..features.units import UnitsExtractor
from ..io.audio import load_audio, resample_np


class InferFeatures:
    """Per-clip feature extractor for inference (matches preprocessing)."""

    def __init__(
        self,
        spec: FeatureSpec,
        sr: int = 44100,
        hop: int = 512,
        device: str = "cuda",
        dtype: str = "fp32",
    ) -> None:
        assert spec.units is not None, "backend feature_spec requires units"
        assert spec.f0 is not None, "backend feature_spec requires f0"
        self.spec = spec
        self.sr = sr
        self.hop = hop
        self.device = device
        self.dtype = dtype
        self.units_ext = UnitsExtractor(spec.units, sr, hop, device, dtype=dtype)
        self.f0_ext = F0Extractor(spec.f0, sr, hop, device, dtype=dtype)
        self.rms_ext = RmsExtractor(spec.rms, sr, hop) if spec.rms is not None else None
        # Runtime-switchable f0 extractors (keyed by method name).
        self._f0_extractors: dict[str, F0Extractor] = {}

    def _get_f0_extractor(self, method: str) -> F0Extractor:
        """Get (or build + cache) a F0Extractor for the given method.

        Params are copied from spec.f0 with only the extractor name replaced,
        so uv_policy / model_ckpt / thresholds stay identical to training.
        """
        assert method in ("rmvpe", "fcpe"), f"unsupported f0 method: {method}"
        if method == self.f0_ext.extractor:
            return self.f0_ext
        cached = self._f0_extractors.get(method)
        if cached is None:
            params = dict(self.spec.f0.params)
            params["extractor"] = method
            cached = F0Extractor(
                FeatureConfig("f0", params), self.sr, self.hop,
                self.device, dtype=self.dtype,
            )
            self._f0_extractors[method] = cached
        return cached

    @torch.inference_mode()
    def extract(
        self,
        wav_path: str | Path | None = None,
        f0_method: str | None = None,
        *,
        wav: np.ndarray | None = None,
    ) -> dict:
        """Extract features from an audio file.

        Args:
            wav_path: input wav file.
            f0_method: optional runtime f0 extractor override ("rmvpe" |
                "fcpe"); None uses the config default.
            wav: optional pre-loaded float32 audio at self.sr; when given,
                wav_path is ignored (used by callers that normalize the
                audio before feature extraction, e.g. loudness norm).

        Returns:
            dict with:
                units: [T, dim] float32 (aligned to hop frames)
                f0: [T] float32 (Hz, 0 = unvoiced)
                rms: [T] float32 or None
                uv: [T] float32 (voicing flag, derived from f0)
                wav: [L] float32 numpy at self.sr
                sr: int
        """
        if wav is None:
            assert wav_path is not None, "either wav_path or wav must be given"
            wav = load_audio(wav_path, sr=self.sr)  # [L] float32
        # Units: resample to the encoder's native rate (16k).
        wav16k = resample_np(wav, self.sr, 16000)
        wav16k_t = torch.from_numpy(wav16k).float()
        units = self.units_ext.extract(wav16k_t)  # [T, dim]

        wav_t = torch.from_numpy(wav).float()
        f0_ext = (
            self._get_f0_extractor(f0_method) if f0_method is not None else self.f0_ext
        )
        f0 = f0_ext.extract(wav_t)  # [T]
        rms = self.rms_ext.extract(wav_t) if self.rms_ext is not None else None  # [T]
        uv = (f0 > 0).float()

        # Trim to the common frame count (all extractors use frame_len_for).
        T = units.shape[0]
        if f0.shape[0] != T:
            f0 = f0[:T]
            uv = uv[:T]
        if rms is not None and rms.shape[0] != T:
            rms = rms[:T]

        return {
            "units": units,
            "f0": f0,
            "rms": rms,
            "uv": uv,
            "wav": wav,
            "sr": self.sr,
        }


def synth_test_wav(path: str | Path, sr: int = 44100, duration: float = 2.0,
                   f0_base: float = 220.0) -> np.ndarray:
    """Synthesize a simple test tone wav (for smoke tests)."""
    import soundfile as sf

    t = np.arange(int(sr * duration)) / sr
    vib = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0_base + vib) / sr
    wav = 0.5 * np.sin(phase) * np.exp(-t / duration)
    sf.write(path, wav.astype(np.float32), sr)
    return wav
