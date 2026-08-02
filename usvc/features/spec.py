"""Linear spectrogram extractor (DSP-001 §4.5).

Used by so-vits and RVC backends: a plain STFT magnitude spectrogram
(no mel filterbank). Output [T, n_fft//2+1], aligned to dataset hop.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.feature_config import FeatureConfig

__all__ = ["SpecExtractor"]


class SpecExtractor:
    def __init__(self, fc: FeatureConfig, sr: int, hop: int) -> None:
        assert fc.kind == "spec"
        self.fc = fc
        self.sr = sr
        self.hop = hop
        p = fc.params
        self.n_fft = int(p.get("n_fft", 2048))
        self.win_length = int(p.get("win_length", self.n_fft))
        self.center = bool(p.get("center", False))
        self._hann: torch.Tensor | None = None
        self._device = "cpu"

    def to(self, device: str) -> "SpecExtractor":
        self._device = device
        if self._hann is not None:
            self._hann = self._hann.to(device)
        return self

    @torch.inference_mode()
    def extract(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Returns [T, n_fft//2+1] magnitude spectrogram, T = len//hop+1."""
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        wav = wav.to(self._device)
        if self._hann is None or str(self._hann.device) != self._device:
            self._hann = torch.hann_window(self.win_length).to(self._device)

        if not self.center:
            # Same reflect-pad convention as mel: pad n_fft//2 each side so the
            # output frame count equals len//hop+1 (matches mel/f0/rms).
            pad = self.n_fft // 2
            wav = torch.nn.functional.pad(wav.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)

        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self._hann,
            center=self.center,
            return_complex=True,
        )
        spec = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-6)  # magnitude
        return spec.T  # [T, n_fft//2+1]
