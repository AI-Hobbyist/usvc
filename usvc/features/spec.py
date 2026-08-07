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
        """Returns [T, n_fft//2+1] magnitude spectrogram.

        Uses the nvSTFT ``center=False`` padding convention (same as MelExtractor
        and the so-vits/RVC internal ``spectrogram_torch``), so the spec grid
        matches the mel re-extracted inside the GAN models during training.
        """
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        wav = wav.to(self._device)
        if self._hann is None or str(self._hann.device) != self._device:
            self._hann = torch.hann_window(self.win_length).to(self._device)

        if not self.center:
            from .mel import nvstft_pad
            wav = nvstft_pad(wav, self.win_length, self.hop)

        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self._hann,
            center=self.center,
            pad_mode="reflect",
            return_complex=True,
        )
        spec = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-6)  # magnitude
        return spec.T  # [T, n_fft//2+1]
