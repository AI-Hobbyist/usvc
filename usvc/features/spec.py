"""Linear spectrogram extractor (DSP-001 §4.5, so-vits/rvc): plain STFT
magnitude, [T, n_fft//2+1], nvSTFT center=False grid."""

import numpy as np
import torch

from ..data.feature_config import FeatureConfig


class SpecExtractor:
    def __init__(self, fc, sr, hop):
        assert fc.kind == "spec"
        self.fc, self.sr, self.hop = fc, sr, hop
        p = fc.params
        self.n_fft, self.win_length = int(p.get("n_fft", 2048)), int(p.get("win_length", p.get("n_fft", 2048)))
        self.center = bool(p.get("center", False))
        self._hann = None
        self._device = "cpu"

    def to(self, device):
        self._device = device
        if self._hann is not None:
            self._hann = self._hann.to(device)
        return self

    @torch.inference_mode()
    def extract(self, wav):  # -> [T, n_fft//2+1], same padding grid as GAN-internal re-extract
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        wav = wav.to(self._device)
        if self._hann is None or str(self._hann.device) != self._device:
            self._hann = torch.hann_window(self.win_length).to(self._device)
        if not self.center:
            from .mel import nvstft_pad
            wav = nvstft_pad(wav, self.win_length, self.hop)
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length,
                          window=self._hann, center=self.center, pad_mode="reflect", return_complex=True)
        return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-6).T