"""Mel extractor (DSP-001 §4.1): torch STFT + librosa filterbank, [T, n_mels]
frames-first. nvSTFT center=False grid (shared with ddsp6/sovits/rvc internals):
pad_left=(win-hop)//2, pad_right=(win-hop+1)//2 -> window centered at
k*hop+(win-hop)//2, frame count (L-hop)//hop+1.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.feature_config import FeatureConfig


def nvstft_pad(wav, win_length, hop):
    """nvSTFT center=False pad (constant fallback when shorter than one window)."""
    pad_left = (win_length - hop) // 2
    pad_right = max((win_length - hop + 1) // 2, win_length - wav.shape[-1] - pad_left)
    mode = "reflect" if pad_right < wav.shape[-1] else "constant"
    return F.pad(wav.unsqueeze(0), (pad_left, pad_right), mode=mode).squeeze(0)


def default_mel_config(*, n_fft=2048, win_length=2048, n_mels=128, fmin=40.0, fmax=16000.0, center=False):
    # sr/hop injected at runtime (dataset-global)
    return {"n_fft": n_fft, "win_length": win_length, "n_mels": n_mels, "fmin": fmin, "fmax": fmax, "center": center}


class MelExtractor:
    def __init__(self, fc, sr, hop):
        assert fc.kind in ("mel", "aug_mel")
        self.fc, self.sr, self.hop = fc, sr, hop
        p = fc.params
        self.n_fft, self.win_length, self.n_mels = int(p["n_fft"]), int(p["win_length"]), int(p["n_mels"])
        self.fmin, self.center = float(p["fmin"]), bool(p["center"])
        self.fmax = None if p["fmax"] is None else float(p["fmax"])
        self._mel_basis = None
        self._hanns = {0.0: None}  # hann cache per keyshift, base window built with basis
        self._device = "cpu"

    def to(self, device):
        self._device = device
        if self._mel_basis is not None:
            self._mel_basis = self._mel_basis.to(device)
        for ks in self._hanns:
            self._hanns[ks] = self._hanns[ks].to(device) if self._hanns[ks] is not None else None
        return self

    def _ensure_basis(self, device):
        if self._mel_basis is not None and str(self._mel_basis.device) == device:
            return
        import librosa
        self._mel_basis = torch.from_numpy(librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)).float().to(device)
        self._hanns[0.0] = torch.hann_window(self.win_length).to(device)

    def _get_hann(self, keyshift, win_new):
        if keyshift not in self._hanns:
            self._hanns[keyshift] = torch.hann_window(win_new).to(self._device)
        return self._hanns[keyshift]

    @torch.inference_mode()
    def extract(self, wav, keyshift=0.0):
        """wav 1d -> [T, n_mels]. keyshift: filterbank-equivalent shift — window
        scaled by 2**(keyshift/12), freq axis resized back to n_fft//2+1 with
        amplitude comp win/win_new; hop unchanged (same frame grid)."""
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        wav = wav.to(self._device)
        self._ensure_basis(self._device)
        if keyshift != 0.0:
            factor = 2 ** (keyshift / 12)
            n_fft_new, win_new, hop_new = int(np.round(self.n_fft * factor)), int(np.round(self.win_length * factor)), self.hop
            spec = torch.abs(torch.stft(nvstft_pad(wav, win_new, hop_new), n_fft=n_fft_new, hop_length=hop_new, win_length=win_new,
                                         window=self._get_hann(keyshift, win_new), center=False, pad_mode="reflect", return_complex=True))
            size = self.n_fft // 2 + 1
            if spec.size(0) < size: spec = F.pad(spec, (0, 0, 0, size - spec.size(0)))
            spec = spec[:size, :] * (self.win_length / win_new)  # amplitude comp
        else:
            if not self.center:
                wav = nvstft_pad(wav, self.win_length, self.hop)
            spec = torch.abs(torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length,
                                        window=self._hanns[0.0], center=self.center, pad_mode="reflect", return_complex=True))
        return torch.log(torch.clamp(self._mel_basis @ spec, min=1e-5)).T

    @torch.inference_mode()
    def extract_batch(self, wavs, keyshifts=None):
        if not wavs:
            return []
        if keyshifts is not None:
            assert len(keyshifts) == len(wavs), "keyshifts must match wavs length"
            if any(ks != 0.0 for ks in keyshifts):
                return [self.extract(w, ks) for w, ks in zip(wavs, keyshifts)]
        return self._extract_batch_fused(wavs)

    def _extract_batch_fused(self, wavs):
        """One fused stft for N clips (keyshift=0): each clip padded individually,
        right-zero-padded to batch max, trailing frames discarded."""
        self._ensure_basis(self._device)
        norms = [w.to(self._device).float() if not isinstance(w, np.ndarray) else torch.from_numpy(w).float().to(self._device) for w in wavs]
        keep = [w for w in norms if w.shape[0] >= self.win_length]  # degenerate clips would crash the batch stft
        if not keep:
            return [torch.zeros(1, self.n_mels, device=self._device) for _ in norms]
        padded = [nvstft_pad(w, self.win_length, self.hop) if not self.center else w for w in keep]
        lens = [w.shape[0] for w in keep]
        batched = torch.zeros(len(padded), max(p.shape[0] for p in padded), device=self._device)
        for i, p in enumerate(padded):
            batched[i, : p.shape[0]] = p
        mel = torch.log(torch.clamp(self._mel_basis @ torch.abs(torch.stft(batched, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length,
                                    window=self._hanns[0.0], center=self.center, pad_mode="reflect", return_complex=True)), min=1e-5))
        return [mel[i, :, :max(1, (n - self.hop) // self.hop + 1)].T for i, n in enumerate(lens)]