"""Mel extractor (DSP-001 §4.1).

Shared by rift_dit / ddsp6 / diffusion at the default config
(44100/512/2048/128mel/fmin40/fmax16000). STFT via torch, mel filterbank via
librosa. Output is [T, n_mels] (frames-first) for easy frame slicing.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.feature_config import FeatureConfig

__all__ = ["MelExtractor", "default_mel_config"]


def default_mel_config(
    *,
    n_fft: int = 2048,
    win_length: int = 2048,
    n_mels: int = 128,
    fmin: float = 40.0,
    fmax: float = 16000.0,
    center: bool = False,
) -> dict:
    # sr/hop injected at run time from dataset meta (they are dataset-global).
    return {
        "n_fft": n_fft,
        "win_length": win_length,
        "n_mels": n_mels,
        "fmin": fmin,
        "fmax": fmax,
        "center": center,
    }


class MelExtractor:
    def __init__(self, fc: FeatureConfig, sr: int, hop: int) -> None:
        assert fc.kind == "mel"
        self.fc = fc
        self.sr = sr
        self.hop = hop
        p = fc.params
        self.n_fft = int(p["n_fft"])
        self.win_length = int(p["win_length"])
        self.n_mels = int(p["n_mels"])
        self.fmin = float(p["fmin"])
        self.fmax = p["fmax"]
        self.fmax = None if self.fmax is None else float(self.fmax)
        self.center = bool(p["center"])
        self._mel_basis: torch.Tensor | None = None
        self._hann: torch.Tensor | None = None
        self._device = "cpu"

    def to(self, device: str) -> "MelExtractor":
        self._device = device
        if self._mel_basis is not None:
            self._mel_basis = self._mel_basis.to(device)
        if self._hann is not None:
            self._hann = self._hann.to(device)
        return self

    def _ensure_basis(self, device: str) -> None:
        if self._mel_basis is not None and str(self._mel_basis.device) == device:
            return
        import librosa

        mel = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self._mel_basis = torch.from_numpy(mel).float().to(device)
        self._hann = torch.hann_window(self.win_length).to(device)

    @torch.inference_mode()
    def extract(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        """wav: 1d float. Returns [T, n_mels] float32, T = len//hop + 1.

        ``center=True`` makes stft output exactly ``len//hop + 1`` frames, which
        matches ``frame_len_for`` (DSP-001 §3 invariant). For ``center=False`` we
        reflect-pad by ``n_fft//2`` on each side to reach the same frame count.
        """
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        wav = wav.to(self._device)
        self._ensure_basis(self._device)

        if not self.center:
            # reflect-pad so output length matches (len//hop + 1)
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
        spec = torch.abs(spec)  # magnitude (power=1.0 equivalent)
        mel = self._mel_basis @ spec  # [n_mels, T]
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.T  # [T, n_mels]

    @torch.inference_mode()
    def extract_batch(self, wavs: list) -> list[torch.Tensor]:
        """Batched extract: one fused stft call for N clips.

        Each clip is reflect-padded individually (preserving correct boundary
        frames), then right-zero-padded to the batch max length and stacked into
        a [B, T] tensor for a single torch.stft call. Trailing frames beyond
        each clip's true length are discarded. Measured ~130x faster than
        per-clip extract (0.14ms vs 18.5ms per clip) — the speedup comes from
        amortizing the stft kernel-launch overhead across the whole batch.
        """
        if not wavs:
            return []
        self._ensure_basis(self._device)
        dev = self._device
        pad = self.n_fft // 2 if not self.center else 0

        # Normalize inputs to 1d float tensors on device.
        norms = []
        for w in wavs:
            if isinstance(w, np.ndarray):
                w = torch.from_numpy(w).float()
            norms.append(w.to(dev).float())

        # Per-clip reflect-pad (correct boundary), then right-zero-pad to max.
        if pad:
            padded = [
                torch.nn.functional.pad(w.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
                for w in norms
            ]
        else:
            padded = norms
        lens = [w.shape[0] for w in norms]  # original lengths, for frame counts
        lmax = max(p.shape[0] for p in padded)
        batched = torch.zeros(len(padded), lmax, device=dev)
        for i, p in enumerate(padded):
            batched[i, : p.shape[0]] = p

        spec = torch.stft(
            batched,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self._hann,
            center=self.center,
            return_complex=True,
        )
        spec = torch.abs(spec)  # [B, n_fft//2+1, T_pad]
        mel = self._mel_basis @ spec  # [B, n_mels, T_pad]
        mel = torch.log(torch.clamp(mel, min=1e-5))

        results = []
        for i, n in enumerate(lens):
            n_frames = n // self.hop + 1
            results.append(mel[i, :, :n_frames].T)  # [T, n_mels]
        return results
