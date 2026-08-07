"""Mel extractor (DSP-001 §4.1).

Shared by rift_dit / ddsp6 / diffusion at the default config
(44100/512/2048/128mel/fmin40/fmax16000). STFT via torch, mel filterbank via
librosa. Output is [T, n_mels] (frames-first) for easy frame slicing.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.feature_config import FeatureConfig

__all__ = ["MelExtractor", "default_mel_config", "nvstft_pad"]


def nvstft_pad(wav: torch.Tensor, win_length: int, hop: int) -> torch.Tensor:
    """Pad a 1d waveform with the nvSTFT ``center=False`` convention.

    Mirrors ``nsf_hifigan/nvSTFT.py::STFT.get_mel`` so the frame grid
    (window centered at ``k*hop + (win-hop)//2``, frame count ``(L-hop)//hop+1``)
    is identical to the openvpi NSF-HiFiGAN / DDSP-SVC internal MelExtractor.
    Falls back to constant padding when the clip is shorter than one window
    (reflect would be undefined), which keeps very short clips from crashing.
    """
    pad_left = (win_length - hop) // 2
    pad_right = max((win_length - hop + 1) // 2, win_length - wav.shape[-1] - pad_left)
    mode = "reflect" if pad_right < wav.shape[-1] else "constant"
    return torch.nn.functional.pad(wav.unsqueeze(0), (pad_left, pad_right), mode=mode).squeeze(0)


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
        """wav: 1d float. Returns [T, n_mels] float32.

        Uses the nvSTFT ``center=False`` padding convention (identical to the
        openvpi NSF-HiFiGAN / DDSP-SVC internal ``MelExtractor``):
        ``pad_left = (win-hop)//2``, ``pad_right = (win-hop+1)//2``, so every
        frame window is centered at ``k*hop + (win-hop)//2`` and the frame count
        is ``(L-hop)//hop + 1``. This MUST match the mel grid used inside the
        ddsp6/sovits/rvc models (which re-extract mel from the synthesized wav
        with the same padding), otherwise ``F.mse_loss(ddsp_mel, gt_spec)`` and
        the reflow conditioning compare systematically half-hop-shifted mels.
        """
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        wav = wav.to(self._device)
        self._ensure_basis(self._device)

        if not self.center:
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
        # nvstft_pad padding length is asymmetric and length-dependent, so the
        # batch path still pads each clip individually then right-zero-pads to
        # the batch max for a single fused stft (trailing frames are discarded).

        # Normalize inputs to 1d float tensors on device.
        norms = []
        for w in wavs:
            if isinstance(w, np.ndarray):
                w = torch.from_numpy(w).float()
            norms.append(w.to(dev).float())

        # Skip degenerate clips (empty / too short for a single window).  They
        # cannot be reflect-padded and would crash the whole batch.
        keep = [w for w in norms if w.shape[0] >= self.win_length]
        if not keep:
            return [torch.zeros(1, self.n_mels, device=dev) for _ in norms]

        if not self.center:
            padded = [nvstft_pad(w, self.win_length, self.hop) for w in keep]
        else:
            padded = keep
        lens = [w.shape[0] for w in keep]  # original lengths, for frame counts
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
            pad_mode="reflect",
            return_complex=True,
        )
        spec = torch.abs(spec)  # [B, n_fft//2+1, T_pad]
        mel = self._mel_basis @ spec  # [B, n_mels, T_pad]
        mel = torch.log(torch.clamp(mel, min=1e-5))

        results = []
        for i, n in enumerate(lens):
            # Match frame_len_for nvSTFT grid: (L-hop)//hop+1.
            n_frames = max(1, (n - self.hop) // self.hop + 1)
            results.append(mel[i, :, :n_frames].T)  # [T, n_mels]
        return results
