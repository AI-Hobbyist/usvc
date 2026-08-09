"""RMS / volume extractor (DSP-001 §4.3)."""
from __future__ import annotations

import numpy as np
import torch

from ..data.feature_config import FeatureConfig

__all__ = ["RmsExtractor"]


class RmsExtractor:
    def __init__(self, fc: FeatureConfig, sr: int, hop: int, device: str = "cpu") -> None:
        assert fc.kind == "rms"
        self.fc = fc
        self.sr = sr
        self.hop = hop
        self.win_length = int(fc.params.get("win_length", hop))
        self.device = device

    @torch.inference_mode()
    def extract(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Returns [T] float32. T = (L-hop)//hop+1 (nvSTFT grid).

        Vectorized: unfold all frames at once and compute RMS in one shot.
        On CUDA the unfold runs on GPU (~0.1ms vs ~30ms CPU for a 10s clip);
        falls back to CPU when device is not available.
        Uses the nvSTFT ``center=False`` padding (pad_left=(win-hop)//2,
        pad_right=(win-hop+1)//2) so the frame grid matches MelExtractor /
        SpecExtractor exactly (window centered at ``i*hop + (win-hop)//2``).
        """
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        wav = wav.float()
        if self.device != "cpu":
            try:
                wav = wav.to(self.device)
            except Exception:
                pass  # device unavailable; fall back to CPU compute
        n = wav.shape[0]

        # nvSTFT center=False padding: asymmetric, window centered at i*hop+256.
        from .mel import nvstft_pad
        padded = nvstft_pad(wav, self.win_length, self.hop)
        n_frames = max(1, (n - self.hop) // self.hop + 1)
        # framed: [n_frames, win_length]
        framed = padded.unfold(0, self.win_length, self.hop)[:n_frames]
        rms = torch.sqrt((framed * framed).mean(dim=1) + 1e-9)
        return rms
