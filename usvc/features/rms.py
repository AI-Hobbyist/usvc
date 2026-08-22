"""RMS / volume extractor (DSP-001 §4.3), nvSTFT grid."""

import numpy as np
import torch

from ..data.feature_config import FeatureConfig


class RmsExtractor:
    def __init__(self, fc, sr, hop, device="cpu"):
        assert fc.kind == "rms"
        self.fc, self.sr, self.hop, self.device = fc, sr, hop, device
        self.win_length = int(fc.params.get("win_length", hop))

    @torch.inference_mode()
    def extract(self, wav):  # -> [T] float32, T=(L-hop)//hop+1 (nvSTFT grid)
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        wav = wav.float()
        if self.device != "cpu":
            try:
                wav = wav.to(self.device)
            except Exception:
                pass  # device unavailable; fall back to CPU
        from .mel import nvstft_pad
        padded = nvstft_pad(wav, self.win_length, self.hop)
        n_frames = max(1, (wav.shape[0] - self.hop) // self.hop + 1)
        framed = padded.unfold(0, self.win_length, self.hop)[:n_frames]
        return torch.sqrt((framed * framed).mean(dim=1) + 1e-9)