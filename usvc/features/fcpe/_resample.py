"""Pure-torch resample (drop-in for torchaudio.transforms.Resample).

torchaudio is not a core dependency; this provides the same callable interface
(``Resample(orig_freq, new_freq)(wav)``) using librosa's polyphase backend.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Resample(nn.Module):
    """Resample a 1D/2D waveform tensor.

    Matches torchaudio.transforms.Resample call signature. Uses librosa under
    the hood (soxr backend) — accepts a small CPU/GPU roundtrip per call, which
    is fine for FCPE's per-clip inference.
    """

    def __init__(self, orig_freq: int, new_freq: int, lowpass_filter_width: int = 128) -> None:
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        import librosa

        device = wav.device
        x = wav.detach().cpu().numpy()
        orig_ndim = x.ndim
        if x.ndim == 1:
            x = x[None, :]
        # librosa expects (channels, samples); operate per-channel
        out = np.stack(
            [librosa.resample(c, orig_sr=self.orig_freq, target_sr=self.new_freq, res_type="soxr_hq") for c in x]
        )
        if orig_ndim == 1:
            out = out[0]
        return torch.from_numpy(out).to(device).to(wav.dtype)
