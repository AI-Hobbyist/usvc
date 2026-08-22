"""Pure-torch resample (drop-in for torchaudio.transforms.Resample), librosa
polyphase backend — same callable interface (Resample(orig, new)(wav))."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Resample(nn.Module):
    def __init__(self, orig_freq, new_freq, lowpass_filter_width=128):
        super().__init__()
        self.orig_freq, self.new_freq = orig_freq, new_freq

    def forward(self, wav):
        import librosa
        device, dtype = wav.device, wav.dtype
        x = wav.detach().cpu().numpy()
        ndim = x.ndim
        if ndim == 1: x = x[None, :]
        out = np.stack([librosa.resample(c, orig_sr=self.orig_freq, target_sr=self.new_freq, res_type="soxr_hq") for c in x])
        return torch.from_numpy(out[0] if ndim == 1 else out).to(device).to(dtype)