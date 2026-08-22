import os

import numpy as np
import torch
import torch.nn.functional as F

# Cap torch's LRU audio cache to avoid unbounded memory growth during batch mel
# extraction (fcpe re-reads wav windows heavily). Must run before first DataLoader use.
os.environ.setdefault("LRU_CACHE_CAPACITY", "3")

from librosa.filters import mel as librosa_mel_fn  # core dep (original try/except fallback removed)
from ._resample import Resample


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class HannWindow(torch.nn.Module):
    def __init__(self, win_size):
        super().__init__()
        self.register_buffer('window', torch.hann_window(win_size), persistent=False)

    def forward(self): return self.window


class MelModule(torch.nn.Module):
    """Mel extractor: nvSTFT center=False manual pad -> magnitude -> librosa mel basis."""
    def __init__(self, sr, n_mels, n_fft, win_size, hop_length, fmin=None, fmax=None, clip_val=1e-5, out_stft=False):
        super().__init__()
        self.target_sr, self.n_mels, self.n_fft, self.win_size, self.hop_length = sr, n_mels, n_fft, win_size, hop_length
        self.fmin, self.fmax, self.clip_val, self.out_stft = 0 if fmin is None else fmin, sr / 2 if fmax is None else fmax, clip_val, out_stft
        self.register_buffer('mel_basis', torch.tensor(librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=self.fmin, fmax=self.fmax)).float(), persistent=False)
        self.hann_window = torch.nn.ModuleDict()

    @torch.no_grad()
    def __call__(self, y, key_shift=0, speed=1, center=False, no_cache_window=False):  # y (B,T,1) -> (B,T,n_mels)
        factor = 2 ** (key_shift / 12)
        n_fft_new, win_size_new, hop_length_new = int(np.round(self.n_fft * factor)), int(np.round(self.win_size * factor)), int(np.round(self.hop_length * speed))
        y = y.squeeze(-1)
        key = str(key_shift)
        if not no_cache_window:
            if key not in self.hann_window:
                self.hann_window[key] = HannWindow(win_size_new).to(self.mel_basis.device)
            hann_window_tensor = self.hann_window[key]()
        else:
            hann_window_tensor = torch.hann_window(win_size_new).to(self.mel_basis.device)
        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode='reflect' if pad_right < y.size(-1) else 'constant').squeeze(1)
        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=hann_window_tensor, center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        if key_shift != 0:
            size = self.n_fft // 2 + 1
            if spec.size(1) < size: spec = F.pad(spec, (0, 0, 0, size - spec.size(1)))
            spec = spec[:, :size, :] * self.win_size / win_size_new
        spec = spec[:, :512, :] if self.out_stft else torch.matmul(self.mel_basis, spec)  # stft-out hard cap 512 (original)
        return dynamic_range_compression_torch(spec, clip_val=self.clip_val).transpose(-1, -2)


class Wav2MelModule(torch.nn.Module):
    def __init__(self, sr, n_mels, n_fft, win_size, hop_length, fmin=None, fmax=None, clip_val=1e-5, mel_type="default"):
        super().__init__()
        self.sampling_rate, self.n_mels, self.n_fft, self.win_size, self.hop_size = sr, n_mels, n_fft, win_size, hop_length
        self.fmin, self.fmax, self.clip_val, self.mel_type = 0 if fmin is None else fmin, sr / 2 if fmax is None else fmax, clip_val, mel_type
        self.register_buffer('tensor_device_marker', torch.tensor(1.0).float(), persistent=False)
        self.resample_kernel = torch.nn.ModuleDict()
        self.mel_extractor = MelModule(sr, n_mels, n_fft, win_size, hop_length, self.fmin, self.fmax, clip_val, out_stft=mel_type == "stft")

    def device(self): return self.tensor_device_marker.device

    @torch.no_grad()
    def __call__(self, audio, sample_rate, keyshift=0, no_cache_window=False):  # audio (B,T,1) -> mel (B,T,n_mels) @16k grid
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                if len(self.resample_kernel) > 8: self.resample_kernel.clear()
                self.resample_kernel[key_str] = Resample(sample_rate, self.sampling_rate, lowpass_filter_width=128).to(self.tensor_device_marker.device)
            audio_res = self.resample_kernel[key_str](audio.squeeze(-1)).unsqueeze(-1)
        mel = self.mel_extractor(audio_res, keyshift, no_cache_window=no_cache_window)
        # n_frames MUST come from the *resampled* (16k) length: a 44.1k input would
        # inflate n_frames ~2.76x and the tail-constant fill corrupts the global
        # (non-causal) attention f0 output. Equivalent when sample_rate==sampling_rate.
        n_frames = int(audio_res.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]): mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]): mel = mel[:, :n_frames, :]
        return mel