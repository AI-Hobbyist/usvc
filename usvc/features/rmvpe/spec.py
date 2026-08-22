import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel


class MelSpectrogram(torch.nn.Module):
    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        self.register_buffer("mel_basis", torch.from_numpy(mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)).float())
        self.n_fft, self.hop_length, self.win_length, self.sampling_rate, self.n_mel_channels, self.clamp = n_fft, hop_length, win_length, sampling_rate, n_mel_channels, clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new, win_length_new, hop_length_new = int(np.round(self.n_fft * factor)), int(np.round(self.win_length * factor)), int(np.round(self.hop_length * speed))
        key = str(keyshift) + '_' + str(audio.device)
        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(win_length_new).to(audio.device)
        fft = torch.stft(audio, n_fft=n_fft_new, hop_length=hop_length_new, win_length=win_length_new, window=self.hann_window[key], center=center, return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            if magnitude.size(1) < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - magnitude.size(1)))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, magnitude), min=self.clamp))