"""RMVPE inference wrapper (RIFT-SVC port). Only change vs upstream: torchaudio
Resample -> torch F.resample-free librosa path so deps stay torch/numpy/librosa."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import MEL_FMAX, MEL_FMIN, N_MELS, SAMPLE_RATE, WINDOW_LENGTH
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


class RMVPE:
    """f0 estimator on 16kHz input; anything else resampled via librosa soxr_lq."""

    def __init__(self, model_path, hop_length=160, device="cpu", *, dtype=torch.float32):
        model = E2E0(4, 1, (2, 2))
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu")["model"], strict=False)
        model.eval()
        # Only E2E0 runs at dtype; mel_extractor stays FP32 (mel_basis+STFT need matching fp32)
        self.model = model.to(device=device, dtype=dtype)
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX).to(device)
        self.device, self.dtype = device, dtype

    def mel2hidden(self, mel):
        with torch.no_grad():
            n = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n), mode="constant").to(self.dtype)
            return self.model(mel)[:, :n]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        return to_viterbi_f0(hidden, thred=thred) if use_viterbi else to_local_average_f0(hidden, thred=thred)

    @torch.inference_mode()
    def infer_from_audio(self, audio, sample_rate=16000, device=None, thred=0.03, use_viterbi=False):
        device = device or self.device
        if audio.dim() == 1: audio = audio.unsqueeze(0)
        # mel/STFT path stays FP32; low precision only at mel2hidden (matches RVC/RIFT-SVC)
        audio = audio.to(device=device).float()
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio = torch.from_numpy(librosa.resample(audio.squeeze(0).cpu().float().numpy(), orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_lq")).to(device=device).float().unsqueeze(0)  # soxr_lq ~12x faster, f0-insensitive
        return self.decode(self.mel2hidden(self.mel_extractor(audio, center=True)), thred=thred, use_viterbi=use_viterbi)