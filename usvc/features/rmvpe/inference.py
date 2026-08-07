"""RMVPE inference wrapper — torchaudio-free (DSP-002 P0).

Adapted from RIFT-SVC; the only change is replacing
``torchaudio.transforms.Resample`` with ``torch.nn.functional.resample`` so the
core deps stay at (torch/numpy/librosa/...).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import MEL_FMAX, MEL_FMIN, N_MELS, SAMPLE_RATE, WINDOW_LENGTH
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0

__all__ = ["RMVPE"]


class RMVPE:
    """Root mean square F0 estimator (RMVPE).

    Operates on 16kHz input. ``infer_from_audio`` resamples anything else to
    16kHz via torch.functional (no torchaudio dependency).
    """

    def __init__(
        self, model_path: str, hop_length: int = 160, device: str = "cpu", *, dtype: torch.dtype = torch.float32
    ) -> None:
        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, weights_only=True, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        # Only the E2E0 model runs at low precision; mel_extractor keeps FP32
        # (its mel_basis + STFT require matching float32 dtypes).
        self.model = model.to(device=device, dtype=dtype)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(device)
        self.device = device
        self.dtype = dtype

    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="constant")
            mel = mel.to(self.dtype)  # cast to model precision (fp32/fp16/bf16)
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden: torch.Tensor, thred: float = 0.03, use_viterbi: bool = False) -> np.ndarray:
        if use_viterbi:
            return to_viterbi_f0(hidden, thred=thred)
        return to_local_average_f0(hidden, thred=thred)

    @torch.inference_mode()
    def infer_from_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        device: str | None = None,
        thred: float = 0.03,
        use_viterbi: bool = False,
    ) -> np.ndarray:
        device = device or self.device
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        # Keep the mel/STFT path in float32: MelSpectrogram's hann_window and
        # mel_basis are fp32 buffers, and torch.stft requires input & window to
        # share a dtype. Casting audio to fp16/bf16 here either crashes or relies
        # on undocumented implicit promotion. Low precision is applied only at
        # mel2hidden (right before the E2E model) — matching RVC/RIFT-SVC.
        audio = audio.to(device=device).float()

        if sample_rate != SAMPLE_RATE:
            # librosa resample (no torchaudio dependency)
            import librosa
            import numpy as np

            wav_np = librosa.resample(
                audio.squeeze(0).cpu().float().numpy(),
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE,
                res_type="soxr_lq",  # F0 is quality-insensitive; soxr_lq is ~12x faster
            )
            audio = torch.from_numpy(wav_np).to(device=device).float().unsqueeze(0)

        mel = self.mel_extractor(audio, center=True)
        hidden = self.mel2hidden(mel)
        return self.decode(hidden, thred=thred, use_viterbi=use_viterbi)
