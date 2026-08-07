"""Audio source layer: resample/normalize + shared 16kHz cache (DSP-001 §3, §4.4.4).

The resampled-to-sr audio under ``audio/`` is the single source of truth. A
16kHz derivative is memoized for the units encoders (HubertSoft / ContentVec
both require 16kHz input) so the resample happens once even when two different
encoders run over the same clip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

__all__ = [
    "load_audio",
    "save_audio",
    "resample_to",
    "SharedAudio16kCache",
    "frame_len_for",
]


def load_audio(path: str | Path, *, sr: int, mono: bool = True, quality: str = "hq") -> np.ndarray:
    """Load audio, resample to ``sr``, optional mono. Returns float32 [samples].

    ``quality``: "hq" (soxr_hq, for content-encoder input) or "fast"
    (soxr_lq, ~75x faster, fine for mel/f0/rms which are quality-insensitive).
    """
    wav, orig_sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.T  # (channels, samples)
    if mono and wav.shape[0] > 1:
        wav = wav.mean(axis=0, keepdims=True)
    wav = wav[0]
    if orig_sr != sr:
        wav = resample_np(wav, orig_sr, sr, quality=quality)
    return wav.astype("float32", copy=False)


def save_audio(path: str | Path, wav: np.ndarray, *, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype("float32"), sr, subtype="FLOAT")


def resample_to(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    return resample_np(wav, orig_sr, target_sr)


def resample_np(
    wav: np.ndarray, orig_sr: int, target_sr: int, *, quality: str = "hq"
) -> np.ndarray:
    """Resample via librosa (no torchaudio / no resampy).

    ``quality`` selects the soxr backend:
      - "hq"  -> soxr_hq (slow; needed for content encoders which are
        sensitive to resample artifacts)
      - "fast"-> soxr_lq (~75x faster; fine for mel/f0/rms — measured
        <0.01% relative diff in those features)
    """
    import librosa

    res_type = "soxr_hq" if quality == "hq" else "soxr_lq"
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)


def frame_len_for(n_samples: int, hop: int) -> int:
    """Canonical feature frame count: matches the nvSTFT ``center=False`` grid.

    nvSTFT pads ``(win-hop)//2`` left / ``(win-hop+1)//2`` right and runs
    ``center=False`` stft, yielding ``(L + (win-hop)) // hop - (win//hop) + 1``
    frames. For the standard ``win == 4*hop`` config this simplifies to
    ``(L - hop) // hop + 1``, i.e. ``max(1, L // hop)`` for ``L >= hop`` — which
    is the exact frame count produced by MelExtractor / SpecExtractor and by
    the ddsp6/sovits internal mel extractors. Keeping f0/rms/units on this same
    count is the DSP-001 §3 invariant.
    """
    if n_samples <= 0:
        return 0
    # (L - hop) // hop + 1 == L // hop for L >= hop; clamp to >= 1.
    return max(1, (n_samples - hop) // hop + 1)


class SharedAudio16kCache:
    """In-process cache of 16kHz audio for units encoders (DSP-001 §4.4.4).

    Both HubertSoft and ContentVec need 16kHz input. Without this, two encoders
    running on the same clip would each resample. The cache key is the
    dataset-relative (speaker, file_name) tuple, which is unique per source clip.
    """

    def __init__(self, target_sr: int = 16000) -> None:
        self.target_sr = target_sr
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def get(self, speaker: str, file_name: str, audio_dir: Path, dataset_sr: int) -> torch.Tensor:
        key = (speaker, file_name)
        if key in self._cache:
            return self._cache[key]
        wav_path = audio_dir / speaker / file_name
        wav = load_audio(wav_path, sr=dataset_sr)
        if dataset_sr != self.target_sr:
            wav = resample_np(wav, dataset_sr, self.target_sr)
        t = torch.from_numpy(wav).float()
        self._cache[key] = t
        return t

    def clear(self) -> None:
        self._cache.clear()
