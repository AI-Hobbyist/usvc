"""Audio source layer (DSP-001 §3, §4.4.4): resampled ``audio/`` is the single source of truth;
a 16kHz derivative is memoized for units encoders (HubertSoft / ContentVec both need 16kHz input).
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch


def load_audio(path, *, sr, mono=True, quality="hq"):
    # quality: "hq"=soxr_hq for content-encoder input; "fast"=soxr_lq (~75x faster,
    # <0.01% relative diff in mel/f0/rms which are quality-insensitive).
    wav, orig_sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.T  # (channels, samples)
    if mono and wav.shape[0] > 1: wav = wav.mean(axis=0, keepdims=True)
    wav = wav[0]
    if orig_sr != sr: wav = resample_np(wav, orig_sr, sr, quality=quality)
    return wav.astype("float32", copy=False)


def save_audio(path, wav, *, sr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype("float32"), sr, subtype="FLOAT")


def resample_to(wav, orig_sr, target_sr):
    return wav if orig_sr == target_sr else resample_np(wav, orig_sr, target_sr)


def resample_np(wav, orig_sr, target_sr, *, quality="hq"):  # librosa soxr; no torchaudio/resampy
    import librosa
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr,
                            res_type="soxr_hq" if quality == "hq" else "soxr_lq")


def frame_len_for(n_samples, hop):
    # nvSTFT pads (win-hop)//2 left / (win-hop+1)//2 right, center=False stft ->
    # frames = (L + win - hop)//hop - win//hop + 1; standard win==4*hop gives
    # (L-hop)//hop+1 == L//hop for L>=hop. f0/rms/units share this count (DSP-001 §3 invariant).
    if n_samples <= 0: return 0
    return max(1, (n_samples - hop) // hop + 1)


class SharedAudio16kCache:  # key=(speaker, file_name): two encoders over one clip resample once
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr; self._cache = {}

    def get(self, speaker, file_name, audio_dir, dataset_sr):
        key = (speaker, file_name)
        if key not in self._cache:
            wav = load_audio(audio_dir / speaker / file_name, sr=dataset_sr)
            if dataset_sr != self.target_sr: wav = resample_np(wav, dataset_sr, self.target_sr)
            self._cache[key] = torch.from_numpy(wav).float()
        return self._cache[key]

    def clear(self): self._cache.clear()
