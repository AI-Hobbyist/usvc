"""MLX Mel 提取器(nvSTFT center=False 约定,镜像 features/mel.py)。

帧网格红线(跨模型一致性):逐位复现 ``nvstft_pad`` 的
pad_left=(win-hop)//2, pad_right=max((win-hop+1)//2, win-L-pad_left)
与帧数 (L-hop)//hop+1;否则与模型内重提取 mel 半 hop 错位 → 系统性偏差。
torch.stft → _ops.stft_nv;F.pad reflect → _ops.pad_reflect1d_fallback;
hann 窗与 torch 逐位一致。
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._common import FeatureConfigLike
from ._ops import frame_len_nv, hann_window, pad_reflect1d_fallback, stft_nv  # noqa: F401

__all__ = ["MelExtractor", "default_mel_config", "nvstft_pad"]


def nvstft_pad(wav: mx.array, win_length: int, hop: int) -> mx.array:
    """nvSTFT center=False 补边(复现 features/mel.py):pad_left=(win-hop)//2,
    pad_right=max((win-hop+1)//2, win-L-pad_left);短信号回退 constant。"""
    L = wav.shape[-1]
    pad_left = (win_length - hop) // 2
    pad_right = max((win_length - hop + 1) // 2, win_length - L - pad_left)
    return pad_reflect1d_fallback(wav, pad_left, pad_right)


def default_mel_config(*, n_fft: int = 2048, win_length: int = 2048,
                       n_mels: int = 128, fmin: float = 40.0,
                       fmax: float = 16000.0, center: bool = False) -> dict:
    return {"n_fft": n_fft, "win_length": win_length, "n_mels": n_mels,
            "fmin": fmin, "fmax": fmax, "center": center}


class MelExtractor:
    def __init__(self, fc: FeatureConfigLike, sr: int, hop: int):
        assert fc.kind in ("mel", "aug_mel")
        self.fc, self.sr, self.hop = fc, sr, hop
        p = fc.params
        self.n_fft = int(p["n_fft"])
        self.win_length = int(p["win_length"])
        self.n_mels = int(p["n_mels"])
        self.fmin = float(p["fmin"])
        self.fmax = None if p["fmax"] is None else float(p["fmax"])
        self.center = bool(p["center"])
        self._mel_basis: mx.array | None = None
        self._hanns: dict[float, mx.array] = {}

    def to(self, device: str = "gpu") -> "MelExtractor":
        # mx device 全局控制,no-op 仅 API 兼容
        return self

    def _ensure_basis(self) -> None:
        if self._mel_basis is not None:
            return
        import librosa  # filterbank 纯 numpy,与 torch 无关
        mel = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                  fmin=self.fmin, fmax=self.fmax)
        self._mel_basis = mx.array(mel.astype(np.float32))  # [n_mels, F]
        self._hanns[0.0] = hann_window(self.win_length)

    def _get_hann(self, keyshift: float, win_new: int) -> mx.array:
        if keyshift not in self._hanns:
            self._hanns[keyshift] = hann_window(win_new)
        return self._hanns[keyshift]

    def extract(self, wav, keyshift: float = 0.0) -> mx.array:
        """wav 1d → [T, n_mels]。keyshift 变 n_fft/win(hop 不变→帧数不变)。"""
        if not isinstance(wav, mx.array):
            wav = mx.array(np.asarray(wav, dtype=np.float32))
        wav = wav.astype(mx.float32)
        self._ensure_basis()

        if keyshift != 0.0:
            factor = 2 ** (keyshift / 12)
            n_fft_new = int(round(self.n_fft * factor))
            win_new = int(round(self.win_length * factor))
            L = wav.shape[0]
            pad_left = (win_new - self.hop) // 2
            pad_right = max((win_new - self.hop + 1) // 2, win_new - L - pad_left)
            spec = stft_nv(pad_reflect1d_fallback(wav, pad_left, pad_right),
                           n_fft_new, self.hop, win_new,
                           self._get_hann(keyshift, win_new))  # [freq_new, T]
            spec = mx.abs(spec) if spec.dtype != mx.float32 else spec
            size = self.n_fft // 2 + 1
            if spec.shape[0] < size:
                spec = mx.pad(spec, [(0, size - spec.shape[0]), (0, 0)])
            spec = spec[:size, :] * (self.win_length / win_new)  # amplitude comp
        else:
            if not self.center:
                wav = nvstft_pad(wav, self.win_length, self.hop)
            spec = stft_nv(wav, self.n_fft, self.hop, self.win_length, self._hanns[0.0])
        mel = self._mel_basis @ spec  # [n_mels, T]
        return mx.log(mx.maximum(mel, 1e-5)).T  # [T, n_mels]

    def extract_batch(self, wavs: list, keyshifts: list[float] | None = None) -> list:
        if not wavs:
            return []
        if keyshifts is not None:
            assert len(keyshifts) == len(wavs)
            return [self.extract(w, ks) for w, ks in zip(wavs, keyshifts)]
        return [self.extract(w) for w in wavs]