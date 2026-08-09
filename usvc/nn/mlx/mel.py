"""MLX 版 Mel 提取器(nvSTFT center=False 约定)。

镜像 ``usvc/features/mel.py::MelExtractor``。帧网格红线:必须逐位复现
``nvstft_pad`` 的 ``pad_left=(win-hop)//2, pad_right=(win-hop+1)//2``
与帧数 ``(L-hop)//hop+1``,否则与 ddsp6/sovits/rvc 模型内重提取的 mel
半 hop 错位,导致 mse_loss / reflow conditioning 系统性偏差。

与 torch 版差异:
  - torch.stft → ``_ops.stft_nv``(MLX 无 STFT,手写 rfft)
  - F.pad(mode='reflect') → ``_ops.pad_reflect1d_fallback``
  - librosa mel filterbank: 预计算(纯 numpy/CTypeslibso,与 torch 无关)
    → 直接复用 librosa 生成 basis,转 mx.array
  - torch.hann_window → 手写 hann(与 torch 逐位一致)
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from ._common import FeatureConfigLike
from ._ops import (
    frame_len_nv,
    hann_window,
    pad_reflect1d_fallback,
    stft_nv,
)

__all__ = ["MelExtractor", "default_mel_config", "nvstft_pad"]


def nvstft_pad(wav: mx.array, win_length: int, hop: int) -> mx.array:
    """nvSTFT center=False 补边(复现 features/mel.py::nvstft_pad)。

    pad_left = (win - hop) // 2
    pad_right = max((win - hop + 1) // 2, win - L - pad_left)
    mode = reflect if pad_right < L else constant
    """
    L = wav.shape[-1]
    pad_left = (win_length - hop) // 2
    pad_right = max((win_length - hop + 1) // 2, win_length - L - pad_left)
    return pad_reflect1d_fallback(wav, pad_left, pad_right)


def default_mel_config(
    *,
    n_fft: int = 2048,
    win_length: int = 2048,
    n_mels: int = 128,
    fmin: float = 40.0,
    fmax: float = 16000.0,
    center: bool = False,
) -> dict:
    return {
        "n_fft": n_fft,
        "win_length": win_length,
        "n_mels": n_mels,
        "fmin": fmin,
        "fmax": fmax,
        "center": center,
    }


class MelExtractor:
    """MLX Mel 提取器。

    与 torch 版 API 对齐(``extract(wav) -> [T, n_mels]``),但返回 mx.array。
    用于预处理 + mel-domain 推理时的特征提取。

    Note:
        ``to(device)`` 在 MLX 下无意义(Metal 设备由 mx.set_default_device
        全局控制),保留方法签名仅为 API 兼容。
    """

    def __init__(self, fc: FeatureConfigLike, sr: int, hop: int):
        assert fc.kind in ("mel", "aug_mel")
        self.fc = fc
        self.sr = sr
        self.hop = hop
        p = fc.params
        self.n_fft = int(p["n_fft"])
        self.win_length = int(p["win_length"])
        self.n_mels = int(p["n_mels"])
        self.fmin = float(p["fmin"])
        self.fmax = p["fmax"]
        self.fmax = None if self.fmax is None else float(self.fmax)
        self.center = bool(p["center"])
        self._mel_basis: mx.array | None = None
        self._hanns: dict[float, mx.array] = {}

    def to(self, device: str = "gpu") -> "MelExtractor":
        # MLX device 全局控制,no-op 保持 API 兼容
        return self

    def _ensure_basis(self) -> None:
        if self._mel_basis is not None:
            return
        import librosa
        # librosa mel filterbank 是纯 numpy,与 torch 无关,直接复用
        mel = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
        )
        self._mel_basis = mx.array(mel.astype(np.float32))  # [n_mels, n_fft//2+1]
        self._hanns[0.0] = hann_window(self.win_length)

    def _get_hann(self, keyshift: float, win_new: int) -> mx.array:
        if keyshift not in self._hanns:
            self._hanns[keyshift] = hann_window(win_new)
        return self._hanns[keyshift]

    def extract(self, wav, keyshift: float = 0.0) -> mx.array:
        """wav: 1d float(mx.array 或可转 mx.array)→ [T, n_mels] mx.array。

        帧网格: center=False 时 nvstft_pad,帧数 = (L-hop)//hop+1。
        """
        if not isinstance(wav, mx.array):
            wav = mx.array(np.asarray(wav, dtype=np.float32))
        wav = wav.astype(mx.float32)
        self._ensure_basis()

        if keyshift != 0.0:
            factor = 2 ** (keyshift / 12)
            n_fft_new = int(round(self.n_fft * factor))
            win_new = int(round(self.win_length * factor))
            hop_new = self.hop  # speed=1: hop 不变 → 帧数不变
            hann = self._get_hann(keyshift, win_new)
            # nvSTFT pad with scaled window
            L = wav.shape[0]
            pad_left = (win_new - hop_new) // 2
            pad_right = max((win_new - hop_new + 1) // 2, win_new - L - pad_left)
            y = pad_reflect1d_fallback(wav, pad_left, pad_right)
            spec = stft_nv(y, n_fft_new, hop_new, win_new, hann)  # [freq_new, T]
            spec = mx.abs(spec) if spec.dtype != mx.float32 else spec
            # resize freq axis to n_fft//2+1
            size = self.n_fft // 2 + 1
            resize = spec.shape[0]
            if resize < size:
                spec = mx.pad(spec, [(0, size - resize), (0, 0)])
            spec = spec[:size, :] * (self.win_length / win_new)  # amplitude comp
        else:
            if not self.center:
                wav = nvstft_pad(wav, self.win_length, self.hop)
            spec = stft_nv(wav, self.n_fft, self.hop, self.win_length, self._hanns[0.0])
        mel = self._mel_basis @ spec  # [n_mels, T]
        mel = mx.log(mx.maximum(mel, 1e-5))
        return mel.T  # [T, n_mels]

    def extract_batch(self, wavs: list, keyshifts: list[float] | None = None) -> list:
        """批量提取。MLX 版暂用逐条(后续可优化为 fused 批 stft)。

        与 torch 版不同,这里不强制 keyshift 全 0 走 fused 路径
        (MLX 的 compile 对固定 shape 批次效果好,但逐条已足够正确)。
        """
        if not wavs:
            return []
        if keyshifts is not None:
            assert len(keyshifts) == len(wavs)
            return [self.extract(w, ks) for w, ks in zip(wavs, keyshifts)]
        return [self.extract(w) for w in wavs]
