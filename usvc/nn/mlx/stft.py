"""MLX 可微 STFT/ISTFT + mel 提取(GAN mel loss、DDSP 谱路径共用;镜像
torch 版 so-vits/RVC 的 mel_spectrogram_torch:pad reflect → as_strided 分帧
→ hann → rfft;mel filterbank slaney/htk=False)。全 mx 算子,梯度可回传。
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

__all__ = ["stft", "istft", "mel_filterbank", "spec_to_mel", "mel_spectrogram",
           "hann_window"]


def hann_window(n: int, dtype=mx.float32) -> mx.array:
    """periodic hann: np.hanning(n+1)[:-1](float64→f32)。
    ⚠ 与 _ops.hann_window(mx.cos 直接 f32)数学同、位不同(实测 ~2e-7),须共存。"""
    return mx.array(np.hanning(n + 1)[:-1].astype(np.float32), dtype=dtype)


def _pad_reflect(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """F.pad reflect 等价(含端点镜像;numpy 路径与 torch 逐位一致)。"""
    if pad_left == 0 and pad_right == 0:
        return x
    x_np = np.array(x)
    return mx.array(np.pad(x_np, (pad_left, pad_right), mode="reflect"))


def _stft_sovits(x: mx.array, n_fft: int, hop_length: int, win_length: int,
                 window: mx.array, eps: float = 1e-6) -> mx.array:
    B, L = x.shape
    pad = (n_fft - hop_length) // 2
    x = _pad_reflect(x, pad, pad)
    x_np = np.array(x)
    n_frames = 1 + (x_np.shape[1] - win_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        x_np, shape=(B, n_frames, win_length),
        strides=(x_np.strides[0], x_np.strides[1] * hop_length, x_np.strides[1]),
        writeable=False,
    ).copy()
    spec = mx.fft.rfft(mx.array(frames) * window[None, None, :], axis=-1, n=n_fft)
    return (spec.real * spec.real + spec.imag * spec.imag).transpose(0, 2, 1)


def stft(x: mx.array, n_fft: int, hop_length: int, win_length: int,
         window: mx.array | None = None, center: bool = True,
         pad_mode: str = "reflect") -> mx.array:
    B, L = x.shape
    if center:
        x = _pad_reflect(x, n_fft // 2, n_fft // 2)
        L = x.shape[1]
    if window is None:
        window = hann_window(win_length)
    n_frames = 1 + (L - win_length) // hop_length
    x_np = np.array(x)
    frames = np.lib.stride_tricks.as_strided(
        x_np, shape=(B, n_frames, win_length),
        strides=(x_np.strides[0], x_np.strides[1] * hop_length, x_np.strides[1]),
        writeable=False,
    ).copy()
    spec = mx.fft.rfft(mx.array(frames) * window.astype(frames.dtype)[None, None, :],
                       axis=-1, n=n_fft)
    return spec.transpose(0, 2, 1)


def istft(spec: mx.array, n_fft: int, hop_length: int, win_length: int,
          window: mx.array | None = None, center: bool = True,
          length: int | None = None) -> mx.array:
    if window is None:
        window = hann_window(win_length)
    B, F, T = spec.shape
    frames = mx.fft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1) * window[None, None, :]
    out_len = hop_length * (T - 1) + win_length
    out_np = np.zeros((B, out_len), dtype=np.float32)
    for t in range(T):
        out_np[:, t * hop_length: t * hop_length + win_length] += np.array(frames)[:, t, :]
    out = mx.array(out_np)
    if center:
        pad = n_fft // 2
        out = out[:, pad: pad + out_len - 2 * pad] if length is None else out[:, pad: pad + length]
    elif length is not None:
        out = out[:, :length]
    return out


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0,
                   fmax: float | None = None, dtype=mx.float32) -> mx.array:
    """[n_mels, n_fft//2+1] slaney(htk=False,与 librosa 一致;torch 版存 buffer)。"""
    import librosa
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin,
                             fmax=fmax, htk=False).astype(np.float32)
    return mx.array(fb, dtype=dtype)


def spec_to_mel(spec: mx.array, n_fft: int, n_mels: int, sr: int,
                fmin: float = 0.0, fmax: float | None = None,
                clip_val: float = 1e-5) -> mx.array:
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax).astype(spec.dtype)
    return mx.log(mx.clip(fb @ spec, clip_val, None))


def mel_spectrogram(wav: mx.array, n_fft: int, n_mels: int, sr: int,
                    hop_length: int, win_length: int, fmin: float = 0.0,
                    fmax: float | None = None, clip_val: float = 1e-5,
                    eps: float = 1e-6) -> mx.array:
    """wav [B, L](或 1d)→ mel [B, n_mels, T](可微,GAN mel loss):
    pad=(n_fft-hop)/2 reflect, mag=sqrt(power+eps), fb @ mag, log(clamp)。"""
    if wav.ndim == 1:
        wav = wav[None, :]
    mag = mx.sqrt(
        _stft_sovits(wav, n_fft, hop_length, win_length,
                     window=hann_window(win_length, wav.dtype)) + eps)
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax).astype(mag.dtype)
    return mx.log(mx.clip(fb @ mag, clip_val, None))