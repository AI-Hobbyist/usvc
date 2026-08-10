"""MLX 可微 STFT/ISTFT + mel 提取(GAN mel loss、DDSP 谱路径共用)。

镜像 torch 版 so-vits/RVC 的 mel_spectrogram_torch:
    - STFT:pad reflect → 分帧(as_strided)→ hann 窗 → rfft
    - mel filterbank:slaney(htk=False),与 librosa.filters.mel 一致
    - spec_to_mel:linear spec → mel(log10(clip(x, clip_val)) * 20 域)
    - mel_spectrogram:wav → spec → mel(so-vits/RVC 的 mel 域)

全部基于 mlx 算子,梯度可回传(供 GAN 的 mel loss 用)。
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

__all__ = [
    "stft",
    "istft",
    "mel_filterbank",
    "spec_to_mel",
    "mel_spectrogram",
    "hann_window",
]


def hann_window(n: int, dtype=mx.float32) -> mx.array:
    """Hann 窗(与 torch.hann_window 一致,periodic=False)。"""
    w = np.hanning(n + 1)[:-1].astype(np.float32)
    return mx.array(w, dtype=dtype)


def _pad_reflect(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """1D reflect padding(与 torch F.pad reflect 一致,含端点镜像)。"""
    if pad_left == 0 and pad_right == 0:
        return x
    x_np = np.array(x)
    return mx.array(np.pad(x_np, (pad_left, pad_right), mode="reflect"))


def _stft_sovits(
    x: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """so-vits/RVC 训练用的 STFT(镜像 _spectrogram_torch)。

    pad = (n_fft - hop)/2 reflect(手动),center=False,返回功率谱 [B, F, T]。
    """
    B, L = x.shape
    pad = (n_fft - hop_length) // 2
    x = _pad_reflect(x, pad, pad)
    # 分帧 as_strided [B, T, win_length]
    x_np = np.array(x)
    Lp = x_np.shape[1]
    n_frames = 1 + (Lp - win_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        x_np,
        shape=(B, n_frames, win_length),
        strides=(x_np.strides[0], x_np.strides[1] * hop_length, x_np.strides[1]),
        writeable=False,
    ).copy()
    frames = mx.array(frames)
    frames = frames * window[None, None, :]
    spec = mx.fft.rfft(frames, axis=-1, n=n_fft)  # [B, T, F]
    power = spec.real * spec.real + spec.imag * spec.imag
    return power.transpose(0, 2, 1)  # [B, F, T]


def stft(
    x: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: mx.array | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
) -> mx.array:
    """1D STFT。[B, L] → [B, n_fft//2+1, T]。

    与 torch.stft(return_complex=True) 数值对齐(center=True,reflect pad)。
    默认 center=True 用 n_fft/2 pad(与 torch.stft 默认一致);
    GAN mel 训练用 mel_spectrogram(内部 center=False + (n_fft-hop)/2 pad)。
    """
    B, L = x.shape
    if center:
        pad = n_fft // 2
        x = _pad_reflect(x, pad, pad)
        L = x.shape[1]
    if window is None:
        window = hann_window(win_length)
    # 分帧:as_strided [B, T, win_length]
    n_frames = 1 + (L - win_length) // hop_length
    x_np = np.array(x)
    frames = np.lib.stride_tricks.as_strided(
        x_np,
        shape=(B, n_frames, win_length),
        strides=(x_np.strides[0], x_np.strides[1] * hop_length, x_np.strides[1]),
        writeable=False,
    ).copy()
    frames = mx.array(frames)
    w = window.astype(frames.dtype)
    frames = frames * w[None, None, :]
    # rfft 沿最后一维 → [B, T, n_fft//2+1](实部/虚部)
    spec = mx.fft.rfft(frames, axis=-1, n=n_fft)
    return spec.transpose(0, 2, 1)  # [B, n_fft//2+1, T]


def istft(
    spec: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: mx.array | None = None,
    center: bool = True,
    length: int | None = None,
) -> mx.array:
    """1D ISTFT(overlap-add)。[B, n_fft//2+1, T] → [B, L]。"""
    if window is None:
        window = hann_window(win_length)
    B, F, T = spec.shape
    # irfft → [B, T, n_fft]
    frames = mx.fft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1)
    frames = frames * window[None, None, :]
    # overlap-add
    out_len = hop_length * (T - 1) + win_length
    out = mx.zeros((B, out_len), dtype=frames.dtype)
    out_np = np.zeros((B, out_len), dtype=np.float32)
    frames_np = np.array(frames)
    for t in range(T):
        out_np[:, t * hop_length : t * hop_length + win_length] += frames_np[:, t, :]
    out = mx.array(out_np)
    if center:
        pad = n_fft // 2
        out = out[:, pad : pad + out_len - 2 * pad] if length is None else out[:, pad : pad + length]
    elif length is not None:
        out = out[:, :length]
    return out


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    dtype=mx.float32,
) -> mx.array:
    """Mel filterbank [n_mels, n_fft//2+1](slaney,htk=False,与 librosa 一致)。

    torch 版 so-vits/RVC 用 librosa.filters.mel 预计算后存 buffer,
    因此直接复刻 librosa 的 mel 计算。
    """
    import librosa
    fb = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False,
    ).astype(np.float32)
    return mx.array(fb, dtype=dtype)


def spec_to_mel(
    spec: mx.array,
    n_fft: int,
    n_mels: int,
    sr: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    clip_val: float = 1e-5,
) -> mx.array:
    """linear spec [B, n_fft//2+1, T] → mel [B, n_mels, T]。

    与 torch 版(so-vits spec_to_mel)一致:
        mel = filterbank @ spec; log(clamp(mel, min=clip_val))  [自然对数]
    """
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax).astype(spec.dtype)
    mel = fb @ spec  # [B, n_mels, T]
    mel = mx.clip(mel, clip_val, None)
    return mx.log(mel)


def mel_spectrogram(
    wav: mx.array,
    n_fft: int,
    n_mels: int,
    sr: int,
    hop_length: int,
    win_length: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    clip_val: float = 1e-5,
    eps: float = 1e-6,
) -> mx.array:
    """wav [B, L] → mel [B, n_mels, T](可微,GAN mel loss 用)。

    镜像 torch 版 mel_spectrogram_torch(so-vits/RVC 训练用):
        pad=(n_fft-hop)/2 reflect,center=False,hann 窗,
        mag=sqrt(real^2+imag^2+eps); mel = filterbank @ mag; log(clamp(mel, clip_val))
    """
    if wav.ndim == 1:
        wav = wav[None, :]
    power = _stft_sovits(
        wav, n_fft, hop_length, win_length,
        window=hann_window(win_length, wav.dtype),
    )  # [B, F, T] power(real^2+imag^2)
    mag = mx.sqrt(power + eps)
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax).astype(mag.dtype)
    mel = fb @ mag
    mel = mx.clip(mel, clip_val, None)
    return mx.log(mel)
