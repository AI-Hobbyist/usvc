"""MLX 缺失算子统一实现(0.32.0 适配层):weight_norm 合并、STFT(帧网格
红线逐位复现 features/mel.py 约定)、interpolate、reflect pad、共享 ODE/loss
kernel。惰性求值,返回未 eval 的 mx.array。
"""
from __future__ import annotations

import math

__all__ = [
    "merge_weight_norm",
    "stft_nv",
    "stft_center",
    "frame_len_nv",
    "hann_window",
    "interpolate_linear",
    "interpolate_nearest",
    "pad_reflect1d",
    "pad_reflect1d_fallback",
    "masked_fill",
    "norm_spec",
    "denorm_spec",
    "ode_euler",
    "ode_rk4",
    "reflow_loss",
    "sequence_mask",
    "stft_center_complex",
    "istft_center_complex",
]


def merge_weight_norm(sd: dict, *, norm_dim_overrides: dict[str, int] | None = None) -> dict:
    """torch weight_norm (weight_g/weight_v) → 普通 weight;torch-free(numpy)。

    weight = g * v / ||v||_dim。norm_dim_overrides: ContentVec pos_conv 用
    dim=2(features/cvec/contentvec.py:90),其余默认 dim=0。
    刻意不依赖 mlx/torch:ckpt 桥接层调用,产出 numpy dict 后统一转 mx.array,
    合并在 float64 做保证精度。
    """
    import numpy as np

    overrides = norm_dim_overrides or {}
    out: dict = {}
    g_bases: dict[str, str] = {}  # base -> g_key(weight_g 9 字符)
    for k in sd:
        if k.endswith(".weight_g"):
            g_bases[k[:-9]] = k
        # parametrizations.weight.original0/1(torch>=2.1)罕见布局,不处理

    skip: set[str] = set()
    for base, gk in g_bases.items():
        vk = base + ".weight_v"
        if vk not in sd:
            continue  # 异常 ckpt,原样保留
        g = _to_numpy(sd[gk])
        v = _to_numpy(sd[vk])
        dim = overrides.get(base, 0)
        # torch weight_norm(dim=d):沿「除 d 外所有轴」求 L2 范数
        #   (O,C,K) dim=0 -> axis=(1,2);dim=2 -> axis=(0,1)
        norm_axes = tuple(i for i in range(v.ndim) if i != dim)
        norm = np.sqrt(np.sum(v * v, axis=norm_axes, keepdims=True) + 1e-12)
        # g 已是可广播形状(dim=2 时);dim=0 时 (out,) -> (out,1,...)
        if g.ndim == 1 and v.ndim > 1:
            g_shape = [1] * v.ndim
            g_shape[dim] = g.shape[0]
            g_b = g.reshape(g_shape)
        else:
            g_b = g
        out[base + ".weight"] = g_b * v / norm
        skip.add(gk)
        skip.add(vk)

    for k, val in sd.items():
        if k not in skip:
            out[k] = _to_numpy(val)
    return out


def _to_numpy(x):
    """torch.Tensor / numpy / mx.array → numpy float64(weight_norm 合并精度安全)。"""
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = np.array(x)
    arr = np.asarray(x)
    if arr.dtype != np.float64 and arr.dtype.kind == "f":
        arr = arr.astype(np.float64)
    return arr


# --- STFT(MLX 无内建,手写;帧网格红线) -----------------------------------

def hann_window(win_length: int):
    """torch.hann_window 逐位一致:**periodic**(分母 N,非 numpy.hanning 的
    symmetric)。usvc 所有 STFT 用 torch 公式 ``0.5 - 0.5*cos(2πn/N)``。
    ⚠ 与 stft.hann_window(np.hanning float64→f32)数学同、位不同,须共存。
    """
    import mlx.core as mx
    if win_length == 1:
        return mx.ones((1,))
    n = mx.arange(win_length)
    return 0.5 - 0.5 * mx.cos(2 * math.pi * n / win_length)


def frame_len_nv(L: int, hop: int) -> int:
    """nvSTFT center=False 帧数 ``(L - hop) // hop + 1``(与 frame_len_for 一致)。"""
    if L < hop:
        return 1
    return (L - hop) // hop + 1


def _unfold_frames(wav, win_length: int, hop: int):
    """1d wav → [n_frames, win_length] 滑窗(frames[i] = wav[i*hop : i*hop+win])。"""
    import mlx.core as mx
    n_frames = (wav.shape[0] - win_length) // hop + 1
    if n_frames <= 0:
        return mx.zeros((1, win_length), dtype=wav.dtype)
    idx = (mx.arange(n_frames) * hop)[:, None] + mx.arange(win_length)[None, :]
    return wav[idx]


def stft_nv(wav, n_fft: int, hop: int, win_length: int, window):
    """nvSTFT center=False STFT(wav 已由调用方补边)→ mag [n_fft//2+1, frames]。"""
    import mlx.core as mx
    frames = _unfold_frames(wav, win_length, hop) * window
    return mx.abs(mx.fft.rfft(frames, n=n_fft)).T  # [freq, frames]


def stft_center(wav, n_fft: int, hop: int, win_length: int, window,
                pad_mode: str = "reflect"):
    """center=True STFT(RMVPE 约定),两端各补 n_fft//2。

    ⚠ torch.stft(center=True) 默认 pad_mode='reflect',必须镜像补边而非零补
    (RMVPE mel 曾差 ~2.1,修复后 <1e-4)。信号短于 pad 量回退零补。
    """
    import mlx.core as mx
    pad = n_fft // 2
    if pad_mode == "reflect" and pad < wav.shape[-1]:
        padded = pad_reflect1d(wav, pad, pad)
    else:
        padded = mx.pad(wav, [(pad, pad)])
    frames = _unfold_frames(padded, win_length, hop) * window
    return mx.abs(mx.fft.rfft(frames, n=n_fft)).T


# --- F.interpolate 等价 -----------------------------------------------------

def interpolate_linear(x, size: int, align_corners: bool = True):
    """F.interpolate(mode='linear') 沿最后一维 [..., T] → [..., size]。

    align_corners=True: pos=linspace(0,T-1,size);False:
    pos=(arange+0.5)*T/size-0.5 边界 clamp(usvc 内部主要用 True)。
    """
    import mlx.core as mx
    T = x.shape[-1]
    if T == size:
        return x
    if size == 1:
        idx = mx.zeros((1,), dtype=mx.int32) if align_corners else mx.array([T // 2], dtype=mx.int32)
        return mx.take(x, idx, axis=-1)

    if align_corners:
        pos = mx.linspace(0, T - 1, size)
    else:
        scale = T / size
        pos = mx.maximum((mx.arange(size) + 0.5) * scale - 0.5, 0.0)
        pos = mx.minimum(pos, T - 1.0)

    idx_lo = mx.floor(pos).astype(mx.int32)
    idx_hi = mx.maximum(mx.minimum(idx_lo + 1, T - 1), 0)
    frac = pos - idx_lo.astype(mx.float32)
    a, b = mx.take(x, idx_lo, axis=-1), mx.take(x, idx_hi, axis=-1)
    return a * (1 - frac) + b * frac


def interpolate_nearest(x, scale_factor: float):
    """F.interpolate(mode='nearest') 沿最后一维(RVC SineGen rad_values 上采样)。"""
    import mlx.core as mx
    T = x.shape[-1]
    new_T = int(round(T * scale_factor))
    if new_T == T:
        return x
    idx = mx.minimum((mx.arange(new_T) / scale_factor).astype(mx.int32), T - 1)
    return mx.take(x, idx, axis=-1)


# --- reflect pad(mx.pad 仅 constant/edge) ----------------------------------

def pad_reflect1d(x, left: int, right: int):
    """F.pad(mode='reflect')(1d,最后一维;镜像不含边界点)。left/right < len。"""
    import mlx.core as mx
    if left == 0 and right == 0:
        return x
    parts = [x]
    if left > 0:
        parts.insert(0, x[left:0:-1])
    if right > 0:
        parts.append(x[-2:-right - 2:-1])
    return mx.concatenate(parts, axis=-1)


def pad_reflect1d_fallback(x, left: int, right: int):
    """nvstft_pad 语义:right >= len(wav)(短 clip)时回退 constant 零填充。"""
    import mlx.core as mx
    L = x.shape[-1]
    if right < L and left < L:
        return pad_reflect1d(x, left, right)
    return mx.pad(x, [(left, right)])


def masked_fill(x, mask, fill_value):
    """x.masked_fill(mask, fill_value):mask True 处填 fill(cond True 取 fill)。"""
    import mlx.core as mx
    return mx.where(mask, mx.full(x.shape, fill_value, dtype=x.dtype), x)


# --- E4 共享 kernel(原 per-backend verbatim 搬迁;layout 差异显式化) ---------

def norm_spec(x, spec_min: float, spec_max: float, cast: bool = False):
    """mel 归一化到 [-1,1];cast=True 先 f32(diffusion/shallow/vae 旧行为;
    ddsp6/rift_dit 旧代码无 astype → cast=False)。"""
    import mlx.core as mx
    if cast:
        x = x.astype(mx.float32)
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x, spec_min: float, spec_max: float):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


def ode_euler(v, x, t, dt, cond):
    x = x + v(x, 1000 * t, cond) * dt
    return x, t + dt


def ode_rk4(v, x, t, dt, cond):
    k1 = v(x, 1000 * t, cond)
    k2 = v(x + 0.5 * k1 * dt, 1000 * (t + 0.5 * dt), cond)
    k3 = v(x + 0.5 * k2 * dt, 1000 * (t + 0.5 * dt), cond)
    k4 = v(x + k3 * dt, 1000 * (t + dt), cond)
    x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return x, t + dt


def reflow_loss(v, x_1, t, cond, loss_type="l2_lognorm", mask=None, x_0=None,
                layout="btm"):
    """Flow-matching 训练 loss(ddsp6/reflow_shallow/reflow_vae 收敛)。

    layout "btm"=[B,T,M](mask [B,T]),"b1mt"=[B,1,M,T](ddsp6;mask [B,1,1,T]);
    两 layout 广播维与归一分母均不同(b1mt: m.sum()*M*T;btm: m.sum()*M),
    verbatim 保留各自原式。x_0 None 时内部采样(ddsp6/shallow);vae 外部传。
    """
    import mlx.core as mx
    b1mt = layout == "b1mt"
    if x_0 is None:
        x_0 = mx.random.normal(x_1.shape, dtype=x_1.dtype)
    t_b = t[:, None, None, None] if b1mt else t[:, None, None]
    x_t = x_0 + t_b * (x_1 - x_0)
    v_pred = v(x_t, 1000 * t, cond)

    if mask is not None:
        m = mask.astype(x_1.dtype)
        m = m[:, None, None, :] if b1mt else m[:, :, None]
        elem = mx.square(x_1 - x_0 - v_pred)
        if loss_type == "l2_lognorm":
            weights = 0.398942 / t / (1 - t) * mx.exp(
                -0.5 * mx.square(mx.log(t / (1 - t)))
            )
            w_b = weights[:, None, None, None] if b1mt else weights[:, None, None]
            elem = w_b * elem
        denom = (m.sum() * elem.shape[2] * elem.shape[3] if b1mt
                 else m.sum() * x_1.shape[2]) + 1e-8
        loss = (elem * m).sum() / denom
    elif loss_type == "l1":
        loss = mx.abs(x_1 - x_0 - v_pred).mean()
    elif loss_type == "l2":
        loss = mx.mean(mx.square(x_1 - x_0 - v_pred))
    elif loss_type == "l2_lognorm":
        weights = 0.398942 / t / (1 - t) * mx.exp(
            -0.5 * mx.square(mx.log(t / (1 - t)))
        )
        w_b = weights[:, None, None, None] if b1mt else weights[:, None, None]
        loss = mx.mean(w_b * mx.square(x_1 - x_0 - v_pred))
    else:
        raise NotImplementedError(loss_type)
    return loss


def sequence_mask(lengths, max_length=None):
    """[B] lengths → [B, T] bool(int32 arange;rvc/sovits 收敛)。"""
    import mlx.core as mx
    if max_length is None:
        max_length = int(lengths.max().item())
    return mx.arange(max_length, dtype=mx.int32)[None, :] < lengths[:, None]


def _pad_reflect_last(x, left: int, right: int):
    """reflect 沿最后一维(1D/2D 都正确;⚠ 不同于切 dim0 的 pad_reflect1d)。"""
    import mlx.core as mx
    if left == 0 and right == 0:
        return x
    parts = [x]
    if left > 0:
        parts.insert(0, x[..., left:0:-1])
    if right > 0:
        parts.append(x[..., -2:-right - 2:-1])
    return mx.concatenate(parts, axis=-1)


def stft_center_complex(wav, n_fft: int, hop: int, window, pad_mode: str = "reflect"):
    """torch.stft(center=True, return_complex=True):wav [B,L] → [B,F,L//hop+1]。"""
    import mlx.core as mx
    B, L = wav.shape
    pad = n_fft // 2
    if pad_mode == "reflect" and pad < L:
        padded = _pad_reflect_last(wav, pad, pad)
    else:
        padded = mx.pad(wav, [(0, 0), (pad, pad)])
    n_frames = (padded.shape[1] - n_fft) // hop + 1
    idx = mx.arange(n_frames)[:, None] * hop + mx.arange(n_fft)[None, :]
    frames = mx.take(padded, idx, axis=1) * window  # [B, n_frames, n_fft]
    return mx.fft.rfft(frames, n=n_fft, axis=-1).transpose(0, 2, 1)  # [B,F,n_frames]


def istft_center_complex(spec, n_fft: int, hop: int, window):
    """torch.istft(center=True):spec [B,F,n_frames] → [B, hop*(n_frames-1)]。"""
    import mlx.core as mx
    B, F, n_frames = spec.shape
    frames = mx.fft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1) * window
    expected = n_fft + hop * (n_frames - 1)
    y = mx.zeros((B, expected))
    win_env = mx.zeros((B, expected))
    w2 = window * window
    for i in range(n_frames):
        sl = slice(i * hop, i * hop + n_fft)
        y = y.at[:, sl].add(frames[:, i])
        win_env = win_env.at[:, sl].add(w2)
    start = n_fft // 2
    end = expected - n_fft // 2
    return y[:, start:end] / win_env[:, start:end]
