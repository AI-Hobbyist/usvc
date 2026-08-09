"""MLX 缺失算子的统一实现(MLX 0.32.0 适配层)。

本模块集中实现 usvc 各模型移植到 MLX 时缺失或语义不同的算子:
  - ``merge_weight_norm``: torch weight_norm (weight_g/weight_v) → 普通 weight
  - ``stft_*``: 复现 usvc nvSTFT 帧网格(center=False)与 RMVPE center=True 约定
  - ``interpolate_*``: F.interpolate(linear/nearest, align_corners) 等价
  - ``pad_reflect1d``: F.pad(mode='reflect') 等价

设计原则:数值与 torch 严格对齐(bit-exact 为目标),帧网格公式逐位复现
``features/mel.py::nvstft_pad`` 与 ``frame_len_for`` 的约定 —— 这是跨模型
一致性红线(见 scout 报告难点 #5)。

MLX 是惰性求值,本模块的函数返回 mx.array(未 eval),由调用方决定何时 eval。
"""
from __future__ import annotations

from typing import Literal

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
]


# ===========================================================================
# weight_norm 合并(torch state_dict 预处理,返回 numpy/torch-free dict)
# ===========================================================================

def merge_weight_norm(
    sd: dict,
    *,
    norm_dim_overrides: dict[str, int] | None = None,
) -> dict:
    """把 torch weight_norm 的 weight_g + weight_v 合并回普通 weight。

    torch ``weight_norm`` 公式::

        weight = g * v / ||v||_dim

    其中 ``g`` 是沿 ``dim`` 的范数方向(形状可广播到 v),``v`` 与最终
    weight 同形。合并后 MLX 侧只需普通 Linear/Conv,运行时零开销。

    Args:
        sd: torch state_dict(值可以是 torch.Tensor / numpy / 任何带 .shape
            和 .detach() 的对象)。**不会被修改**。
        norm_dim_overrides: ``{base_key: dim}`` 显式覆盖。usvc 绝大多数
            weight_norm 是默认 dim=0(输出通道),但 ContentVec pos_conv 用
            ``dim=2``(Conv1d weight (O,C,K) 的 kernel 维,见
            ``features/cvec/contentvec.py:90``)。未覆盖的键默认 dim=0。

    Returns:
        新 dict:weight_g/weight_v 被移除,新增合并后的 ``base.weight``。
        其余键原样保留(numpy 形式)。

    Note:
        本函数刻意不依赖 mlx/torch —— 它在 ckpt 桥接层调用,产出 numpy
        dict 后再统一转 mx.array(见 ``_convert.py``)。这样 weight_norm
        合成逻辑可被 torch 版本单元测试直接验证(数值 bit-exact)。
    """
    import numpy as np

    overrides = norm_dim_overrides or {}
    out: dict = {}
    # 收集所有 weight_g 键(同时兼容 parametrizations.weight.original0/1)
    g_bases: dict[str, str] = {}  # base -> g_key
    for k in sd:
        if k.endswith(".weight_g"):
            # ".weight_g" 是 9 个字符,k[:-9] 得到 base(如 "conv.weight" 的 "conv")
            g_bases[k[:-9]] = k
        # parametrizations.weight.original0 (g) / original1 (v) —— torch>=2.1
        # 罕见布局,这里只处理 weight_g/weight_v(usvc 实际 ckpt 格式)

    skip: set[str] = set()
    for base, gk in g_bases.items():
        vk = base + ".weight_v"
        if vk not in sd:
            # 没有 weight_v,保留原样(异常 ckpt)
            continue
        g = _to_numpy(sd[gk])
        v = _to_numpy(sd[vk])
        dim = overrides.get(base, 0)
        # torch weight_norm(dim=d):沿 d 维切分,每个切片是 v 中
        # **固定 d 索引、其他维任意** 的子张量,对其求 L2 范数。
        # 即 norm = ||v|| 沿「除 dim 外的所有轴」求和。
        #   例:weight (O,C,K) dim=0 -> 每个 (C,K) 切片求范数 -> axis=(1,2)
        #      weight (O,C,K) dim=2 -> 每个 (O,C) 切片求范数 -> axis=(0,1)
        norm_axes = tuple(i for i in range(v.ndim) if i != dim)
        norm = np.sqrt(np.sum(v * v, axis=norm_axes, keepdims=True) + 1e-12)
        # torch weight_norm 的 g(original0)已是可广播形状:
        #   dim=0 时 g 形状 (out,) —— 需 reshape 到 (out,1,1,...)
        #   dim=2 时 g 形状 (1,1,dim2_size) —— 已就绪,直接广播
        if g.ndim == 1 and v.ndim > 1:
            g_shape = [1] * v.ndim
            g_shape[dim] = g.shape[0]
            g_b = g.reshape(g_shape)
        else:
            g_b = g
        merged = g_b * v / norm
        out[base + ".weight"] = merged
        skip.add(gk)
        skip.add(vk)

    # 其余键(含已合并 base 的 bias 等)原样转 numpy
    for k, val in sd.items():
        if k in skip:
            continue
        out[k] = _to_numpy(val)
    return out


def _to_numpy(x):
    """torch.Tensor / numpy / mx.array → numpy float64(合并用,精度安全)。"""
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):  # mx.array
        x = np.array(x)
    arr = np.asarray(x)
    # weight_norm 合并对精度敏感,统一用 float64 计算,加载时再降精度
    if arr.dtype != np.float64 and arr.dtype.kind == "f":
        arr = arr.astype(np.float64)
    return arr


# ===========================================================================
# STFT —— MLX 无内建,手写(帧网格红线)
# ===========================================================================

import math


def hann_window(win_length: int):
    """与 ``torch.hann_window`` 逐位一致的 **periodic**(DFT-even)hann 窗。

    torch 公式: ``0.5 - 0.5 * cos(2π n / N)``, n = 0..N-1(N=win_length)。
    注意是 **periodic**(分母 N),不是 ``numpy.hanning`` 的 symmetric(分母 N-1)。
    usvc 所有 STFT 通路用 ``torch.hann_window``,故 mlx 必须对齐 periodic 公式
    (帧网格红线的一部分,已由 torch vs mlx 数值测试验证)。
    """
    import mlx.core as mx
    if win_length == 1:
        return mx.ones((1,))
    n = mx.arange(win_length)
    return 0.5 - 0.5 * mx.cos(2 * math.pi * n / win_length)


def frame_len_nv(L: int, hop: int) -> int:
    """nvSTFT center=False 帧数公式: ``(L - hop) // hop + 1``。

    与 ``features/mel.py`` / ``io/audio.py`` 的 ``frame_len_for`` 逐位一致。
    所有 mel/rms/spec 提取器都依赖这个帧计数 —— MLX 侧必须用同一公式。
    """
    if L < hop:
        return 1
    return (L - hop) // hop + 1


def _unfold_frames(wav, win_length: int, hop: int):
    """把 1d wav 切成 [n_frames, win_length] 的滑动窗(无 overlap 求和)。

    等价 torch 的 as_strided / sliding_window。MLX 用 reshape+取切片实现。
    """
    import mlx.core as mx
    n_frames = (wav.shape[0] - win_length) // hop + 1
    if n_frames <= 0:
        return mx.zeros((1, win_length), dtype=wav.dtype)
    # 用 take + 索引矩阵:frames[i, :] = wav[i*hop : i*hop + win_length]
    starts = mx.arange(n_frames) * hop               # [n_frames]
    offsets = mx.arange(win_length)                  # [win_length]
    idx = starts[:, None] + offsets[None, :]         # [n_frames, win_length]
    return wav[idx]


def stft_nv(wav, n_fft: int, hop: int, win_length: int, window):
    """nvSTFT ``center=False`` 约定的 STFT(复现 usvc ``features/mel.py``)。

    输入 wav 已按 ``pad_reflect1d_fallback`` 补边(调用方负责,见
    ``MelExtractor`` 逻辑)。返回 magnitude ``[n_fft//2+1, n_frames]``,
    与 ``torch.stft(...).abs()`` 布局一致(freq, frames)。

    Args:
        wav: 1d mx.array(已补边)。
        window: hann 窗 mx.array,长度 = win_length。
    """
    import mlx.core as mx
    frames = _unfold_frames(wav, win_length, hop)    # [n_frames, win_length]
    frames = frames * window                          # hann 加窗
    # rfft 沿最后一维,n_fft zero-pad 到 n_fft(若 win_length < n_fft)
    spec = mx.fft.rfft(frames, n=n_fft)               # [n_frames, n_fft//2+1] complex
    # magnitude = |spec| = sqrt(re^2 + im^2)
    mag = mx.abs(spec)
    # 转置到 [freq, frames] 对齐 torch.stft 输出
    return mag.T


def stft_center(wav, n_fft: int, hop: int, win_length: int, window):
    """``center=True`` 约定的 STFT(复现 RMVPE ``spec.py``)。

    center=True: 两端各补 n_fft//2 零,帧数 = 1 + (L + n_fft - n_fft) // hop
    (近似 L//hop + 1)。RMVPE 用这个约定,与 FCPE/usvc mel 的 center=False 不同。
    """
    import mlx.core as mx
    pad = n_fft // 2
    padded = mx.pad(wav, [(pad, pad)])                # 两端零补
    frames = _unfold_frames(padded, win_length, hop)
    frames = frames * window
    spec = mx.fft.rfft(frames, n=n_fft)
    mag = mx.abs(spec)
    return mag.T                                      # [freq, frames]


# ===========================================================================
# F.interpolate 等价(MLX 无 interpolate)
# ===========================================================================

def interpolate_linear(x, size: int, align_corners: bool = True):
    """``F.interpolate(x, size, mode='linear', align_corners=True)`` 等价。

    沿最后一维插值。x: ``[..., T]`` → ``[..., size]``。

    align_corners=True(usvc 推理侧 f0/units 对齐用):端点对齐,
    位置 = linspace(0, T-1, size)。

    align_corners=False:位置 = (arange(size)+0.5)*T/size - 0.5(可能越界,
    torch 用边界 clamp)。usvc 内部主要用 True,这里完整实现两种。
    """
    import mlx.core as mx
    T = x.shape[-1]
    if T == size:
        return x
    if size == 1:
        # torch: align_corners=True 时 size=1 取 index 0;False 取中点
        idx = mx.zeros((1,), dtype=mx.int32) if align_corners else mx.array([T // 2], dtype=mx.int32)
        return mx.take(x, idx, axis=-1)

    if align_corners:
        pos = mx.linspace(0, T - 1, size)
    else:
        # torch 公式:center = (arange + 0.5) * scale - 0.5, scale = T/size
        scale = T / size
        pos = (mx.arange(size) + 0.5) * scale - 0.5
        pos = mx.maximum(pos, 0.0)
        pos = mx.minimum(pos, T - 1.0)

    idx_lo = mx.floor(pos).astype(mx.int32)
    idx_hi = mx.minimum(idx_lo + 1, T - 1)
    idx_hi = mx.maximum(idx_hi, 0)
    frac = pos - idx_lo.astype(mx.float32)            # [..., size]
    # 广播 gather:take 沿最后一维
    a = mx.take(x, idx_lo, axis=-1)
    b = mx.take(x, idx_hi, axis=-1)
    # frac 广播到 a 的形状(已是 size 维)
    return a * (1 - frac) + b * frac


def interpolate_nearest(x, scale_factor: float):
    """``F.interpolate(x, scale_factor=sf, mode='nearest')`` 等价。

    沿最后一维最近邻重复。x: ``[..., T]`` → ``[..., int(T*sf)]``。
    用于 RVC SineGen 的 rad_values 上采样。
    """
    import mlx.core as mx
    T = x.shape[-1]
    new_T = int(round(T * scale_factor))
    if new_T == T:
        return x
    # nearest: index = floor(arange(new_T) / sf),clamp 到 [0, T-1]
    idx = (mx.arange(new_T) / scale_factor).astype(mx.int32)
    idx = mx.minimum(idx, T - 1)
    return mx.take(x, idx, axis=-1)


# ===========================================================================
# pad reflect(MLX mx.pad 仅 constant/edge)
# ===========================================================================

def pad_reflect1d(x, left: int, right: int):
    """``F.pad(x, (left, right), mode='reflect')`` 等价(1d,沿最后一维)。

    reflect 语义:镜像不含边界点。例 [a,b,c,d], left=2 → [c,b,a,b,c,d]。
    当 pad 量 >= 信号长度时,torch 抛错;usvc 的 nvstft_pad 会先 fallback
    到 constant(见 ``pad_reflect1d_fallback``)。本函数假设 left/right 均
    < len(调用方负责 fallback 判断)。
    """
    import mlx.core as mx
    if left == 0 and right == 0:
        return x
    parts = [x]
    # 左侧:x[left:0:-1] 即翻转 x[1..left](不含 x[0])
    if left > 0:
        # torch reflect 左侧 = x[left:0:-1] = [x[left], x[left-1], ..., x[1]]
        lo = x[left:0:-1]
        parts.insert(0, lo)
    if right > 0:
        # 右侧:x[-2:-right-2:-1] = [x[-2], x[-3], ..., x[-right-1]]
        hi = x[-2:-right - 2:-1]
        parts.append(hi)
    return mx.concatenate(parts, axis=-1)


def pad_reflect1d_fallback(x, left: int, right: int):
    """nvSTFT 的 reflect+constant fallback(复现 ``nvstft_pad`` 逻辑)。

    当 pad_right >= len(wav) 时(短 clip),reflect 无定义,回退 constant(零)。
    与 ``features/mel.py::nvstft_pad`` 逐行对应::

        mode = "reflect" if pad_right < wav.shape[-1] else "constant"
    """
    import mlx.core as mx
    L = x.shape[-1]
    use_reflect = right < L
    if use_reflect and left < L:
        # 两端都能 reflect
        return pad_reflect1d(x, left, right)
    # fallback: constant 零填充
    return mx.pad(x, [(left, right)])


def masked_fill(x, mask, fill_value):
    """``x.masked_fill(mask, fill_value)`` 等价(torch 语义)。

    torch: **mask 为 True 处填 fill_value**,False 处保留 x。
    (usvc contentvec 用法 ``attn.masked_fill(~m, -inf)`` 传的是 ~m,
    即 padding 位置 True → 填 -inf。)

    mlx: mx.where(mask, fill, x) —— cond True 取 fill,False 取 x。

    Args:
        mask: bool mx.array,广播到 x 的形状。
    """
    import mlx.core as mx
    return mx.where(mask, mx.full(x.shape, fill_value, dtype=x.dtype), x)
