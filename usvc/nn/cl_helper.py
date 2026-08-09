"""channels_last (NHWC) 数据流的 conv1d 等价变换 helper。

把 1D conv/transposed-conv 当作 2D conv 的特例（H=1 维度），让权重布局落在
channels_last 上，从而复用 cuDNN NHWC fast path，消除逐层 nchw<->nhwc 转换。

数值契约：``_wn_weight`` 必须用 ``torch._weight_norm``（与模块 forward 的
compute_weight 完全一致）；手写 ``v*(g/||v||)`` 在 fp32 下偏差可达 0.16
（torch 内部实现不同）。推理时权重冻结，按 ``(conv 对象, _version)`` 缓存避免
每次 conv 重复归一化——不用 ``id()`` 作 key（对象销毁后 id 复用会把缓存错配
给新 conv）。

被 rvc/sovits Generator、NSF-HiFiGAN vocoder、diffusion WaveNet 共享。
engine 创建实例时调用 :func:`clear_wn_cache` 从干净状态开始。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "WN_CACHE",
    "wn_weight",
    "conv2d_cl",
    "convT2d_cl",
    "clear_wn_cache",
]

# key = (conv 对象, weight_v._version, weight_g._version)。
# conv 对象本身被持有（强引用），随 engine 释放而自然失效；_version 变化时
#（罕见，仅权重原地修改）自动失效重算。
WN_CACHE: dict[tuple[object, int, int], torch.Tensor] = {}


def wn_weight(conv: torch.nn.Module) -> torch.Tensor:
    """weight_norm 包装 conv 的归一化权重（否则返回原始权重）。

    必须用 torch._weight_norm（与模块 forward 的 compute_weight 完全一致）；
    手写 v*(g/||v||) 在 fp32 下偏差达 0.16（torch 内部实现不同）。
    推理时权重冻结，按 (conv 对象, _version) 缓存避免每次 conv 重复归一化；
    不用 id() 作 key——对象销毁后 id 复用会把缓存错配给新 conv。
    """
    g = getattr(conv, "weight_g", None)
    v = getattr(conv, "weight_v", None)
    if g is None or v is None:
        return conv.weight  # 未 weight_norm 的 conv（如 bias-only 路径）

    key = (conv, v._version, g._version)
    w = WN_CACHE.get(key)
    if w is None:
        # torch._weight_norm(v, g, 0) 与 ParametrizedWeightNorm.forward 完全一致。
        w = torch._weight_norm(v, g, 0)
        WN_CACHE[key] = w
    return w


def conv2d_cl(x4: torch.Tensor, conv: torch.nn.Module) -> torch.Tensor:
    """Conv1d -> Conv2d(kernel_height=1) on channels_last [B,C,1,T].

    权重 [C_out, C_in, K] -> [C_out, C_in, 1, K]，输入已是 [B,C,1,T]。
    stride 透传（noise_convs 有非 1 stride）。padding 沿时间轴。
    """
    w = wn_weight(conv)[:, :, None, :]
    return F.conv2d(
        x4, w, conv.bias,
        stride=(1, conv.stride[0]),
        padding=(0, conv.padding[0]),
        dilation=(1, conv.dilation[0]),
    )


def convT2d_cl(x4: torch.Tensor, conv: torch.nn.Module) -> torch.Tensor:
    """ConvTranspose1d -> ConvTranspose2d(1,k) on channels_last [B,C,1,T]."""
    w = wn_weight(conv)[:, :, None, :]
    return F.conv_transpose2d(
        x4, w, conv.bias,
        stride=(1, conv.stride[0]),
        padding=(0, conv.padding[0]),
        output_padding=(0, conv.output_padding[0]),
        dilation=(1, conv.dilation[0]),
    )


def clear_wn_cache() -> None:
    """清空 weight_norm 缓存（engine 新建实例时调用，保证从干净状态开始）。"""
    WN_CACHE.clear()
