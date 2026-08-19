"""torch checkpoint → MLX 权重桥接。

usvc 所有 ckpt 是 torch pickle(``torch.load``,见 ``io/ckpt.py``),含
lightning/so-vits/RVC/裸权重四种顶层 wrapper。本模块:

1. 复用 ``io/ckpt.load_ckpt_state`` 解包顶层 wrapper(state_dict/model/weight)。
2. ``_ops.merge_weight_norm`` 把 weight_g/weight_v 合并成普通 weight。
3. 剥 ``_orig_mod.`` 前缀(torch.compile 包装残留)。
4. torch/numpy → mx.array,Conv 权重按需转置(channels-last)。

设计:本模块**只依赖 torch 做 pickle 反序列化**(CPU 侧,一次性),产出
mx.array dict 后 torch 即可释放。长期目标是给 hub 加 safetensors 镜像,
彻底去 torch(见 scout 报告难点 #6)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = [
    "load_torch_sd",
    "to_mlx_sd",
    "transpose_conv_weights",
    "load_torch_ckpt_to_mlx",
    "CONTENTVEC_WEIGHT_NORM_DIMS",
]


# ContentVec pos_conv 用 dim=2(Conv1d weight (O,C,K) 的 kernel 维),
# 是 usvc 唯一非默认 dim 的 weight_norm(见 features/cvec/contentvec.py:90)。
# 其余 weight_norm 全是默认 dim=0(输出通道维)。
CONTENTVEC_WEIGHT_NORM_DIMS = {
    "encoder.pos_conv_embed.conv": 2,
}


# ---------------------------------------------------------------------------
# Step 1: torch pickle → numpy state_dict(解包 wrapper)
# ---------------------------------------------------------------------------

def load_torch_sd(ckpt_path: str | Path) -> dict:
    """加载 torch ckpt,解包顶层 wrapper,返回 numpy state_dict。

    复用 ``usvc.io.ckpt.load_ckpt_state`` 的 wrapper 解包逻辑(lightning
    state_dict / so-vits model / RVC weight / 裸 dict),消除
    ``infer/engine.py:150-165`` 的重复(scout 指出的重构机会)。

    返回的 dict 值为 numpy float32(weight_norm 合并前保持原精度,
    合并在 _ops.merge_weight_norm 内部用 float64)。
    """
    import numpy as np
    # 复用现有 ckpt 入口(它会 import torch 做 torch.load)
    from ...io.ckpt import load_ckpt_state
    sd = load_ckpt_state(ckpt_path)
    # 统一转 numpy(此时可能是 torch.Tensor)
    out = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        out[k] = np.asarray(v)
    return out


# ---------------------------------------------------------------------------
# Step 2: numpy sd → mx.array sd(Conv 权重转置 + key 清理)
# ---------------------------------------------------------------------------

def to_mlx_sd(
    np_sd: dict,
    *,
    conv_transpose_keys: Iterable[str] | None = None,
    convtranspose_keys: Iterable[str] | None = None,
    key_remap: dict[str, str] | None = None,
    strip_orig_mod: bool = True,
    dtype=None,
):
    """numpy state_dict → mx.array state_dict。

    Args:
        np_sd: numpy 值的 dict。
        conv_transpose_keys: 普通 Conv(Conv1d/2d)权重键,转置 (O,C,..)→(O,..,C)。
        convtranspose_keys: **ConvTranspose**(ConvTranspose1d/2d)权重键,
            torch (I,O,..) → mlx (O,..,I),转置规则不同(见 _transpose_conv_transpose)。
        key_remap: {torch_key: mlx_key} 显式重映射。
        strip_orig_mod: 剥 ``._orig_mod.``。
        dtype: mx.float16 / bfloat16 / None(fp32)。
    """
    import mlx.core as mx
    out = {}
    transpose_set = set(conv_transpose_keys or ())
    ctranspose_set = set(convtranspose_keys or ())
    remap = key_remap or {}
    for k, v in np_sd.items():
        mk = remap.get(k, k)
        if strip_orig_mod:
            mk = mk.replace("._orig_mod.", ".")
        arr = mx.array(v)
        if dtype is not None:
            arr = arr.astype(dtype)
        if mk in ctranspose_set:
            arr = _transpose_conv_transpose(arr, np_sd[k].ndim)
        elif mk in transpose_set:
            arr = _transpose_one_conv(mk, arr, np_sd[k].ndim)
        out[mk] = arr
    return out


def transpose_conv_weights(np_sd: dict) -> set[str]:
    """扫描 state_dict,返回所有 Conv 权重键(需在 to_mlx_sd 时转置)。

    启发式:权重 ndim==3 且键以 .weight 结尾(Conv1d (O,C,K))→ 转置。
    ndim==4(Conv2d)→ 转置。ndim==2(Linear)不转置。

    更稳妥的方式是按 mlx 模块结构精确匹配,但 usvc 的 ckpt 键命名规整
    (conv/ups/resblocks/conv_pre 都含 conv 子串),启发式足够。**调用方
    可用精确 key 列表覆盖**(传 conv_transpose_keys)。
    """
    conv_keys = set()
    for k, v in np_sd.items():
        if not k.endswith(".weight"):
            continue
        arr = v
        nd = arr.ndim if hasattr(arr, "ndim") else 0
        if nd == 3:
            # Conv1d (O,C,K);排除 Linear(已是 2d)和 weight_g(标量/向量)
            # 但 weight_g 已被 merge_weight_norm 移除;此处安全
            conv_keys.add(k)
        elif nd == 4:
            # Conv2d (O,C,kH,kW)
            conv_keys.add(k)
    return conv_keys


def _transpose_one_conv(key: str, arr, ndim: int):
    """单个 Conv 权重转置:torch (O,C,...) → mlx (O,...,C)。

    Conv1d: (O,C,K) → (O,K,C)  即 transpose(0,2,1)
    Conv2d: (O,C,kH,kW) → (O,kH,kW,C)  即 transpose(0,2,3,1)
    分组卷积 groups>1 时,C 维是 C/g,转置规则相同。
    """
    import mlx.core as mx
    if ndim == 3:
        # (O,C,K) -> (O,K,C)
        return mx.transpose(arr, (0, 2, 1))
    elif ndim == 4:
        # (O,C,kH,kW) -> (O,kH,kW,C)
        return mx.transpose(arr, (0, 2, 3, 1))
    # Conv3d 或未知,原样返回(移植时按需扩展)
    return arr


def _transpose_conv_transpose(arr, ndim: int):
    """ConvTranspose 权重转置(输入/输出通道维度互换)。

    torch ConvTranspose1d (I,O,K) → mlx (O,K,I): transpose(1,2,0)
    torch ConvTranspose2d (I,O,kH,kW) → mlx (O,kH,kW,I): transpose(1,2,3,0)
    """
    import mlx.core as mx
    if ndim == 3:
        return mx.transpose(arr, (1, 2, 0))
    elif ndim == 4:
        return mx.transpose(arr, (1, 2, 3, 0))
    return arr


# ---------------------------------------------------------------------------
# 一站式:torch ckpt path → mlx-ready state_dict
# ---------------------------------------------------------------------------

def strip_prefix(sd: dict, prefix: str) -> dict:
    """机械剥离顶层前缀(仅删 ``prefix`` 开头的键,其余丢弃)。

    前缀探测/回退逻辑由各 from_ckpt 自行保留(探测顺序是冻结契约,
    见 backends/*.py);此处只做纯字符串剥离,prefix="" 时原样返回。
    """
    if not prefix:
        return dict(sd)
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_torch_ckpt_to_mlx(
    ckpt_path: str | Path,
    *,
    weight_norm_dims: dict[str, int] | None = None,
    key_remap: dict[str, str] | None = None,
    dtype=None,
    conv_keys: set[str] | None = None,
    convtranspose_keys: set[str] | None = None,
) -> dict:
    """端到端:torch ckpt / mlx npz → mlx state_dict(可直接 load_weights)。

    Args:
        ckpt_path: torch ckpt(.pth/.pt/.ckpt)或 mlx 训练产物(.npz)。
            .npz 直接加载(键已是扁平 path 形式,mlx tree_flatten 布局)。
        weight_norm_dims: ContentVec 传 ``CONTENTVEC_WEIGHT_NORM_DIMS``;
        key_remap: 裸键 → 前缀键映射。
        dtype: 目标精度(None=fp32, mx.float16, mx.bfloat16)。
        conv_keys: 普通 Conv 权重键集合(覆盖启发式);None 则自动扫描。
        convtranspose_keys: ConvTranspose 权重键集合(用不同转置规则)。
    """
    if str(ckpt_path).endswith(".npz"):
        # mlx 训练产物:扁平 path→array,直接返回
        import mlx.core as mx
        import numpy as np
        data = np.load(str(ckpt_path))
        sd = {k: mx.array(data[k]) for k in data.files}
        if dtype is not None and dtype != mx.float32:
            sd = {k: v.astype(dtype) for k, v in sd.items()}
        return sd
    np_sd = load_torch_sd(ckpt_path)
    # weight_norm 合并(在 numpy 侧,精度安全)
    from ._ops import merge_weight_norm
    merged = merge_weight_norm(np_sd, norm_dim_overrides=weight_norm_dims or {})
    # 确定 Conv 转置键(ConvTranspose 从 conv_keys 排除,避免双重转置)
    if conv_keys is None:
        conv_keys = transpose_conv_weights(merged)
    if convtranspose_keys:
        conv_keys = conv_keys - convtranspose_keys
    return to_mlx_sd(
        merged,
        conv_transpose_keys=conv_keys,
        convtranspose_keys=convtranspose_keys,
        key_remap=key_remap,
        dtype=dtype,
    )
