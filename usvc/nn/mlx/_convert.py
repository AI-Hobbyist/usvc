"""torch ckpt → MLX 权重桥接:解包 wrapper(io.ckpt)→ weight_norm 合并
(numpy float64)→ 剥 _orig_mod → mx.array + Conv 转置(channels-last)。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = [
    "load_torch_sd",
    "to_mlx_sd",
    "transpose_conv_weights",
    "load_torch_ckpt_to_mlx",
    "strip_prefix",
    "CONTENTVEC_WEIGHT_NORM_DIMS",
]

# ContentVec pos_conv 是 usvc 唯一非默认 dim 的 weight_norm((O,C,K) kernel 维)
CONTENTVEC_WEIGHT_NORM_DIMS = {
    "encoder.pos_conv_embed.conv": 2,
}


def load_torch_sd(ckpt_path: str | Path) -> dict:
    """torch ckpt → numpy state_dict(wrapper 解包复用 io.ckpt.load_ckpt_state)。"""
    import numpy as np
    from ...io.ckpt import load_ckpt_state  # 复用现有入口(torch.load)
    out = {}
    for k, v in load_ckpt_state(ckpt_path).items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        out[k] = np.asarray(v)
    return out


def to_mlx_sd(
    np_sd: dict,
    *,
    conv_transpose_keys: Iterable[str] | None = None,
    convtranspose_keys: Iterable[str] | None = None,
    key_remap: dict[str, str] | None = None,
    strip_orig_mod: bool = True,
    dtype=None,
):
    """numpy sd → mx.array sd;Conv (O,C,..)→(O,..,C),ConvTranspose 规则不同。"""
    import mlx.core as mx
    transpose_set = set(conv_transpose_keys or ())
    ctranspose_set = set(convtranspose_keys or ())
    remap = key_remap or {}
    out = {}
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
    """启发式扫描 Conv 权重键(ndim==3/4 且 .weight 结尾;Linear 2d 不转置;
    weight_g 已被 merge 移除)。调用方可用精确 key 列表覆盖。"""
    conv_keys = set()
    for k, v in np_sd.items():
        if not k.endswith(".weight"):
            continue
        nd = v.ndim if hasattr(v, "ndim") else 0
        if nd in (3, 4):
            conv_keys.add(k)
    return conv_keys


def _transpose_one_conv(key: str, arr, ndim: int):
    """Conv: torch (O,C,K)/(O,C,kH,kW) → mlx (O,K,C)/(O,kH,kW,C);groups>1 同。"""
    import mlx.core as mx
    if ndim == 3:
        return mx.transpose(arr, (0, 2, 1))
    if ndim == 4:
        return mx.transpose(arr, (0, 2, 3, 1))
    return arr  # Conv3d/未知,原样返回


def _transpose_conv_transpose(arr, ndim: int):
    """ConvTranspose: torch (I,O,K)/(I,O,kH,kW) → mlx (O,K,I)/(O,kH,kW,I)。"""
    import mlx.core as mx
    if ndim == 3:
        return mx.transpose(arr, (1, 2, 0))
    if ndim == 4:
        return mx.transpose(arr, (1, 2, 3, 0))
    return arr


def strip_prefix(sd: dict, prefix: str) -> dict:
    """纯字符串剥前缀(仅保留 prefix 开头的键);prefix="" 原样返回。
    前缀探测/回退逻辑是各 from_ckpt 的冻结契约,不在此处。"""
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
    """端到端 torch ckpt(.pth/.pt/.ckpt)/ mlx 训练产物(.npz)→ mlx state_dict。

    .npz 直接加载(扁平 path→array);否则 merge_weight_norm → 确定 Conv 转置键
    (ConvTranspose 从 conv_keys 排除避免双重转置)→ to_mlx_sd。
    """
    if str(ckpt_path).endswith(".npz"):
        import mlx.core as mx
        import numpy as np
        data = np.load(str(ckpt_path))
        sd = {k: mx.array(data[k]) for k in data.files}
        if dtype is not None and dtype != mx.float32:
            sd = {k: v.astype(dtype) for k, v in sd.items()}
        return sd
    from ._ops import merge_weight_norm
    merged = merge_weight_norm(load_torch_sd(ckpt_path),
                               norm_dim_overrides=weight_norm_dims or {})
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
