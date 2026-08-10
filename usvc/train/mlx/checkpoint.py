"""MLX 训练 checkpoint:torch ckpt → mlx 权重、mlx 训练产物保存/恢复。

- torch ckpt(.pth/.pt/.ckpt):委托 backend 模型的 from_ckpt(已有,
  处理 key 映射/reshape),作为训练起点(继续训练)。
- mlx 训练产物:.npz(扁平 path→array)+ .json(step/epoch/history)。

兼容性:mlx 训练的产物(.npz)可直接被 load_pretrained_mlx 加载,
即 mlx 训练 → mlx 推理 的权重通路。
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.optimizers as mopt
import numpy as np

__all__ = [
    "save_mlx_checkpoint",
    "load_mlx_checkpoint",
    "load_torch_weights",
    "torch_ckpt_to_mlx",
]


def save_mlx_checkpoint(
    model, path: str | Path, step: int = 0, epoch: int = 0, extra: dict | None = None
) -> Path:
    """保存 mlx 模型参数为 .npz + .json。返回 .npz 路径。"""
    import json

    path = Path(path)
    params = model.trainable_parameters()
    arrs = {k: np.array(v) for k, v in mopt.tree_flatten(params)}
    np.savez(str(path), **arrs)
    meta = {"step": step, "epoch": epoch, **(extra or {})}
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return path


def load_mlx_checkpoint(model, path: str | Path, strict: bool = True) -> dict:
    """从 .npz 加载权重到 mlx 模型。返回 meta dict(step/epoch 等)。"""
    import json

    path = Path(path)
    data = np.load(str(path))
    leaves = [(k, mx.array(data[k])) for k in data.files]
    params = mopt.tree_unflatten(leaves)
    model.update(params)
    meta = {}
    jpath = path.with_suffix(".json")
    if jpath.exists():
        meta = json.loads(jpath.read_text())
    return meta


def load_torch_weights(model, ckpt_path: str | Path, strict: bool = True):
    """torch ckpt → mlx 模型权重。委托 model.from_ckpt(backend 实现)。"""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".npz":
        return load_mlx_checkpoint(model, ckpt_path)
    if hasattr(model, "from_ckpt"):
        return model.from_ckpt(str(ckpt_path))
    raise ValueError(f"model has no from_ckpt and ckpt is not .npz: {ckpt_path}")


def torch_ckpt_to_mlx(model, torch_ckpt: str | Path, out_npz: str | Path):
    """转换 torch ckpt 为 mlx .npz 权重文件(推理/训练通用)。"""
    load_torch_weights(model, torch_ckpt)
    return save_mlx_checkpoint(model, out_npz)
