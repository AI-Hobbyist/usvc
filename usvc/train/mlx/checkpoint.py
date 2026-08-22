"""MLX 训练 checkpoint:torch ckpt → mlx 权重、mlx 训练产物保存/恢复。
torch ckpt 委托 backend from_ckpt(key 映射/reshape);mlx 产物 .npz(扁平
path→array)+ .json(step/epoch/history),可直接被推理侧加载。"""
from __future__ import annotations
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as mopt
__all__ = ["save_mlx_checkpoint", "load_mlx_checkpoint", "load_torch_weights",
           "torch_ckpt_to_mlx"]
def save_mlx_checkpoint(model, path: str | Path, step: int = 0, epoch: int = 0,
                        extra: dict | None = None) -> Path:  # → .npz 路径
    import json
    import numpy as np
    path = Path(path)
    np.savez(str(path),
             **{k: np.array(v) for k, v in mopt.tree_flatten(model.trainable_parameters())})
    meta = {"step": step, "epoch": epoch, **(extra or {})}
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return path
def load_mlx_checkpoint(model, path: str | Path, strict: bool = True) -> dict:
    """.npz → mlx 权重;返回 meta dict。"""
    import json
    import numpy as np
    path = Path(path)
    data = np.load(str(path))
    model.update(mopt.tree_unflatten([(k, mx.array(data[k])) for k in data.files]))
    meta = {}
    jpath = path.with_suffix(".json")
    if jpath.exists():
        meta = json.loads(jpath.read_text())
    return meta
def load_torch_weights(model, ckpt_path: str | Path, strict: bool = True):
    """torch ckpt / mlx npz → mlx 权重(委托 model.from_ckpt)。"""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".npz":
        return load_mlx_checkpoint(model, ckpt_path)
    if hasattr(model, "from_ckpt"):
        return model.from_ckpt(str(ckpt_path))
    raise ValueError(f"model has no from_ckpt and ckpt is not .npz: {ckpt_path}")
def torch_ckpt_to_mlx(model, torch_ckpt: str | Path, out_npz: str | Path):
    load_torch_weights(model, torch_ckpt)
    return save_mlx_checkpoint(model, out_npz)
