"""MLX 训练数据加载:复用 torch Dataset(.pt 由 torch.load 读取),collate 后
把 SVCBatch 转成 mlx array dict(batch 字段名与 torch SVCBatch 一致)供
loss_fn(model, batch) 消费。"""
from __future__ import annotations
from typing import Iterator
import mlx.core as mx
__all__ = ["MlxBatch", "mlx_collate", "mlx_dataloader"]
def _prop(name, required=False, default=None):
    def getter(self):
        return self[name] if required else self.get(name, default)
    return property(getter)
class MlxBatch(dict):  # key 与 torch SVCBatch 字段一致
    pass
for _n, _req, _d in (
    ("units", True, None), ("f0", True, None), ("frame_len", True, None),
    ("mel", False, None), ("wav", False, None), ("rms", False, None),
    ("spk_id", False, None), ("uv", False, None), ("aug_shift", False, None),
    ("pitch_token", False, None), ("key", False, []),
):
    setattr(MlxBatch, _n, _prop(_n, _req, _d))
def _to_mx(t):  # torch tensor → mlx array(经 numpy)
    import numpy as np
    if t is None:
        return None
    return mx.array(np.asarray(t.detach().cpu()))
def mlx_collate(batch: dict) -> MlxBatch:  # SVCBatch(torch) → MlxBatch(mlx array)
    out = MlxBatch()
    for k, v in batch.__dict__.items():
        if k == "key":
            out["key"] = list(v)
        elif isinstance(v, list):
            out[k] = v  # list 字段原样保留(不转 mlx)
        else:
            out[k] = _to_mx(v)
    return out
def mlx_dataloader(dataset, batch_size: int = 16, shuffle: bool = True,
                   num_workers: int = 0, drop_last: bool = True) -> Iterator[MlxBatch]:
    """包 torch DataLoader(仅数据搬运),yield MlxBatch。
    注意:num_workers>0 时 worker 进程需可导入 torch(数据加载进程)。"""
    from torch.utils.data import DataLoader
    from ..dataset import svc_collate
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=drop_last,
                        collate_fn=svc_collate, persistent_workers=num_workers > 0)
    for batch in loader:
        yield mlx_collate(batch)
