"""MLX 训练数据加载。

复用 torch Dataset(特征文件 .pt 由 torch.load 读取,数据加载不含
autograd,不是训练核心),collate 后把 SVCBatch 转成 mlx array dict,
供 MlxTrainer 的 loss_fn(model, batch) 消费。

batch 结构(match torch SVCBatch):
    units [B, T, D], f0 [B, T], mel [B, T, M]?, wav [B, L]?,
    rms [B, T]?, spk_id [B]?, uv [B, T]?, aug_shift [B]?,
    frame_len [B], pitch_token [B, T]?
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import mlx.core as mx

__all__ = ["MlxBatch", "mlx_collate", "mlx_dataloader"]


class MlxBatch(dict):
    """mlx array 批处理(dict 子类,key 与 torch SVCBatch 字段一致)。"""

    @property
    def units(self):
        return self["units"]

    @property
    def f0(self):
        return self["f0"]

    @property
    def frame_len(self):
        return self["frame_len"]

    @property
    def mel(self):
        return self.get("mel")

    @property
    def wav(self):
        return self.get("wav")

    @property
    def rms(self):
        return self.get("rms")

    @property
    def spk_id(self):
        return self.get("spk_id")

    @property
    def uv(self):
        return self.get("uv")

    @property
    def aug_shift(self):
        return self.get("aug_shift")

    @property
    def pitch_token(self):
        return self.get("pitch_token")

    @property
    def key(self):
        return self.get("key", [])


def _to_mx(t):
    """torch tensor → mlx array(经 numpy)。"""
    import numpy as np
    if t is None:
        return None
    return mx.array(np.asarray(t.detach().cpu()))


def mlx_collate(batch: dict) -> MlxBatch:
    """SVCBatch(torch)→ MlxBatch(mlx array)。"""
    out = MlxBatch()
    for k, v in batch.__dict__.items():
        if k == "key":
            out["key"] = list(v)
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = _to_mx(v)
    return out


def mlx_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> Iterator[MlxBatch]:
    """包 torch DataLoader(仅数据搬运),yield MlxBatch。

    注意:num_workers>0 时 worker 进程需可导入 torch(数据加载进程)。
    """
    from torch.utils.data import DataLoader
    from ..dataset import svc_collate

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=svc_collate,
        persistent_workers=num_workers > 0,
    )
    for batch in loader:
        yield mlx_collate(batch)
