"""ddsp6 MLX 训练 module。

build(cfg) -> (model, None, {"loss": loss_fn})。
loss_fn(model, batch) -> ddsp + reflow 总 loss(标量 mlx array)。
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["build"]

# 与 torch usvc/backends/ddsp6/module.py 一致的版本默认值
# 与 torch usvc/backends/ddsp6/module.py 共享同一份默认表/解析函数
from ....backends._defaults import (
    DDSP6_VERSION_DEFAULTS as _VERSION_DEFAULTS,
    resolve_ddsp6_opts as _resolve_opts,
)

def build(cfg):
    from ....nn.mlx.backends.ddsp6 import DDSP6Model

    opts = _resolve_opts(dict(cfg.backend_opts or {}))
    model = DDSP6Model(
        n_units=opts.get("n_units", 768),
        n_spk=opts.get("num_speaker", 1),
        t_start=opts.get("t_start", 0.0),
        use_norm=opts.get("use_norm", True),
        n_aux_layers=opts.get("n_aux_layers", 6),
        n_aux_chans=opts.get("n_aux_chans", 512),
        n_layers=opts.get("n_layers", 6),
        n_chans=opts.get("n_chans", 1024),
        out_dims=opts.get("out_dims", 128),
        velocity_net=opts.get("velocity_net", "LYNXNet2"),
    )

    if getattr(cfg, "pretrained", None):
        from ..checkpoint import load_torch_weights
        load_torch_weights(model, cfg.pretrained)
        print(f"[mlx-ddsp6] loaded pretrained {cfg.pretrained}")

    def loss_fn(model, batch):
        units = batch.units
        f0 = batch.f0
        volume = batch.rms if batch.rms is not None else None
        spk_id = batch.spk_id
        gt_spec = batch.mel
        if gt_spec is None:
            raise RuntimeError("ddsp6 requires mel in batch")
        if volume is None:
            volume = (
                (batch.mel.exp() - 1e-6).clip(min=1e-6).sqrt().mean(axis=-1)
            )
        loss, _sub = model.forward(
            units, f0, volume, spk_id, gt_spec,
            aug_shift=batch.aug_shift, frame_len=batch.frame_len,
        )
        return loss

    return model, None, {"loss": loss_fn}
