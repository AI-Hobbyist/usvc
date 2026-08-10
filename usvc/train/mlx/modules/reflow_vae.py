"""reflow_vae MLX 训练 module。

build(cfg) -> (model, None, {"loss": loss_fn})。
loss_fn(model, batch) -> Bi_RectifiedFlow reflow loss(标量 mlx array)。
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["build"]


def build(cfg):
    from ....nn.mlx.backends.reflow_vae import ReflowVAEModel

    opts = dict(cfg.backend_opts or {})
    model = ReflowVAEModel(
        n_units=opts.get("n_units", 768),
        n_spk=opts.get("num_speaker", 1),
        win_length=opts.get("win_length", 2048),
        sampling_rate=opts.get("sampling_rate", 44100),
        block_size=opts.get("block_size", 512),
        use_pitch_aug=opts.get("use_pitch_aug", True),
        out_dims=opts.get("out_dims", 128),
        n_layers=opts.get("n_layers", 6),
        n_chans=opts.get("n_chans", 512),
        n_hidden=opts.get("n_hidden", 256),
        back_bone=opts.get("back_bone", "lynxnet"),
        use_attention=opts.get("use_attention", True),
        t_start=opts.get("t_start", 0.0),
    )

    if getattr(cfg, "pretrained", None):
        from ..checkpoint import load_torch_weights
        load_torch_weights(model, cfg.pretrained)
        print(f"[mlx-reflow_vae] loaded pretrained {cfg.pretrained}")

    def loss_fn(model, batch):
        units = batch.units
        f0 = batch.f0
        volume = batch.rms if batch.rms is not None else None
        spk_id = batch.spk_id
        gt_spec = batch.mel
        if gt_spec is None:
            raise RuntimeError("reflow_vae requires mel in batch")
        if volume is None:
            volume = (
                (batch.mel.exp() - 1e-6).clip(min=1e-6).sqrt().mean(axis=-1)
            )
        loss, _sub = model.forward_loss(
            units, f0, volume, spk_id, gt_spec,
            aug_shift=batch.aug_shift, frame_len=batch.frame_len,
        )
        return loss

    return model, None, {"loss": loss_fn}
