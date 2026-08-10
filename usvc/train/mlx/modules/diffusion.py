"""diffusion MLX 训练 module。

build(cfg) -> (model, None, {"loss": loss_fn})。
loss_fn(model, batch) -> DDPM epsilon-prediction loss(标量 mlx array)。
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["build"]


def build(cfg):
    from ....nn.mlx.backends.diffusion import DiffusionModel

    opts = dict(cfg.backend_opts or {})
    k_step = int(opts.get("k_step", 1000))
    model = DiffusionModel(
        n_units=int(opts.get("n_units", 768)),
        n_spk=int(opts.get("num_speaker", 1)),
        out_dims=int(opts.get("out_dims", 128)),
        timesteps=int(opts.get("timesteps", 1000)),
        n_hidden=int(opts.get("n_hidden", 256)),
        use_pitch_aug=bool(opts.get("use_pitch_aug", False)),
        k_step=k_step,
        denoiser=opts.get("denoiser", "convnext"),
        wn_layers=int(opts.get("wn_layers", 20)),
        wn_chans=int(opts.get("wn_chans", 384)),
        cn_layers=int(opts.get("cn_layers", 20)),
        cn_chans=int(opts.get("cn_chans", 384)),
        mlp_factor=int(opts.get("mlp_factor", 4)),
        gradient_checkpointing=bool(opts.get("gradient_checkpointing", False)),
    )

    if getattr(cfg, "pretrained", None):
        from ..checkpoint import load_torch_weights
        load_torch_weights(model, cfg.pretrained)
        print(f"[mlx-diffusion] loaded pretrained {cfg.pretrained}")

    def loss_fn(model, batch):
        gt_spec = batch.mel
        if gt_spec is None:
            raise RuntimeError("diffusion requires mel in batch")
        units = batch.units
        f0 = batch.f0
        volume = batch.rms
        if volume is None:
            # 与 torch module.py 一致:从 mel 能量估计 volume
            volume = (mx.exp(gt_spec) - 1e-6).clip(min=1e-6).sqrt().mean(axis=-1)
        spk_id = batch.spk_id
        if spk_id is None:
            spk_id = mx.zeros((units.shape[0],), dtype=mx.int32)

        loss, _sub = model.forward(
            units, f0, volume, spk_id, gt_spec,
            aug_shift=batch.aug_shift,
            frame_len=batch.frame_len,
        )
        return loss

    return model, None, {"loss": loss_fn}
