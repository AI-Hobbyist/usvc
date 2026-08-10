"""reflow_shallow MLX 训练 module。

build(cfg) -> (model, None, {"loss": loss_fn})。
loss_fn(model, batch) -> reflow loss(degrade(clean_mel) 条件,clean mel 目标)。
"""
from __future__ import annotations

__all__ = ["build"]


def build(cfg):
    from ....nn.mlx.backends.reflow_shallow import ReflowShallowModel

    opts = dict(cfg.backend_opts or {})
    model = ReflowShallowModel(
        out_dims=int(opts.get("out_dims", opts.get("n_mels", 128))),
        n_layers=int(opts.get("n_layers", 6)),
        n_chans=int(opts.get("n_chans", 768)),
        n_units=int(opts.get("n_units", opts.get("n_unit", 768))),
        n_spk=int(opts.get("num_speaker", opts.get("n_spk", 1))),
        n_hidden=int(opts.get("n_hidden", 256)),
    )
    # 训练时的 t_start 下界(0.0 = uniform)
    t_start = float(opts.get("t_start", 0.0))

    if getattr(cfg, "pretrained", None):
        from ..checkpoint import load_torch_weights
        load_torch_weights(model, cfg.pretrained)
        print(f"[mlx-reflow_shallow] loaded pretrained {cfg.pretrained}")

    def loss_fn(model, batch):
        mel = batch.mel
        if mel is None:
            raise RuntimeError("reflow_shallow requires mel in batch")
        units = batch.units
        f0 = batch.f0
        rms = batch.rms
        spk = batch.spk_id
        loss = model.forward_loss(
            units, f0, rms, mel, spk_id=spk,
            t_start=t_start, frame_len=batch.frame_len,
        )
        return loss

    return model, None, {"loss": loss_fn}
