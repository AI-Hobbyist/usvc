"""ddsp6 MLX 训练 module。build(cfg) -> (model, None, {"loss": loss_fn});loss_fn → ddsp+reflow 总 loss。"""
from __future__ import annotations
import mlx.core as mx
__all__ = ["build"]
from ....backends._defaults import (  # 与 torch 版共享同一份默认表/解析函数
    DDSP6_VERSION_DEFAULTS as _VERSION_DEFAULTS,
    resolve_ddsp6_opts as _resolve_opts,
)
def build(cfg):
    from ....nn.mlx.backends.ddsp6 import DDSP6Model
    opts = _resolve_opts(dict(cfg.backend_opts or {}))
    if opts.get("velocity_net") == "LYNXNet2_65":
        raise NotImplementedError("DDSP-SVC 6.5 currently requires the PyTorch backend; MLX supports 6.1-6.3")
    model = DDSP6Model(
        n_units=opts.get("n_units", 768), n_spk=opts.get("num_speaker", 1), t_start=opts.get("t_start", 0.0),
        use_norm=opts.get("use_norm", True), n_aux_layers=opts.get("n_aux_layers", 6), n_aux_chans=opts.get("n_aux_chans", 512),
        n_layers=opts.get("n_layers", 6), n_chans=opts.get("n_chans", 1024), out_dims=opts.get("out_dims", 128),
        velocity_net=opts.get("velocity_net", "LYNXNet2")
    )
    if getattr(cfg, "pretrained", None):
        from ..checkpoint import load_torch_weights
        load_torch_weights(model, cfg.pretrained)
        print(f"[mlx-ddsp6] loaded pretrained {cfg.pretrained}")
    def loss_fn(model, batch):
        gt_spec = batch.mel
        if gt_spec is None:
            raise RuntimeError("ddsp6 requires mel in batch")
        volume = batch.rms
        if volume is None:
            volume = (batch.mel.exp() - 1e-6).clip(min=1e-6).sqrt().mean(axis=-1)
        loss, _sub = model.forward(batch.units, batch.f0, volume, batch.spk_id,
                                   gt_spec, aug_shift=batch.aug_shift,
                                   frame_len=batch.frame_len)
        return loss
    return model, None, {"loss": loss_fn}
