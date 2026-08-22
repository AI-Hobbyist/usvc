"""rift_dit MLX 训练 module。build(cfg) -> (model, None, {"loss": loss_fn});loss_fn → flow-matching loss。"""
from __future__ import annotations
import mlx.core as mx
__all__ = ["build"]
def build(cfg):
    from ....nn.mlx.backends.rift_dit import RIFTDiTModel
    opts = dict(cfg.backend_opts or {})
    model = RIFTDiTModel(
        dim=opts.get("dim", 512), depth=opts.get("depth", 8), head_dim=opts.get("head_dim", 64),
        dropout=opts.get("dropout", 0.0), ff_mult=opts.get("ff_mult", 4), n_mel_channels=opts.get("n_mel_channels", 128),
        num_speaker=opts.get("num_speaker", 1), cvec_dim=opts.get("cvec_dim", 768), kernel_size=opts.get("kernel_size", 31),
        zero_null_spk=opts.get("zero_null_spk", True)
    )
    # 预训练权重(torch ckpt / mlx npz 均可,经 from_ckpt)
    if getattr(cfg, "pretrained", None):
        from ..checkpoint import load_torch_weights
        load_torch_weights(model, cfg.pretrained)
        print(f"[mlx-rift_dit] loaded pretrained {cfg.pretrained}")
    drop_spk_prob = float(opts.get("drop_spk_prob", 0.0))
    def loss_fn(model, batch):
        mel = batch.mel
        if mel is None:
            raise RuntimeError("rift_dit requires mel in batch")
        rms = batch.rms
        if rms is None:
            rms = mx.ones_like(batch.f0) * 0.5
        spk_id = batch.spk_id
        if spk_id is None:
            spk_id = mx.zeros((mel.shape[0],), dtype=mx.int32)
        # speaker dropout(CFG 训练)
        drop_speaker = False
        if drop_spk_prob > 0 and model.training:
            B = mel.shape[0]
            num_drop = int(B * drop_spk_prob)
            drop_speaker = mx.zeros(B, dtype=mx.bool_)
            if num_drop > 0:
                idx = mx.random.permutation(B)[:num_drop]
                drop_speaker = mx.scatter(drop_speaker, idx,
                                          mx.ones(num_drop, dtype=mx.bool_))
        loss, pred = model.forward(mel, spk_id, batch.f0, rms, batch.units,
                                   frame_len=batch.frame_len,
                                   drop_speaker=drop_speaker)
        return loss
    return model, None, {"loss": loss_fn}
