"""diffusion LightningModule (DDPM epsilon-prediction)."""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.module import BaseSVCModule
from .model import DiffusionModel

__all__ = ["DiffusionModule"]


class DiffusionModule(BaseSVCModule):
    """LightningModule for diffusion backend (DDPM + ConvNext)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts

        self.model = DiffusionModel(
            n_units=opts.get("n_units", 768),
            n_spk=opts.get("num_speaker", 1),
            out_dims=opts.get("out_dims", 128),
            timesteps=opts.get("timesteps", 1000),
            n_hidden=opts.get("n_hidden", 256),
            use_pitch_aug=opts.get("use_pitch_aug", False),
            k_step=opts.get("k_step", 1000),
            denoiser=opts.get("denoiser", "convnext"),
            wn_layers=opts.get("wn_layers", 20),
            wn_chans=opts.get("wn_chans", 384),
            cn_layers=opts.get("cn_layers", 20),
            cn_chans=opts.get("cn_chans", 384),
            mlp_factor=opts.get("mlp_factor", 4),
            gradient_checkpointing=opts.get("gradient_checkpointing", False),
        )

    def forward_loss(self, batch: SVCBatch):
        units = batch.units
        f0 = batch.f0
        volume = batch.rms if batch.rms is not None else None
        spk_id = batch.spk_id
        gt_spec = batch.mel
        aug_shift = batch.aug_shift

        if volume is None:
            volume = (batch.mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)

        loss, sub = self.model(
            units, f0, volume, spk_id, gt_spec, aug_shift=aug_shift,
            frame_len=batch.frame_len,
        )
        return loss, sub

    def sample(self, batch: SVCBatch):
        units = batch.units
        f0 = batch.f0
        volume = batch.rms if batch.rms is not None else None
        spk_id = batch.spk_id

        if volume is None:
            volume = (batch.mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)

        with torch.no_grad():
            mel_gen = self.model.sample(
                units, f0, volume, spk_id,
                aug_shift=batch.aug_shift,
                frame_len=batch.frame_len,
            )
        return mel_gen
