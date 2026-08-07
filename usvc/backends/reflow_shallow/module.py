"""ReflowShallowModule — LightningModule for the reflow_shallow backend (v3).

Mel-conditioned reflow: training condition = degrade(clean_mel) +
units/f0/vol embeddings; inference condition = sovits/rvc mel.
"""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.module import BaseSVCModule
from .model import ReflowShallowModel

__all__ = ["ReflowShallowModule"]


class ReflowShallowModule(BaseSVCModule):
    """LightningModule wrapping ReflowShallowModel (v2)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts
        self.model = ReflowShallowModel(
            out_dims=int(opts.get("out_dims", opts.get("n_mels", 128))),
            n_layers=int(opts.get("n_layers", 6)),
            n_chans=int(opts.get("n_chans", 768)),
            n_units=int(opts.get("n_units", opts.get("n_unit", 768))),
            n_spk=int(opts.get("num_speaker", opts.get("n_spk", 1))),
            n_hidden=int(opts.get("n_hidden", 256)),
        )
        # t_start lower bound for training-time sampling.  0.0 = uniform.
        self.t_start = float(opts.get("t_start", 0.0))
        # inference t_start for validation sample
        self.infer_t_start = float(opts.get("infer_t_start", 0.5))

    def forward_loss(self, batch):
        mel = batch.mel  # [B, T, M] clean target
        if mel is None:
            raise RuntimeError("reflow_shallow requires mel in the batch")
        units = batch.units  # [B, T, D]
        f0 = batch.f0  # [B, T]
        rms = batch.rms  # [B, T] or None
        spk = batch.spk_id  # [B] or None
        loss, sub = self.model.forward_loss(
            units, f0, rms, mel, spk_id=spk, t_start=self.t_start,
            frame_len=batch.frame_len,
        )
        return loss, sub

    def sample(self, batch):
        """Shallow-diffuse the batch mel (validation)."""
        mel = batch.mel
        if mel is None:
            return None
        return self.model.sample(
            batch.units, batch.f0, batch.rms, mel,
            t_start=self.infer_t_start, infer_step=10,
            spk_id=batch.spk_id,
        )
