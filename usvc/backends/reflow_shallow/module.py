"""ReflowShallowModule — LightningModule for the reflow_shallow backend.

Training: corrupt(real_mel) → condition, real_mel → target, reflow_loss.
Inference (sample): shallow-diffuse the (corrupted) mel via t_start ODE.
"""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.module import BaseSVCModule
from .corrupt import corrupt_mel
from .model import ReflowShallowModel

__all__ = ["ReflowShallowModule"]


class ReflowShallowModule(BaseSVCModule):
    """LightningModule wrapping ReflowShallowModel."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts
        self.model = ReflowShallowModel(
            out_dims=int(opts.get("out_dims", opts.get("n_mels", 128))),
            n_layers=int(opts.get("n_layers", 6)),
            n_chans=int(opts.get("n_chans", 768)),
        )
        # t_start lower bound for training-time sampling (bias toward the
        # shallow regime the model operates in at inference).  0.0 = uniform.
        self.t_start = float(opts.get("t_start", 0.0))
        # corruption strength range (random per-batch)
        self.corrupt_strength = opts.get(
            "corrupt_strength", (0.3, 1.0)
        )
        if isinstance(self.corrupt_strength, (int, float)):
            self.corrupt_strength = (0.0, float(self.corrupt_strength))

    def forward_loss(self, batch):
        mel = batch.mel  # [B, T, M] clean target
        if mel is None:
            raise RuntimeError("reflow_shallow requires mel in the batch")
        # build the corrupted condition online
        cond = corrupt_mel(mel, strength_range=self.corrupt_strength)
        loss, sub = self.model.forward_loss(cond, mel, t_start=self.t_start)
        return loss, sub

    def sample(self, batch):
        """Shallow-diffuse the batch mel (validation; uses t_start=0.8)."""
        mel = batch.mel
        if mel is None:
            return None
        return self.model.sample(mel, t_start=0.8, infer_step=10)
