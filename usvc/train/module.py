"""Base LightningModule for non-GAN SVC backends (rift_dit, ddsp6, diffusion).

Provides the training/validation loop skeleton. Subclass implements:
  - self.model (nn.Module)
  - forward_loss(batch) -> (total_loss, sub_losses_dict)
  - sample(batch) -> generated mel/spec (for validation)
"""
from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import LightningModule

from .config import TrainConfig
from .dataset import SVCBatch
from .metrics import compute_all_metrics

__all__ = ["BaseSVCModule"]


class BaseSVCModule(LightningModule):
    """Base for non-GAN backends.

    Subclass must implement:
      - forward_loss(batch: SVCBatch) -> (loss: Tensor, sub: dict[str, float])
      - sample(batch: SVCBatch) -> Tensor (generated mel for validation)
      - Optionally: configure_optimizers() if backend needs custom optimizer

    The base class handles:
      - training_step (calls forward_loss, logs)
      - validation_step (calls sample, computes metrics, logs media)
      - schedulefree optimizer train/eval mode switching
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = True
        self.vocoder: Any = None  # lazily loaded for validation
        self._val_metrics: list[dict] = []

    def training_step(self, batch: SVCBatch, batch_idx: int):
        loss, sub = self.forward_loss(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=self._bs(batch))
        for name, val in sub.items():
            self.log(f"train/{name}", val, batch_size=self._bs(batch))
        return loss

    def validation_step(self, batch: SVCBatch, batch_idx: int):
        if not self.trainer.is_global_zero:
            return

        with torch.no_grad():
            mel_gen = self.sample(batch)
        mel_gt = batch.mel
        if mel_gt is None or mel_gen is None:
            return

        mel_gen = mel_gen.float()
        mel_gt = mel_gt.float()
        frame_len = batch.frame_len

        for i in range(mel_gen.shape[0]):
            fl = frame_len[i].item() if frame_len is not None else mel_gen.shape[1]
            gen_i = mel_gen[i, :fl]
            gt_i = mel_gt[i, :fl]
            metrics = compute_all_metrics(gen_i, gt_i)
            self._val_metrics.append({k: v.item() for k, v in metrics.items()})

    def on_validation_epoch_end(self):
        if not self._val_metrics:
            return
        import numpy as np

        keys = self._val_metrics[0].keys()
        for k in keys:
            vals = [m[k] for m in self._val_metrics]
            self.log(f"val/{k}", float(np.mean(vals)), sync_dist=False)
        self._val_metrics.clear()

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if hasattr(opt, "train"):
            opt.train()

    def on_validation_start(self):
        opt = self.optimizers()
        if hasattr(opt, "eval"):
            opt.eval()

    def on_validation_end(self):
        opt = self.optimizers()
        if hasattr(opt, "train"):
            opt.train()

    def on_test_start(self):
        opt = self.optimizers()
        if hasattr(opt, "eval"):
            opt.eval()

    def on_test_end(self):
        opt = self.optimizers()
        if hasattr(opt, "train"):
            opt.train()

    def configure_optimizers(self):
        """Default: build from config. Override for custom param groups."""
        from .optimizer import build_optimizer

        result = build_optimizer(
            self.model.parameters(),
            self.cfg.optimizer,
            self.cfg.schedule,
            self.cfg.max_steps,
        )
        return result

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Gradient clipping with torch-native clip_grad_norm_.

        Lightning's automatic clipping is incompatible with AMP + fused
        AdamW (the precision plugin unscales gradients internally, which the
        fused kernel does itself). Use torch-native clipping instead.
        """
        if gradient_clip_val is None:
            return
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], gradient_clip_val)

    # ---- Subclass hooks ----

    def forward_loss(self, batch: SVCBatch) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss. Must implement in subclass."""
        raise NotImplementedError

    def sample(self, batch: SVCBatch) -> torch.Tensor | None:
        """Generate mel for validation. Must implement in subclass.

        Return None to skip validation metrics.
        """
        raise NotImplementedError

    # ---- Helpers ----

    @staticmethod
    def _unpack(batch: SVCBatch):
        """Unpack shared condition fields; volume falls back to mel energy."""
        volume = batch.rms if batch.rms is not None else None
        if volume is None:
            volume = (batch.mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)
        return batch.units, batch.f0, volume, batch.spk_id, batch.mel, batch.aug_shift

    @staticmethod
    def _bs(batch: SVCBatch) -> int:
        return batch.units.shape[0]
