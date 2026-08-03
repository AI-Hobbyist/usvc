"""Base LightningModule for GAN-based SVC backends (sovits, rvc).

Uses manual_optimization for G/D alternating training.
"""
from __future__ import annotations

import torch
from lightning.pytorch import LightningModule

from .config import TrainConfig
from .dataset import SVCBatch

__all__ = ["BaseGANModule"]


class BaseGANModule(LightningModule):
    """Base for GAN backends (so-vits, rvc) with G/D alternating optimization.

    Subclass must implement:
      - self.generator (nn.Module)
      - self.discriminator (nn.Module)
      - generator_loss(batch) -> (loss_g, sub_dict)
      - discriminator_loss(batch) -> (loss_d, sub_dict)
      - sample(batch) -> generated wav/mel for validation
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False  # manual G/D

    def training_step(self, batch: SVCBatch, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        # --- Discriminator step ---
        loss_d, sub_d = self.discriminator_loss(batch)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        if self.cfg.grad_clip is not None:
            self.clip_gradients(opt_d, gradient_clip_val=self.cfg.grad_clip)
        opt_d.step()

        # --- Generator step ---
        loss_g, sub_g = self.generator_loss(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        if self.cfg.grad_clip is not None:
            self.clip_gradients(opt_g, gradient_clip_val=self.cfg.grad_clip)
        opt_g.step()

        bs = self._bs(batch)
        self.log("train/loss_g", loss_g, prog_bar=True, batch_size=bs)
        self.log("train/loss_d", loss_d, prog_bar=True, batch_size=bs)
        for name, val in {**sub_g, **sub_d}.items():
            self.log(f"train/{name}", val, batch_size=bs)

    def validation_step(self, batch: SVCBatch, batch_idx: int):
        if not self.trainer.is_global_zero:
            return
        # GAN backends override this for wav-based validation
        pass

    def configure_optimizers(self):
        from .optimizer import build_optimizer_gan

        (opt_g, opt_d), scheds = build_optimizer_gan(
            self.generator.parameters(),
            self.discriminator.parameters(),
            self.cfg.optimizer,
            self.cfg.schedule,
            self.cfg.max_steps,
        )
        if scheds:
            return [opt_g, opt_d], scheds
        return [opt_g, opt_d]

    # ---- Subclass hooks ----

    def generator_loss(self, batch: SVCBatch) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def discriminator_loss(self, batch: SVCBatch) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def sample(self, batch: SVCBatch):
        raise NotImplementedError

    @staticmethod
    def _bs(batch: SVCBatch) -> int:
        return batch.units.shape[0]
