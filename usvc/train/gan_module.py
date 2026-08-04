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
      - forward_generator(batch) -> out dict (single generator forward per step)
      - generator_loss(batch, out) -> (loss_g, sub_dict)
      - discriminator_loss(batch, out) -> (loss_d, sub_dict)
      - sample(batch) -> generated wav/mel for validation

    IMPORTANT (autograd version-check constraint):
    `training_step` runs the order used by so-vits-svc/RVC original train.py:
        G fwd once -> D fwd#1 (on wave_gen.detach()) -> D bwd -> D opt.step()
        -> D fwd#2 (on wave_gen with grad) -> G bwd -> G opt.step()
    The generator is forwarded exactly once per step; its outputs are shared
    by both the D loss and the G loss. Do NOT restructure it into "forward
    both losses first, then backward both": AdamW's step() mutates shared
    weight_norm parameters in-place (twice per param with default
    weight_decay), and a second graph built before that step fails its
    autograd version check at backward with "modified by an inplace
    operation" RuntimeError. This is a PyTorch design constraint, not a
    weight_norm API bug (parametrizations.weight_norm fails identically).
    Keep every optimizer step between its loss's forward and the next loss's
    forward; the D step must come before the G loss's D fwd#2.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False  # manual G/D

    def training_step(self, batch: SVCBatch, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        # Single generator forward per step, shared by D and G losses
        # (matches so-vits-svc/RVC original loops: net_g once, net_d twice).
        out = self.forward_generator(batch)

        # --- Discriminator step (D fwd#1 on detached wave_gen) ---
        loss_d, sub_d = self.discriminator_loss(batch, out)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        if self.cfg.grad_clip is not None:
            # Lightning's clip_gradients is incompatible with AMP + fused
            # optimizers (plugin unscales internally); use torch-native clip.
            for group in opt_d.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], self.cfg.grad_clip)
        opt_d.step()

        # --- Generator step (D fwd#2 on wave_gen with grad, reuses `out`) ---
        # NOTE: must run AFTER opt_d.step() (autograd version check).
        loss_g, sub_g = self.generator_loss(batch, out)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        if self.cfg.grad_clip is not None:
            for group in opt_g.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], self.cfg.grad_clip)
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

    def forward_generator(self, batch: SVCBatch):
        raise NotImplementedError

    def generator_loss(self, batch: SVCBatch, out=None) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def discriminator_loss(self, batch: SVCBatch, out=None) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def sample(self, batch: SVCBatch):
        raise NotImplementedError

    @staticmethod
    def _bs(batch: SVCBatch) -> int:
        return batch.units.shape[0]
