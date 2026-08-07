"""rift_dit LightningModule.

Wraps RIFTDiTModel into BaseSVCModule. Uses FlowMatchingObjective for loss.
"""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.module import BaseSVCModule
from .model import RIFTDiTModel

__all__ = ["RIFTModule"]


class RIFTModule(BaseSVCModule):
    """LightningModule for rift_dit (DiT + Rectified Flow)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts

        self.model = RIFTDiTModel(
            dim=opts.get("dim", 512),
            depth=opts.get("depth", 8),
            head_dim=opts.get("head_dim", 64),
            dropout=opts.get("dropout", 0.0),
            ff_mult=opts.get("ff_mult", 4),
            n_mel_channels=opts.get("n_mel_channels", 128),
            num_speaker=opts.get("num_speaker", 1),
            cvec_dim=opts.get("cvec_dim", 768),       # RIFT-SVC default; 256 only for hubertsoft
            kernel_size=opts.get("kernel_size", 31),     # RIFT-SVC dit-base default (depthwise conv receptive field)
            zero_null_spk=opts.get("zero_null_spk", True),
            time_schedule=opts.get("time_schedule", "lognorm"),
            loss_type=opts.get("loss_type", "l2"),
        )
        self.drop_spk_prob = opts.get("drop_spk_prob", 0.0)
        self.eval_sample_steps = opts.get("eval_sample_steps", 50)

    def forward_loss(self, batch: SVCBatch):
        """Compute flow-matching loss. Delegates to RIFTDiTModel.forward."""
        mel = batch.mel
        spk_id = batch.spk_id
        f0 = batch.f0
        rms = batch.rms
        cvec = batch.units
        frame_len = batch.frame_len

        # Speaker dropout for CFG training
        drop_speaker = False
        if self.drop_spk_prob > 0 and self.training:
            B = spk_id.shape[0]
            num_drop = int(B * self.drop_spk_prob)
            drop_speaker = torch.zeros(B, dtype=torch.bool, device=spk_id.device)
            drop_speaker[:num_drop] = True
            drop_speaker = drop_speaker[torch.randperm(B)]

        loss, pred = self.model(
            mel, spk_id, f0, rms, cvec, frame_len, drop_speaker=drop_speaker
        )
        return loss, {"flow_loss": loss.item()}

    def sample(
        self,
        batch: SVCBatch,
        ds_cfg_strength: float = 0.0,
        spk_cfg_strength: float = 0.0,
        skip_cfg_strength: float = 0.0,
        cfg_rescale: float = 0.7,
        bad_cvec: torch.Tensor | None = None,
        steps: int | None = None,
    ):
        """Generate mel via ODE sampling.

        CFG (classifier-free guidance) parameters are passed through to
        RIFTDiTModel.sample; defaults keep validation CFG-free. bad_cvec is
        required when ds_cfg_strength > 0 (RIFT-SVC semantics).
        steps: ODE integration steps (default eval_sample_steps).
        """
        with torch.no_grad():
            result = self.model.sample(
                batch.mel,
                batch.spk_id,
                batch.f0,
                batch.rms,
                batch.units,
                batch.frame_len,
                steps=steps if steps is not None else self.eval_sample_steps,
                bad_cvec=bad_cvec,
                ds_cfg_strength=ds_cfg_strength,
                spk_cfg_strength=spk_cfg_strength,
                skip_cfg_strength=skip_cfg_strength,
                cfg_rescale=cfg_rescale,
            )
        # sample returns (mel_gen, trajectory)
        if isinstance(result, tuple):
            return result[0]
        return result
