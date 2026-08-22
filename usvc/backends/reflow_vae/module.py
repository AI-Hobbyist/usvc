"""reflow_vae LightningModule (Reflow-VAE-SVC: Bi_RectifiedFlow + VAE).

backend_opts (defaults match configs/reflow-vae-lynxnet.yaml): back_bone
'lynxnet' (NaiveV2Diff) | 'wavenet'; n_layers 6, n_chans 512, n_hidden 256;
use_attention True, use_pitch_aug True, out_dims 128, n_units 768,
num_speaker 1.
"""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.module import BaseSVCModule
from .model import ReflowVAEModel

__all__ = ["ReflowVAEModule"]


class ReflowVAEModule(BaseSVCModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts
        self.model = ReflowVAEModel(
            n_units=opts.get("n_units", 768),
            n_spk=opts.get("num_speaker", 1),
            win_length=opts.get("win_length", 2048),
            sampling_rate=opts.get("sampling_rate", 44100),
            block_size=opts.get("block_size", 512),
            use_pitch_aug=opts.get("use_pitch_aug", True),
            out_dims=opts.get("out_dims", 128),
            n_layers=opts.get("n_layers", 6),
            n_chans=opts.get("n_chans", 512),
            n_hidden=opts.get("n_hidden", 256),
            back_bone=opts.get("back_bone", "lynxnet"),
            use_attention=opts.get("use_attention", True),
            t_start=opts.get("t_start", 0.0),
        )

    def forward_loss(self, batch: SVCBatch):
        units, f0, volume, spk_id, gt_spec, aug_shift = self._unpack(batch)
        loss, sub = self.model(
            units, f0, volume, spk_id, gt_spec, aug_shift=aug_shift,
            frame_len=batch.frame_len,
        )
        return loss, sub

    def sample(self, batch: SVCBatch, steps: int | None = None):
        units, f0, volume, spk_id, _, aug_shift = self._unpack(batch)
        with torch.no_grad():
            return self.model.sample(
                units, f0, volume, spk_id, aug_shift=aug_shift,
                frame_len=batch.frame_len,
                infer_step=steps if steps is not None else 10,
            )