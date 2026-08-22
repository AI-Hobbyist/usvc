"""ReflowShallowModule — LightningModule for the reflow_shallow backend (v3).

Training condition = degrade(clean_mel) + units/f0/vol embeddings; inference
condition = sovits/rvc mel.
"""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.module import BaseSVCModule
from .model import ReflowShallowModel

__all__ = ["ReflowShallowModule"]


class ReflowShallowModule(BaseSVCModule):
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
        self.t_start = float(opts.get("t_start", 0.0))  # training t_start lower bound (0.0 = uniform)
        self.infer_t_start = float(opts.get("infer_t_start", 0.5))  # validation ODE start
        # torch.compile velocity_fn（LYNXNet2 纯 conv 可编译）。实测 32→25ms/step (1.27×)；
        # 模型小、SM 仅 58% launch-bound，compile 减少发射开销收益明显。
        if opts.get("compile_velocity", False):
            try:
                self.model.reflow_model.velocity_fn = torch.compile(
                    self.model.reflow_model.velocity_fn,
                    mode=opts.get("compile_mode", "max-autotune"),
                    dynamic=False,
                )
            except Exception:
                pass

    def forward_loss(self, batch):
        mel = batch.mel  # [B, T, M] clean target
        if mel is None:
            raise RuntimeError("reflow_shallow requires mel in the batch")
        loss, sub = self.model.forward_loss(batch.units, batch.f0, batch.rms, mel,
                                            spk_id=batch.spk_id, t_start=self.t_start,
                                            frame_len=batch.frame_len)
        return loss, sub

    def sample(self, batch):
        """Shallow-diffuse the batch mel (validation)."""
        if batch.mel is None:
            return None
        return self.model.sample(batch.units, batch.f0, batch.rms, batch.mel,
                                 t_start=self.infer_t_start, infer_step=10,
                                 spk_id=batch.spk_id)