"""ddsp6 LightningModule.

Wraps DDSP6Model into BaseSVCModule. DDSP loss + reflow loss.

backend_opts version selects the DDSP-SVC release defaults:
  6.1 -> velocity_net LYNXNet, n_aux_layers=3, n_aux_chans=256, n_chans=512,
         t_start=0.7, use_norm=False, units contentvec (hop 320)
  6.2 -> velocity_net LYNXNet, n_aux_layers=6, n_aux_chans=512, n_chans=1024,
         t_start=0.0, use_norm=True, units contentvec_tta2x
  6.3 -> velocity_net LYNXNet2 (default), same dims as 6.2
Explicit opts (n_aux_layers, n_chans, t_start, ...) always override version
defaults.
"""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.module import BaseSVCModule
from .model import DDSP6Model

__all__ = ["DDSPModule"]

# Per-release default model dims (configs/reflow.yaml model section).
_VERSION_DEFAULTS = {
    "6.1": dict(velocity_net="LYNXNet", n_aux_layers=3, n_aux_chans=256,
                 n_layers=6, n_chans=512, t_start=0.7, use_norm=False),
    "6.2": dict(velocity_net="LYNXNet", n_aux_layers=6, n_aux_chans=512,
                 n_layers=6, n_chans=1024, t_start=0.0, use_norm=True),
    "6.3": dict(velocity_net="LYNXNet2", n_aux_layers=6, n_aux_chans=512,
                 n_layers=6, n_chans=1024, t_start=0.0, use_norm=True),
}


def _resolve_opts(opts: dict) -> dict:
    """Version defaults overlaid by explicit backend_opts."""
    version = str(opts.get("version", "6.3"))
    if version not in _VERSION_DEFAULTS:
        raise ValueError(f"ddsp6 version must be one of {list(_VERSION_DEFAULTS)}, got {version!r}")
    resolved = dict(_VERSION_DEFAULTS[version])
    for k, v in opts.items():
        if k != "version" and k in resolved:
            resolved[k] = v
    return resolved


class DDSPModule(BaseSVCModule):
    """LightningModule for ddsp6 (DDSP + Rectified Flow)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = _resolve_opts(cfg.backend_opts)

        self.model = DDSP6Model(
            n_units=opts.get("n_units", 768),
            n_spk=opts.get("num_speaker", 1),
            t_start=opts.get("t_start", 0.0),
            use_norm=opts.get("use_norm", True),
            n_aux_layers=opts.get("n_aux_layers", 6),
            n_aux_chans=opts.get("n_aux_chans", 512),
            n_layers=opts.get("n_layers", 6),
            n_chans=opts.get("n_chans", 1024),
            out_dims=opts.get("out_dims", 128),
            velocity_net=opts.get("velocity_net", "LYNXNet2"),
        )

    def forward_loss(self, batch: SVCBatch):
        """DDSP + reflow loss."""
        units = batch.units
        f0 = batch.f0
        volume = batch.rms if batch.rms is not None else None
        spk_id = batch.spk_id
        gt_spec = batch.mel
        aug_shift = batch.aug_shift

        if volume is None:
            # Compute volume from mel if not available
            volume = (batch.mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)

        loss, sub = self.model(
            units, f0, volume, spk_id, gt_spec, aug_shift=aug_shift,
            frame_len=batch.frame_len,
        )
        return loss, sub

    def sample(self, batch: SVCBatch):
        """Generate mel for validation via DDSP + reflow sampling."""
        units = batch.units
        f0 = batch.f0
        volume = batch.rms if batch.rms is not None else None
        spk_id = batch.spk_id
        aug_shift = batch.aug_shift

        if volume is None:
            volume = (batch.mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)

        with torch.no_grad():
            mel_gen = self.model.sample(
                units, f0, volume, spk_id, aug_shift=aug_shift,
                frame_len=batch.frame_len,
            )
        return mel_gen
