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
# 版本默认值表与解析函数共享于 backends._defaults(was 本文件内嵌副本)
from .._defaults import (  # noqa: E402  (tests 仍可从本模块导入 _resolve_opts)
    DDSP6_VERSION_DEFAULTS as _VERSION_DEFAULTS,
    resolve_ddsp6_opts as _resolve_opts,
)

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
        # torch.compile velocity_fn（LYNXNet2 纯 conv，可安全编译；DDSP 谱路径含 complex STFT 不能编译）。
        # 实测 83→75ms/step (1.1×)，胶水 41%%→28%%。由 backend_opts.compile_velocity 开关，默认关（保持可重现 + 避免 compile 副作用）。
        if opts.get("compile_velocity", False):
            try:
                self.model.reflow_model.velocity_fn = torch.compile(
                    self.model.reflow_model.velocity_fn,
                    mode=opts.get("compile_mode", "max-autotune"),
                    dynamic=False,
                )
            except Exception:
                pass  # compile 失败回退 eager，不影响正确性

    def forward_loss(self, batch: SVCBatch):
        """DDSP + reflow loss."""
        units, f0, volume, spk_id, gt_spec, aug_shift = self._unpack(batch)
        loss, sub = self.model(
            units, f0, volume, spk_id, gt_spec, aug_shift=aug_shift,
            frame_len=batch.frame_len,
        )
        return loss, sub

    def sample(self, batch: SVCBatch, steps: int | None = None):
        """Generate mel for validation via DDSP + reflow sampling.

        steps: ODE integration steps (default 10, matches DDSP-SVC infer).
        """
        units, f0, volume, spk_id, _, aug_shift = self._unpack(batch)
        with torch.no_grad():
            mel_gen = self.model.sample(
                units, f0, volume, spk_id, aug_shift=aug_shift,
                infer_step=steps if steps is not None else 10,
            )
        return mel_gen
