"""ReflowShallowModel — reflow shallow-diffusion enhancer (mel-conditioned).

Condition = input mel + units/f0/vol embeddings; at training time the mel
condition is a *degraded* clean mel (see ``_degrade``), at inference the
sovits/rvc mel — so the velocity net performs a targeted "dirty -> clean"
correction, not blind regeneration. LYNXNet2 + RectifiedFlow are reused from
the DDSP6 backend. dim_cond = out_dims + n_hidden.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ddsp6.model import LYNXNet2, RectifiedFlow

__all__ = ["ReflowShallowModel"]


class ReflowShallowModel(nn.Module):
    def __init__(self, out_dims=128, n_layers=6, n_chans=768, n_units=768,
                 n_spk=1, n_hidden=256):
        super().__init__()
        self.out_dims = out_dims
        self.n_spk = n_spk
        self.unit_embed = nn.Linear(n_units, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.spk_embed = nn.Embedding(n_spk, n_hidden) if n_spk is not None and n_spk > 1 else None
        # dim_cond = mel (out_dims) + aux embeddings (n_hidden)
        self.reflow_model = RectifiedFlow(LYNXNet2(in_dims=out_dims, dim_cond=out_dims + n_hidden,
                                                   n_layers=n_layers, n_chans=n_chans),
                                          out_dims=out_dims, spec_min=-12, spec_max=2)
        # Degrade params (calibrated from 143 clean/rvc pairs, 8089 silent
        # frames): near-flat +0.76 dB bias, std 0.74->0.10 (low->high freq),
        # lag1 autocorrelation 0.67-0.79.  Sampled uniformly within ranges.
        self.degrade_cfg = {
            "bias": (0.4, 0.8),          # near-flat mel bias (dB)
            "noise_amp": (0.35, 0.75),   # target low-freq std after smoothing
            "smooth_win": 5,             # temporal window for lag1~0.7-0.8
        }

    def embed_condition(self, units, f0, volume, mel, spk_id=None):
        """Build [B, T, out_dims + n_hidden] condition: norm(mel) + embeddings."""
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        emb = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log())
        if volume is not None:
            if volume.dim() == 2:
                volume = volume.unsqueeze(-1)
            emb = emb + self.volume_embed(volume)
        if self.spk_embed is not None and spk_id is not None:
            spk = spk_id if spk_id.dim() > 1 else spk_id.unsqueeze(1)
            emb = emb + self.spk_embed(spk)
        return torch.cat([self.reflow_model.norm_spec(mel), emb], dim=-1)

    def _sample_degrade_params(self):
        out = {}
        for k, v in self.degrade_cfg.items():
            if isinstance(v, tuple):
                lo, hi = v
                out[k] = float(torch.empty(()).uniform_(lo, hi).item())
            else:
                out[k] = v
        return out

    @staticmethod
    def _degrade(mel, bias, noise_amp, smooth_win):
        """Degrade clean mel -> rvc-style artifacts (mel [B,T,M] dB domain).

        Two components (calibrated from silent-segment pure artifacts):
        1. near-flat positive bias (~+0.76 dB; extreme-low bin ~+0.3)
        2. temporally-smooth noise, std decreasing low->high freq via
           exp(-bin/62), lag1 autocorrelation ~0.7-0.8.
        """
        B, T, M = mel.shape
        # near-flat bias; extreme-low bin gets ~40% (observed +0.29 vs +0.76)
        bias_curve = mel.new_full((M,), bias)
        bias_curve[:16] = bias * 0.4
        idx = torch.arange(M, device=mel.device).float()
        taper = torch.exp(-idx / 62.0)
        noise_raw = torch.randn(B, M, T, device=mel.device) * taper[None, :, None]
        win = max(3, smooth_win)
        kernel = torch.ones(M, 1, win, device=mel.device) / win
        noise = F.conv1d(noise_raw, kernel, padding=win // 2, groups=M)
        noise = noise * (win ** 0.5) * noise_amp  # undo moving-average shrinkage
        return mel + noise.transpose(1, 2)[:, :T] + bias_curve[None, None, :]

    def forward_loss(self, units, f0, volume, real_mel, spk_id=None, t_start=0.0, frame_len=None):
        """Reflow loss: condition = degrade(clean_mel), target = clean mel."""
        dp = self._sample_degrade_params()
        with torch.no_grad():
            degraded_mel = self._degrade(real_mel, **dp)
        cond = self.embed_condition(units, f0, volume, degraded_mel, spk_id)
        from ...train.objective import lens_to_mask
        mask = lens_to_mask(frame_len, length=real_mel.shape[1]) if frame_len is not None else None
        if mask is not None:
            mask = mask.to(real_mel.device)
        loss = self.reflow_model(condition=cond, gt_spec=real_mel, infer=False,
                                 t_start=t_start, mask=mask)
        return loss, {"reflow_loss": float(loss.item())}

    def sample(self, units, f0, volume, sovits_mel, t_start=0.5, infer_step=10,
               method="euler", spk_id=None):
        """Shallow-diffuse sovits_mel toward a clean mel [B,T,M] dB domain."""
        cond = self.embed_condition(units, f0, volume, sovits_mel, spk_id)
        with torch.no_grad():
            mel = self.reflow_model(condition=cond, gt_spec=sovits_mel, infer=True,
                                    infer_step=infer_step, method=method,
                                    t_start=t_start, use_tqdm=False)
        return mel

    def load_ddsp6_reflow(self, ckpt_path):
        """Load DDSP6 reflow velocity_fn residual layers (projection excluded)."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        sd = {k: v for k, v in sd.items() if k.startswith("reflow_model.")}
        model_sd = self.state_dict()
        sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad_missing = [k for k in missing if k.startswith("reflow_model.velocity_fn.")
                       and "conditioner_projection" not in k]
        report = {"loaded_keys": len(sd), "missing": len(missing),
                  "unexpected": len(unexpected), "missing_velocity": len(bad_missing)}
        if bad_missing:
            report["missing_velocity_keys"] = bad_missing[:8]
        return report