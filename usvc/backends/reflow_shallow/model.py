"""ReflowShallowModel — reflow shallow-diffusion enhancer (v2, DDPM-style).

This version fixes the fundamental flaw of v1: the condition is now
**mel-independent** (units/f0/volume/spk embeddings), exactly like the
Diffusion-SVC NaiveDiff shallow-diffusion model.

v1 problem: condition = corrupt(clean_mel), and at inference condition =
sovits_mel = gt_spec (same source).  The reflow velocity learned to head
*toward* the condition, so output ≈ sovits_mel (no refinement).

v2 fix (DDPM-equivalent shallow diffusion):
  - Training: x_t = (1-t)*noise + t*norm(clean_mel)
              condition = units/f0/vol/spk embeddings (mel-independent)
              velocity target = clean_mel - noise
  - Inference: gt_spec = sovits_mel (the thing to refine)
               condition = units/f0/vol/spk embeddings
               x_start = t_start*norm(sovits_mel) + (1-t_start)*noise
               ODE integrate t_start → 1.0
               → reflow refines sovits_mel toward the clean mel manifold

The velocity network (LYNXNet2) and RectifiedFlow solver are reused from
the DDSP6 backend.  A new ``conditioner`` module projects the units/f0/vol
embedings into the LYNXNet2 ``dim_cond`` channel space.

NOTE: because dim_cond is now the embedding width (not mel width), the
DDSP6 reflow_model weights do **not** transfer the conditioner_projection.
The velocity_fn residual layers still transfer, giving a warm start.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..ddsp6.model import LYNXNet2, RectifiedFlow

__all__ = ["ReflowShallowModel"]


class ReflowShallowModel(nn.Module):
    """Standalone reflow shallow-diffusion model (mel-independent condition).

    Args:
        out_dims: mel dimension (128).
        n_layers / n_chans: LYNXNet2 velocity network depth/width.
        n_units: content-feature dimension (768 for contentvec).
        n_spk: number of speakers (1 = single-speaker, no embedding).
        n_hidden: condition embedding width (also LYNXNet2 dim_cond).
    """

    def __init__(
        self,
        out_dims: int = 128,
        n_layers: int = 6,
        n_chans: int = 768,
        n_units: int = 768,
        n_spk: int = 1,
        n_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.out_dims = out_dims
        self.n_spk = n_spk

        # ---- condition embedr (mel-independent, Diffusion-SVC style) ----
        self.unit_embed = nn.Linear(n_units, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        else:
            self.spk_embed = None

        # ---- reflow (LYNXNet2 velocity net + RectifiedFlow solver) ----
        velocity_fn = LYNXNet2(
            in_dims=out_dims,
            dim_cond=n_hidden,  # condition is the embedding, NOT mel
            n_layers=n_layers,
            n_chans=n_chans,
        )
        self.reflow_model = RectifiedFlow(
            velocity_fn, out_dims=out_dims, spec_min=-12, spec_max=2,
        )

    # -- condition embedding -----------------------------------------------

    def embed_condition(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        volume: torch.Tensor | None,
        spk_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build [B, T, n_hidden] condition embeddings (mel-independent).

        Args:
            units: [B, T, n_units]
            f0: [B, T] or [B, T, 1]
            volume: [B, T] or [B, T, 1] or None (estimated from mel if None)
            spk_id: [B] or None
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log())
        if volume is not None:
            if volume.dim() == 2:
                volume = volume.unsqueeze(-1)
            x = x + self.volume_embed(volume)
        if self.spk_embed is not None and spk_id is not None:
            spk = spk_id if spk_id.dim() > 1 else spk_id.unsqueeze(1)
            x = x + self.spk_embed(spk)
        return x  # [B, T, n_hidden]

    # -- training -----------------------------------------------------------

    def forward_loss(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        volume: torch.Tensor | None,
        real_mel: torch.Tensor,
        spk_id: torch.Tensor | None = None,
        t_start: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the reflow loss with mel-independent conditioning.

        Args:
            units: [B, T, n_units]
            f0: [B, T] or [B, T, 1]
            volume: [B, T] or [B, T, 1] or None
            real_mel: [B, T, M] clean target mel (dB domain).
            spk_id: [B] or None
            t_start: lower bound for the time sample (t ∈ [t_start, 1)).

        Returns:
            (loss, {"reflow_loss": float}).
        """
        cond = self.embed_condition(units, f0, volume, spk_id)  # [B, T, H]
        loss = self.reflow_model(
            condition=cond,
            gt_spec=real_mel,
            infer=False,
            t_start=t_start,
        )
        return loss, {"reflow_loss": float(loss.item())}

    # -- inference ----------------------------------------------------------

    def sample(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        volume: torch.Tensor | None,
        sovits_mel: torch.Tensor,
        t_start: float = 0.5,
        infer_step: int = 10,
        method: str = "euler",
        spk_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shallow-diffuse ``sovits_mel`` toward a clean mel.

        Args:
            units: [B, T, n_units]
            f0: [B, T] or [B, T, 1]
            volume: [B, T] or [B, T, 1] or None
            sovits_mel: [B, T, M] sovits/rvc output mel (dB domain) — the
                thing to refine.  Used as the ODE start point (mixed with
                noise according to t_start).
            t_start: ODE start time in [0, 1).  0 = full generation from
                noise (ignores sovits_mel), →1 = keep sovits_mel unchanged.
                Shallow diffusion: 0.3–0.7.
            infer_step: ODE steps.
            method: 'euler' or 'rk4'.

        Returns:
            Refined mel [B, T, M] (dB domain).
        """
        cond = self.embed_condition(units, f0, volume, spk_id)  # [B, T, H]
        with torch.no_grad():
            mel = self.reflow_model(
                condition=cond,
                gt_spec=sovits_mel,  # ODE start point (mixed with noise)
                infer=True,
                infer_step=infer_step,
                method=method,
                t_start=t_start,
                use_tqdm=False,
            )
        return mel  # [B, T, M]

    # -- pretrained loading -------------------------------------------------

    def load_ddsp6_reflow(self, ckpt_path: str) -> dict:
        """Load velocity_fn residual layers from a DDSP6 reflow checkpoint.

        The LYNXNet2 residual layers (input_projection, residual_blocks,
        norm, output_projection) transfer directly.  The
        ``conditioner_projection`` does NOT transfer (dim_cond changed from
        128→n_hidden), so it is randomly initialised.

        Returns the load report.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # strip the lightning "model." prefix if present
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        # only keep reflow_model.* keys
        sd = {k: v for k, v in sd.items() if k.startswith("reflow_model.")}
        # drop shape-mismatched keys (e.g. conditioner_projection 128→256)
        model_sd = self.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad_missing = [k for k in missing if k.startswith("reflow_model.velocity_fn.")
                       and "conditioner_projection" not in k]
        report = {
            "loaded_keys": len(sd),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "missing_velocity": len(bad_missing),
        }
        if bad_missing:
            report["missing_velocity_keys"] = bad_missing[:8]
        return report
