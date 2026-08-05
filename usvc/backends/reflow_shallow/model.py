"""ReflowShallowModel — a standalone reflow shallow-diffusion enhancer.

Unlike the DDSP6 reflow (which uses DDSP-synthesised mel as the condition),
this model uses the sovits/rvc **output** mel directly as the condition.  At
training time the condition is ``corrupt(real_mel)`` (simulating sovits
artefacts) and the target is ``real_mel`` — so the model learns a
(artefacted_mel → clean_mel) refinement that is in-distribution at inference
time (no DDSP timbre leakage).

The velocity network (LYNXNet2) and RectifiedFlow solver are reused verbatim
from the DDSP6 backend (``usvc.backends.ddsp6.model``).  Only the condition
source changes, so a DDSP6 ``reflow_model`` checkpoint loads directly as
initialisation.

Mathematics (identical to DDSP6 RectifiedFlow):
  - Training:  v_target = x_1 - x_0  (x_1 = norm(real_mel), x_0 = randn)
               loss = reflow_loss(x_1, x_0, t, cond=corrupt_mel)   [l2_lognorm]
  - Inference (shallow diffusion):
               x_start = t_start * norm(sovits_mel) + (1-t_start) * randn
               ODE integrate from t_start → 1.0 with cond = sovits_mel
               output = denorm(x)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..ddsp6.model import LYNXNet2, RectifiedFlow

__all__ = ["ReflowShallowModel"]


class ReflowShallowModel(nn.Module):
    """Standalone reflow shallow-diffusion model (no DDSP).

    Args:
        out_dims: mel dimension (128).
        n_layers / n_chans: LYNXNet2 velocity network depth/width.
    """

    def __init__(
        self,
        out_dims: int = 128,
        n_layers: int = 6,
        n_chans: int = 768,
    ) -> None:
        super().__init__()
        self.out_dims = out_dims
        velocity_fn = LYNXNet2(
            in_dims=out_dims,
            dim_cond=out_dims,  # condition is a mel (same dim as the target)
            n_layers=n_layers,
            n_chans=n_chans,
        )
        self.reflow_model = RectifiedFlow(
            velocity_fn, out_dims=out_dims, spec_min=-12, spec_max=2,
        )

    # -- training -----------------------------------------------------------

    def forward_loss(
        self,
        cond_mel: torch.Tensor,
        real_mel: torch.Tensor,
        t_start: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the reflow loss.

        Args:
            cond_mel: [B, T, M] corrupted mel (simulating sovits/rvc output).
            real_mel: [B, T, M] clean target mel.
            t_start: lower bound for the time sample (t ∈ [t_start, 1)).
                Standard reflow uses 0.0; biasing toward [0.5, 1) emphasises the
                shallow-diffusion regime the model is used in at inference.

        Returns:
            (loss, {"reflow_loss": float}).
        """
        loss = self.reflow_model(
            condition=cond_mel,
            gt_spec=real_mel,
            infer=False,
            t_start=t_start,
        )
        return loss, {"reflow_loss": float(loss.item())}

    # -- inference ----------------------------------------------------------

    def sample(
        self,
        sovits_mel: torch.Tensor,
        t_start: float = 0.8,
        infer_step: int = 10,
        method: str = "euler",
    ) -> torch.Tensor:
        """Shallow-diffuse ``sovits_mel`` toward a clean mel.

        Args:
            sovits_mel: [B, T, M] or [T, M] sovits/rvc output mel (dB domain).
            t_start: ODE start time in [0, 1).  0 = full generation, →1 = keep
                input.  Default 0.8 (conservative shallow diffusion).
            infer_step: ODE steps (10 is plenty for euler).
            method: 'euler' (fast) or 'rk4' (higher order).

        Returns:
            Denoised mel [B, T, M] (dB domain), same layout as input.
        """
        squeeze = sovits_mel.dim() == 2
        if squeeze:
            sovits_mel = sovits_mel.unsqueeze(0)  # [1, T, M]
        with torch.no_grad():
            mel = self.reflow_model(
                condition=sovits_mel,
                gt_spec=sovits_mel,  # start point; reflow mixes with noise
                infer=True,
                infer_step=infer_step,
                method=method,
                t_start=t_start,
                use_tqdm=False,
            )
        if squeeze:
            mel = mel.squeeze(0)
        return mel  # [B, T, M] (dB domain)

    # -- pretrained loading -------------------------------------------------

    def load_ddsp6_reflow(self, ckpt_path: str) -> dict:
        """Load ``reflow_model.*`` weights from a DDSP6 checkpoint.

        The LYNXNet2 velocity network is architecturally identical, so all
        ``reflow_model.*`` keys transfer.  Returns the load report
        (missing/unexpected counts).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # only reflow_model keys are expected from a DDSP6 ckpt
        bad_missing = [k for k in missing if k.startswith("reflow_model.")]
        report = {
            "loaded_keys": len(sd),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "missing_reflow": len(bad_missing),
        }
        if bad_missing:
            report["missing_reflow_keys"] = bad_missing[:8]
        return report
