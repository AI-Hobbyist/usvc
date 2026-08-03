"""reflow_vae dataset: reuses the diffusion dataset (same feature layout).

Reflow-VAE-SVC consumes mel / f0 / volume(=rms) / units / spk_id — identical
to the Diffusion-SVC family, so the DiffusionDataset is reused directly.
"""
from __future__ import annotations

from ..diffusion.dataset import DiffusionDataset

__all__ = ["ReflowVAEDataset"]


class ReflowVAEDataset(DiffusionDataset):
    """Dataset for reflow_vae backend (Bi_RectifiedFlow + VAE)."""
    pass
