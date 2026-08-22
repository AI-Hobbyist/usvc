"""reflow_vae dataset: same feature layout as the Diffusion family.

Reflow-VAE-SVC consumes mel / f0 / volume(=rms) / units / spk_id — identical
to Diffusion-SVC, so DiffusionDataset is reused directly.
"""
from __future__ import annotations

from ..diffusion.dataset import DiffusionDataset

__all__ = ["ReflowVAEDataset"]


class ReflowVAEDataset(DiffusionDataset):
    pass