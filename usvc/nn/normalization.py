"""Mel spectrogram normalization.

All 7 SVC projects normalize mel to [-1, 1] using the same formula with
spec_min=-12, spec_max=2. This is shared across every backend.
"""
from __future__ import annotations

import torch

__all__ = ["MelNormalizer"]


class MelNormalizer:
    """Normalize mel spectrogram to [-1, 1] range.

    Formula (identical across RIFT/DSP/Diffusion/sovits/RVC):
        norm:   (x - spec_min) / (spec_max - spec_min) * 2 - 1
        denorm: (x + 1) / 2 * (spec_max - spec_min) + spec_min

    Default spec_min=-12, spec_max=2 matches all surveyed projects.
    """

    def __init__(self, spec_min: float = -12.0, spec_max: float = 2.0):
        self.spec_min = spec_min
        self.spec_max = spec_max

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
