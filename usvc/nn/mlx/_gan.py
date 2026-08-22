"""MLX GAN discriminators shared by sovits and rvc backends (E5 dedup; torch
side rvc/model.py reuses so-vits DiscriminatorS/P — this is the MLX mirror).
Only the periods list differs: so-vits (2,3,5,7,11) vs RVC v2 +17/23/37.
get_padding from nsf_hifigan.py (single definition).

NOTE (frozen surface): spectral_norm accepted but not implemented in MLX
(callers pass use_spectral_norm=False); keep the flag for API parity.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .nsf_hifigan import get_padding

__all__ = ["DiscriminatorP", "DiscriminatorS", "MultiPeriodDiscriminator"]


class DiscriminatorP(nn.Module):
    """Period discriminator。mlx Conv2d 输入 NLC 4D [B,H,W,C];wav [B,T,1]
    reshape [B, t/period, period, 1](torch 是 [B,C,H,W])。"""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        pad = get_padding(kernel_size, 1)
        self.convs = [
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(pad, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(pad, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(pad, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(pad, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(pad, 0)),
        ]
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def __call__(self, x):
        fmap = []
        B, T, C = x.shape
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            import numpy as _np
            x = mx.array(_np.pad(_np.array(x), [(0, 0), (0, n_pad), (0, 0)], mode="reflect"))
            T = T + n_pad
        x = x.reshape(B, T // self.period, self.period, C)  # [B,H,W,C]
        for l in self.convs:
            x = l(x)
            x = mx.maximum(0.1 * x, x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.reshape(B, -1), fmap


class DiscriminatorS(nn.Module):
    """Scale discriminator(Conv1d NLC)。"""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.convs = [
            nn.Conv1d(1, 16, 15, 1, padding=7),
            nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
            nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
            nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
            nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ]
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)

    def __call__(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = mx.maximum(0.1 * x, x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.reshape(x.shape[0], -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """MPD: DiscriminatorS + DiscriminatorP(periods);RVC v2 子类传 8 periods。"""

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = [DiscriminatorS(use_spectral_norm=use_spectral_norm)] + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]

    def __call__(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
