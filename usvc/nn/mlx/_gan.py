"""MLX GAN discriminators shared by sovits and rvc backends (E5 dedup).

Verbatim move of the DiscriminatorS / DiscriminatorP / MultiPeriodDiscriminator
classes that were defined twice (sovits.py and rvc.py); only the periods list
differs between so-vits (2,3,5,7,11) and RVC v2 (2,3,5,7,11,17,23,37).
get_padding comes from usvc/nn/mlx/nsf_hifigan.py (single definition).

Torch-side counterpart: rvc/model.py already reuses the so-vits model's
DiscriminatorS/P — this module is the MLX-side equivalent of that dedup.

NOTE (frozen surface): spectral_norm is accepted but not implemented in MLX
(callers pass use_spectral_norm=False); keep the flag for API parity.
"""
from __future__ import annotations

import mlx.nn as nn

from .nsf_hifigan import get_padding

__all__ = ["DiscriminatorP", "DiscriminatorS", "MultiPeriodDiscriminator"]


class DiscriminatorP(nn.Module):
    """Period discriminator(so-vits DiscriminatorP)。Conv2d NLC→转 NLC 4D。

    注意 mlx Conv2d 输入 [B, H, W, C](NLC 4D);torch 是 [B, C, H, W]。
    此处输入 wav [B, T, 1] → reshape [B, t/period, period, 1]。
    """

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
        import mlx.core as _mx
        fmap = []
        B, T, C = x.shape
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            import numpy as _np
            x_np = _np.pad(_np.array(x), [(0, 0), (0, n_pad), (0, 0)], mode="reflect")
            x = _mx.array(x_np)
            T = T + n_pad
        x = x.reshape(B, T // self.period, self.period, C)   # [B, H, W, C]
        for l in self.convs:
            x = l(x)
            x = _mx.maximum(0.1 * x, x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.reshape(B, -1)
        return x, fmap


class DiscriminatorS(nn.Module):
    """Scale discriminator(so-vits DiscriminatorS)。Conv1d NLC。"""

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
        import mlx.core as _mx
        fmap = []
        for l in self.convs:
            x = l(x)
            x = _mx.maximum(0.1 * x, x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.reshape(x.shape[0], -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """MPD: DiscriminatorS + DiscriminatorP(periods)。

    so-vits periods (2,3,5,7,11);RVC v2 passes (2,3,5,7,11,17,23,37) via the
    MultiPeriodDiscriminatorV2 subclass in backends/rvc.py。
    """

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm: bool = False):
        super().__init__()
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = discs

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