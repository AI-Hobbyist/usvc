"""sovits LightningModule (GAN: VITS + NSF-HiFiGAN).

Uses BaseGANModule with manual G/D optimization.
"""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.gan_module import BaseGANModule
from ...train.losses import gan_d_loss, gan_g_loss
from .model import MultiPeriodDiscriminator, SoVITSModel

__all__ = ["SoVITSModule"]


class SoVITSModule(BaseGANModule):
    """LightningModule for sovits (VITS GAN training)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts

        spec_channels = opts.get("spec_channels", 1025)  # n_fft//2+1
        inter_channels = opts.get("inter_channels", 192)
        hidden_channels = opts.get("hidden_channels", 192)

        self.generator = SoVITSModel(
            n_units=opts.get("n_units", 768),  # contentvec768 (4.1 默认)
            spec_channels=spec_channels,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=opts.get("filter_channels", 768),
            n_heads=opts.get("n_heads", 2),
            n_layers_dec=opts.get("n_layers_dec", 6),
            n_layers_enc=opts.get("n_layers_enc", 6),
            n_sdp_layers0=opts.get("n_sdp_layers0", 6),
            n_sdp_layers=opts.get("n_sdp_layers", 6),
            n_speakers=opts.get("num_speaker", 1),
            segment_size=opts.get("segment_size", 20),  # FRAMES per segment (10240 samples // 512 hop)
            gin_channels=opts.get("gin_channels", 768),
            use_automatic_f0_prediction=opts.get("use_automatic_f0_prediction", True),
            vol_embedding=opts.get("vol_embedding", False),
        )
        self.discriminator = MultiPeriodDiscriminator(
            use_spectral_norm=opts.get("use_spectral_norm", False),
        )

        # --- sovits 训练加速（不改语义；与 rvc 同套策略）---
        # (1) channels_last_disc: discriminator 的 Conv2d 走 channels_last，消除 nchw↔nhwc 胶水。
        # (2) compile_enc_dec: torch.compile enc_p/dec（SineGen 已用 Triton fused kernel）。
        if opts.get("channels_last_disc", False):
            import torch.nn as _nn
            for _mod in self.discriminator.modules():
                if isinstance(_mod, _nn.Conv2d):
                    _mod.weight.data = _mod.weight.data.to(
                        memory_format=torch.channels_last)
            from .model import DiscriminatorP as _DP
            _LRELU = 0.1
            def _cl_forward(_self, x):
                import torch.nn.functional as _F
                fmap = []; b, c, t = x.shape
                period = _self.period
                if t % period != 0:
                    n_pad = period - (t % period)
                    x = _F.pad(x, (0, n_pad), "reflect"); t = t + n_pad
                x = x.view(b, c, t // period, period).to(
                    memory_format=torch.channels_last)
                for l in _self.convs:
                    x = l(x).to(memory_format=torch.channels_last)
                    x = _F.leaky_relu(x, _LRELU); fmap.append(x)
                x = _self.conv_post(x); fmap.append(x)
                x = torch.flatten(x, 1, -1)
                return x, fmap
            _DP.forward = _cl_forward
        if opts.get("compile_enc_dec", False):
            self.generator.enc_p = torch.compile(
                self.generator.enc_p, mode=opts.get("compile_mode", "default"),
                dynamic=False)
            self.generator.dec = torch.compile(
                self.generator.dec, mode=opts.get("compile_mode", "default"),
                dynamic=False)
            self.generator.flow = torch.compile(
                self.generator.flow, mode=opts.get("compile_mode", "default"),
                dynamic=False)

        # Loss weights
        self.c_mel = opts.get("c_mel", 45.0)
        self.c_kl = opts.get("c_kl", 1.0)

        # Mel extraction params (so-vits config_template.json data section)
        self._mel_args = (
            opts.get("filter_length", 2048),
            opts.get("n_mel_channels", 80),
            opts.get("sampling_rate", 44100),
            opts.get("hop_length", 512),
            opts.get("win_length", 2048),
            opts.get("mel_fmin", 0.0),
            opts.get("mel_fmax", 22050.0),
        )
        self._hop = opts.get("hop_length", 512)

    def forward_generator(self, batch: SVCBatch):
        """Single generator forward per step, shared by D and G losses.

        Matches so-vits-svc train.py: net_g is called once per step and its
        outputs (wave_gen, ids_slice, z_p, ...) are reused for both the D loss
        (on wave_gen.detach()) and the G loss (on wave_gen with grad).
        """
        return self.generator(
            batch.units, batch.f0, batch.mel, batch.wav, batch.spk_id,
            frame_len=batch.frame_len,
            vol=batch.rms,  # [B, T] volume (only used if vol_embedding=True)
            uv=batch.uv,  # [B, T] voicing from raw f0 (pre-interpolation)
        )

    def generator_loss(self, batch: SVCBatch, out=None):
        """G loss: adv + feature_match + mel_L1 + kl (+ optional lf0).

        Must be used in strict alternating order (see BaseGANModule):
        forward_generator once, then D fwd->bwd->step, then this fwd (reusing
        the same `out`), then backward+step. `out` comes from
        forward_generator(batch); when None it is computed here (compat path).
        """
        if out is None:
            out = self.forward_generator(batch)
        return gan_g_loss(
            self.discriminator, out, batch, self._mel_args, self.c_mel, self.c_kl,
            self._hop, use_lf0=True,
        )

    def discriminator_loss(self, batch: SVCBatch, out=None):
        """D loss: LSGAN on real vs fake (strict alternating order, see above)."""
        if out is None:
            with torch.no_grad():
                out = self.forward_generator(batch)
        return gan_d_loss(self.discriminator, out, batch, self._hop)

    def sample(self, batch: SVCBatch):
        """Generate waveform for validation."""
        with torch.no_grad():
            out, _ = self.generator.infer(
                batch.units.transpose(1, 2),  # [B, D, T] (so-vits convention)
                batch.f0,
                batch.uv,
                batch.spk_id.unsqueeze(-1),  # [B, 1]
            )
        return out
