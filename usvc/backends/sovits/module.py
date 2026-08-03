"""sovits LightningModule (GAN: VITS + NSF-HiFiGAN).

Uses BaseGANModule with manual G/D optimization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.gan_module import BaseGANModule
from ...train.losses import (
    feature_match_loss,
    kl_loss,
    lsgan_disc_loss,
    lsgan_gen_loss,
)
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
            n_units=opts.get("n_units", 256),
            spec_channels=spec_channels,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            n_heads=opts.get("n_heads", 2),
            n_layers_dec=opts.get("n_layers_dec", 6),
            n_layers_enc=opts.get("n_layers_enc", 6),
            n_sdp_layers0=opts.get("n_sdp_layers0", 6),
            n_sdp_layers=opts.get("n_sdp_layers", 6),
            n_spk=opts.get("num_speaker", 1),
            segment_size=opts.get("segment_size", 10240),
        )
        self.discriminator = MultiPeriodDiscriminator(
            use_spectral_norm=opts.get("use_spectral_norm", False),
        )

        # Loss weights
        self.c_mel = opts.get("c_mel", 45.0)
        self.c_kl = opts.get("c_kl", 1.0)

    def generator_loss(self, batch: SVCBatch):
        """G loss: adv + feature_match + mel_L1 + kl (+ optional lf0)."""
        units = batch.units
        f0 = batch.f0
        spec = batch.mel  # linear spec [B, spec_channels, T] (sovits convention)
        wave = batch.wav
        spk_id = batch.spk_id

        # Generator forward
        out = self.generator(units, f0, spec, wave, spk_id, frame_len=batch.frame_len)
        wave_gen = out["wave_gen"]  # [B, 1, segment*hop]

        # Slice real wave to match generated segment
        ids_slice = out["ids_slice"]  # [B] start indices
        segment = wave_gen.shape[-1]
        hop = self.cfg.backend_opts.get("hop", 512)

        if wave.dim() == 2:
            wave_real = wave.unsqueeze(1)
        else:
            wave_real = wave
        # Slice: take the same segment the generator produced
        wave_real_sliced = []
        for i in range(wave_real.shape[0]):
            start = int(ids_slice[i].item()) * hop
            end = start + segment
            if end <= wave_real.shape[-1] and start >= 0:
                wave_real_sliced.append(wave_real[i : i + 1, :, start:end])
            else:
                wave_real_sliced.append(wave_real[i : i + 1, :, :segment])
        wave_real_sliced = torch.cat(wave_real_sliced, dim=0)

        # Discriminator on real and fake
        real_logits, fake_logits, real_fmaps, fake_fmaps = self.discriminator(
            wave_real_sliced.detach(), wave_gen
        )

        # GAN generator loss
        loss_gen = lsgan_gen_loss(fake_logits)
        # Feature matching
        loss_fm = feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)

        # Mel loss (L1 between generated and real spectrogram)
        # Full sovits re-extracts mel from wave_gen via mel_spectrogram_torch.
        # For now use zero; full impl would need mel extraction from generated wave.
        loss_mel = torch.tensor(0.0, device=wave_gen.device)

        # KL loss
        loss_kl = kl_loss(
            out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"]
        )

        loss_g_total = loss_gen + loss_fm + self.c_mel * loss_mel + self.c_kl * loss_kl

        sub = {
            "g/gan": loss_gen.item(),
            "g/fm": loss_fm.item(),
            "g/mel": loss_mel.item() if isinstance(loss_mel, torch.Tensor) else loss_mel,
            "g/kl": loss_kl.item(),
        }
        return loss_g_total, sub

    def discriminator_loss(self, batch: SVCBatch):
        """D loss: LSGAN on real vs fake."""
        units = batch.units
        f0 = batch.f0
        spec = batch.mel
        wave = batch.wav
        spk_id = batch.spk_id

        spec = batch.mel  # [B, spec_channels, T]

        with torch.no_grad():
            out = self.generator(units, f0, spec, wave, spk_id, frame_len=batch.frame_len)
            wave_gen = out["wave_gen"].detach()

        ids_slice = out["ids_slice"]
        segment = wave_gen.shape[-1]
        hop = self.cfg.backend_opts.get("hop", 512)

        if wave.dim() == 2:
            wave_real = wave.unsqueeze(1)
        else:
            wave_real = wave
        wave_real_sliced = []
        for i in range(wave_real.shape[0]):
            start = int(ids_slice[i].item()) * hop
            end = start + segment
            if end <= wave_real.shape[-1] and start >= 0:
                wave_real_sliced.append(wave_real[i : i + 1, :, start:end])
            else:
                wave_real_sliced.append(wave_real[i : i + 1, :, :segment])
        wave_real_sliced = (
            torch.cat(wave_real_sliced, dim=0)
        )

        real_logits, fake_logits, _, _ = self.discriminator(wave_real_sliced, wave_gen)
        loss_d = lsgan_disc_loss(real_logits, fake_logits)
        return loss_d, {"d/gan": loss_d.item()}

    def sample(self, batch: SVCBatch):
        """Generate waveform for validation."""
        with torch.no_grad():
            out = self.generator.infer(
                batch.units, batch.f0, batch.spk_id, uv=batch.uv, max_length=None
            )
        return out
