"""rvc LightningModule (GAN: VITS + NSF-HiFiGAN, RVC v2)."""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.gan_module import BaseGANModule
from ...train.losses import (
    feature_match_loss,
    kl_loss,
    lsgan_disc_loss,
    lsgan_gen_loss,
)
from .model import MultiPeriodDiscriminatorV2, RVCModel

__all__ = ["RVCModule"]


class RVCModule(BaseGANModule):
    """LightningModule for rvc (VITS GAN training)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts

        self.generator = RVCModel(
            inter_channels=opts.get("inter_channels", 192),
            hidden_channels=opts.get("hidden_channels", 192),
            filter_channels=opts.get("filter_channels", 768),
            n_units=opts.get("n_units", 768),
            spk_embed_dim=opts.get("num_speaker", 1),
            segment_size=opts.get("segment_size", 20),
            sr=opts.get("sr", 48000),
        )
        self.discriminator = MultiPeriodDiscriminatorV2(
            use_spectral_norm=opts.get("use_spectral_norm", False),
        )

        self.c_mel = opts.get("c_mel", 45.0)
        self.c_kl = opts.get("c_kl", 1.0)

    def _prepare_inputs(self, batch: SVCBatch):
        units = batch.units
        # pitch_token from SVCBatch.pitch_token
        pitch_token = batch.pitch_token
        pitchf = batch.f0  # continuous f0
        spec = batch.mel
        wave = batch.wav
        spk_id = batch.spk_id
        return units, pitch_token, pitchf, spec, wave, spk_id

    def generator_loss(self, batch: SVCBatch):
        units, pitch_token, pitchf, spec, wave, spk_id = self._prepare_inputs(batch)

        out = self.generator(
            units, pitch_token, pitchf, spec, wave, spk_id, frame_len=batch.frame_len
        )
        wave_gen = out["wave_gen"]

        # Slice real wave
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
        wave_real_sliced = torch.cat(wave_real_sliced, dim=0)

        real_logits, fake_logits, real_fmaps, fake_fmaps = self.discriminator(
            wave_real_sliced.detach(), wave_gen
        )

        loss_gen = lsgan_gen_loss(fake_logits)
        loss_fm = feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)

        loss_kl = kl_loss(
            out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"]
        )

        loss_mel = torch.tensor(0.0, device=wave_gen.device)

        loss_g_total = loss_gen + loss_fm + self.c_mel * loss_mel + self.c_kl * loss_kl

        sub = {
            "g/gan": loss_gen.item(),
            "g/fm": loss_fm.item(),
            "g/kl": loss_kl.item(),
        }
        return loss_g_total, sub

    def discriminator_loss(self, batch: SVCBatch):
        units, pitch_token, pitchf, spec, wave, spk_id = self._prepare_inputs(batch)

        with torch.no_grad():
            out = self.generator(
                units, pitch_token, pitchf, spec, wave, spk_id, frame_len=batch.frame_len
            )
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
        wave_real_sliced = torch.cat(wave_real_sliced, dim=0)

        real_logits, fake_logits, _, _ = self.discriminator(
            wave_real_sliced, wave_gen
        )
        loss_d = lsgan_disc_loss(real_logits, fake_logits)
        return loss_d, {"d/gan": loss_d.item()}

    def sample(self, batch: SVCBatch):
        units, pitch_token, pitchf, spec, wave, spk_id = self._prepare_inputs(batch)
        with torch.no_grad():
            out = self.generator.infer(units, pitch_token, pitchf, spk_id)
        return out
