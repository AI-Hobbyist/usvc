"""rvc LightningModule (GAN: VITS + NSF-HiFiGAN, RVC v2)."""
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
from ..sovits.model import mel_spectrogram, spec_to_mel
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

        # Mel extraction params (RVC v2 48k.json data section). RVC mel_processing
        # uses clip_val=2e-6 and eps=2e-7 (differs from so-vits 1e-5/1e-6).
        self._mel_args = (
            opts.get("filter_length", 2048),
            opts.get("n_mel_channels", 128),
            opts.get("sampling_rate", 48000),
            opts.get("hop_length", 480),
            opts.get("win_length", 2048),
            opts.get("mel_fmin", 0.0),
            opts.get("mel_fmax", None),
        )
        self._mel_clip_val = opts.get("mel_clip_val", 2e-6)
        self._mel_eps = opts.get("mel_eps", 2e-7)
        self._hop = opts.get("hop_length", 480)

    def _prepare_inputs(self, batch: SVCBatch):
        units = batch.units
        # pitch_token from SVCBatch.pitch_token
        pitch_token = batch.pitch_token
        pitchf = batch.f0  # continuous f0
        spec = batch.mel
        wave = batch.wav
        spk_id = batch.spk_id
        return units, pitch_token, pitchf, spec, wave, spk_id

    def forward_generator(self, batch: SVCBatch):
        """Single generator forward per step, shared by D and G losses.

        Matches RVC train.py: net_g is called once per step and its outputs
        (wave_gen, ids_slice, z_p, ...) are reused for both the D loss (on
        wave_gen.detach()) and the G loss (on wave_gen with grad).
        """
        units, pitch_token, pitchf, spec, wave, spk_id = self._prepare_inputs(batch)
        return self.generator(
            units, pitch_token, pitchf, spec, wave, spk_id, frame_len=batch.frame_len
        )

    def generator_loss(self, batch: SVCBatch, out=None):
        """G loss: adv + feature_match + mel_L1 + kl (RVC: no lf0).

        Must be used in strict alternating order (see BaseGANModule):
        forward_generator once, then D fwd->bwd->step, then this fwd (reusing
        the same `out`), then backward+step. Building both loss graphs before
        any optimizer step raises an autograd version-check RuntimeError on
        the later backward (AdamW mutates weight_norm params in-place).

        `out` comes from forward_generator(batch); when None it is computed
        here (compat path: two generator forwards per step, old behaviour).
        """
        if out is None:
            out = self.forward_generator(batch)
        wave_gen = out["wave_gen"]
        spec = batch.mel
        wave = batch.wav

        # Slice real wave
        ids_slice = out["ids_slice"]
        segment = wave_gen.shape[-1]
        hop = self.cfg.backend_opts.get("hop", 512)

        if wave.dim() == 2:
            wave_real = wave.unsqueeze(1)
        else:
            wave_real = wave
        # Vectorized slicing (CUDA-graph friendly: no .item() sync, no python loop).
        # Training guarantees ids_slice*hop + segment <= wave length.
        starts = ids_slice * hop  # [B] sample start indices
        idx = starts.unsqueeze(1) + torch.arange(segment, device=wave_real.device)  # [B, segment]
        wave_real_sliced = wave_real[:, 0, :].gather(1, idx).unsqueeze(1)  # [B, 1, segment]

        real_logits, fake_logits, real_fmaps, fake_fmaps = self.discriminator(
            wave_real_sliced.detach(), wave_gen
        )

        loss_gen = lsgan_gen_loss(fake_logits)
        loss_fm = feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)

        loss_kl = kl_loss(
            out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"]
        )

        # Mel loss (RVC: y_mel = slice(spec_to_mel(spec)), y_hat_mel = mel(wave_gen))
        n_fft, n_mels, sr, hop, win, fmin, fmax = self._mel_args
        seg_frames = segment // hop
        mel_full = spec_to_mel(
            spec, n_fft, n_mels, sr, fmin, fmax, clip_val=self._mel_clip_val
        )  # [B, n_mels, T]
        # Vectorized frame slice (CUDA-graph friendly: no .item() sync, no python loop).
        # Training guarantees ids_slice + seg_frames <= T (frame_len).
        idx_mel = ids_slice.unsqueeze(1) + torch.arange(seg_frames, device=mel_full.device)  # [B, seg_frames]
        y_mel = mel_full.gather(
            2, idx_mel.unsqueeze(1).expand(mel_full.shape[0], mel_full.shape[1], seg_frames)
        )  # [B, n_mels, seg_frames]
        y_hat_mel = mel_spectrogram(
            wave_gen.squeeze(1), n_fft, n_mels, sr, hop, win, fmin, fmax,
            clip_val=self._mel_clip_val, eps=self._mel_eps,
        )
        loss_mel = F.l1_loss(y_mel, y_hat_mel)

        loss_g_total = loss_gen + loss_fm + self.c_mel * loss_mel + self.c_kl * loss_kl

        sub = {
            "g/gan": loss_gen.item(),
            "g/fm": loss_fm.item(),
            "g/mel": loss_mel.item(),
            "g/kl": loss_kl.item(),
        }
        return loss_g_total, sub

    def discriminator_loss(self, batch: SVCBatch, out=None):
        """D loss: LSGAN on real vs fake.

        Must be used in strict alternating order (see BaseGANModule): this fwd
        (on out["wave_gen"].detach()), then backward+step, before the G loss's
        forward (D fwd#2). `out` comes from forward_generator(batch); when
        None it is computed here under no_grad (compat path: two generator
        forwards per step, old behaviour).
        """
        if out is None:
            with torch.no_grad():
                out = self.forward_generator(batch)
        wave_gen = out["wave_gen"].detach()
        spec = batch.mel
        wave = batch.wav
        ids_slice = out["ids_slice"]
        segment = wave_gen.shape[-1]
        hop = self.cfg.backend_opts.get("hop", 512)

        if wave.dim() == 2:
            wave_real = wave.unsqueeze(1)
        else:
            wave_real = wave
        # Vectorized slicing (CUDA-graph friendly: no .item() sync, no python loop).
        # Training guarantees ids_slice*hop + segment <= wave length.
        starts = ids_slice * hop  # [B] sample start indices
        idx = starts.unsqueeze(1) + torch.arange(segment, device=wave_real.device)  # [B, segment]
        wave_real_sliced = wave_real[:, 0, :].gather(1, idx).unsqueeze(1)  # [B, 1, segment]

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
