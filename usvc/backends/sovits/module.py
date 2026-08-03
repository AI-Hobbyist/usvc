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
from .model import MultiPeriodDiscriminator, SoVITSModel, mel_spectrogram, spec_to_mel

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
            filter_channels=opts.get("filter_channels", 768),
            n_heads=opts.get("n_heads", 2),
            n_layers_dec=opts.get("n_layers_dec", 6),
            n_layers_enc=opts.get("n_layers_enc", 6),
            n_sdp_layers0=opts.get("n_sdp_layers0", 6),
            n_sdp_layers=opts.get("n_sdp_layers", 6),
            n_spk=opts.get("num_speaker", 1),
            segment_size=opts.get("segment_size", 10240),
            gin_channels=opts.get("gin_channels", 768),
            use_automatic_f0_prediction=opts.get("use_automatic_f0_prediction", True),
        )
        self.discriminator = MultiPeriodDiscriminator(
            use_spectral_norm=opts.get("use_spectral_norm", False),
        )

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
        )

    def generator_loss(self, batch: SVCBatch, out=None):
        """G loss: adv + feature_match + mel_L1 + kl (+ optional lf0).

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
        wave_gen = out["wave_gen"]  # [B, 1, segment*hop]
        spec = batch.mel  # linear spec [B, spec_channels, T] (sovits convention)
        wave = batch.wav

        # Slice real wave to match generated segment
        ids_slice = out["ids_slice"]  # [B] start indices
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

        # Discriminator on real and fake
        real_logits, fake_logits, real_fmaps, fake_fmaps = self.discriminator(
            wave_real_sliced.detach(), wave_gen
        )

        # GAN generator loss
        loss_gen = lsgan_gen_loss(fake_logits)
        # Feature matching
        loss_fm = feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)

        # Mel loss (L1 between real mel and mel re-extracted from wave_gen)
        # so-vits: y_mel = slice_segments(spec_to_mel(spec), ids_slice, seg_frames)
        #          y_hat_mel = mel_spectrogram_torch(wave_gen, ...)
        #          loss_mel = L1(y_mel, y_hat_mel) * c_mel
        n_fft, n_mels, sr, hop, win, fmin, fmax = self._mel_args
        seg_frames = segment // hop
        mel_full = spec_to_mel(spec, n_fft, n_mels, sr, fmin, fmax)  # [B, n_mels, T]
        # Vectorized frame slice (CUDA-graph friendly: no .item() sync, no python loop).
        # Training guarantees ids_slice + seg_frames <= T (frame_len).
        idx_mel = ids_slice.unsqueeze(1) + torch.arange(seg_frames, device=mel_full.device)  # [B, seg_frames]
        y_mel = mel_full.gather(
            2, idx_mel.unsqueeze(1).expand(mel_full.shape[0], mel_full.shape[1], seg_frames)
        )  # [B, n_mels, seg_frames]
        y_hat_mel = mel_spectrogram(
            wave_gen.squeeze(1), n_fft, n_mels, sr, hop, win, fmin, fmax
        )
        loss_mel = F.l1_loss(y_mel, y_hat_mel)

        # KL loss
        loss_kl = kl_loss(
            out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"]
        )

        # f0 prediction loss (so-vits: MSE(pred_lf0, lf0) when f0 decoder enabled)
        pred_lf0 = out.get("pred_lf0")
        if isinstance(pred_lf0, torch.Tensor):
            loss_lf0 = F.mse_loss(pred_lf0, out["lf0"])
        else:
            loss_lf0 = torch.tensor(0.0, device=wave_gen.device)

        loss_g_total = (
            loss_gen + loss_fm + self.c_mel * loss_mel + self.c_kl * loss_kl + loss_lf0
        )

        sub = {
            "g/gan": loss_gen.item(),
            "g/fm": loss_fm.item(),
            "g/mel": loss_mel.item(),
            "g/kl": loss_kl.item(),
            "g/lf0": loss_lf0.item(),
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
        spec = batch.mel  # [B, spec_channels, T]
        wave = batch.wav
        ids_slice = out["ids_slice"]
        segment = wave_gen.shape[-1]
        hop = self.cfg.backend_opts.get("hop", 512)

        if wave.dim() == 2:
            wave_real = wave.unsqueeze(1)
        else:
            wave_real = wave
        # Vectorized slicing (CUDA-graph friendly: no .item() sync, no python loop).
        # Training guarantees ids_slice*hop + segment <= wave length
        # (rand_slice_segments_with_pitch samples ids_str in [0, x_lengths-segment_size]).
        starts = ids_slice * hop  # [B] sample start indices
        idx = starts.unsqueeze(1) + torch.arange(segment, device=wave_real.device)  # [B, segment]
        wave_real_sliced = wave_real[:, 0, :].gather(1, idx).unsqueeze(1)  # [B, 1, segment]

        real_logits, fake_logits, _, _ = self.discriminator(wave_real_sliced, wave_gen)
        loss_d = lsgan_disc_loss(real_logits, fake_logits)
        return loss_d, {"d/gan": loss_d.item()}

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
