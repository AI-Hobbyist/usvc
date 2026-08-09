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


# RVC upsample configs by (version, sr): prod(upsample_rates) == hop (sr/100)
_RVC_UPSAMPLE = {
    ("v1", 32000): ([10, 4, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v1", 40000): ([10, 10, 2, 2], [16, 16, 4, 4]),
    ("v1", 48000): ([10, 6, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v2", 32000): ([10, 8, 2, 2], [20, 16, 4, 4]),
    ("v2", 48000): ([12, 10, 2, 2], [24, 20, 4, 4]),
}


class RVCModule(BaseGANModule):
    """LightningModule for rvc (VITS GAN training)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts

        version = str(opts.get("rvc_version", "v2"))
        sr = int(opts.get("sr", 48000))
        if "upsample_rates" not in opts or "upsample_kernel_sizes" not in opts:
            try:
                up, k = _RVC_UPSAMPLE[(version, sr)]
            except KeyError:
                raise ValueError(
                    f"no RVC upsample defaults for ({version}, {sr}); "
                    f"supported: {sorted(_RVC_UPSAMPLE)}"
                )
        else:
            up, k = opts["upsample_rates"], opts["upsample_kernel_sizes"]

        # RVC v1 uses 256-d units (contentvec_proj / TextEncoder256);
        # v2 uses 768-d (last_hidden_state). Default to the version's native dim
        # unless the user explicitly overrides n_units.
        default_n_units = 256 if version == "v1" else 768
        self.generator = RVCModel(
            inter_channels=opts.get("inter_channels", 192),
            hidden_channels=opts.get("hidden_channels", 192),
            filter_channels=opts.get("filter_channels", 768),
            n_units=opts.get("n_units", default_n_units),
            spk_embed_dim=opts.get("num_speaker", 1),
            segment_size=opts.get("segment_size", 20),
            sr=sr,
            upsample_rates=up,
            upsample_kernel_sizes=k,
            vol_embedding=opts.get("vol_embedding", False),
        )
        self.discriminator = MultiPeriodDiscriminatorV2(
            use_spectral_norm=opts.get("use_spectral_norm", False),
        )

        # --- rvc 训练加速（不改语义）---
        # (1) discriminator 的 Conv2d 权重转 channels_last，+ patch DiscriminatorP.forward
        #     让输入也走 channels_last → 消除 nchw↔nhwc 转换胶水（实测 −5%）。
        # (2) torch.compile enc_p/dec（m_source 已 @disable 避 cumsum blocker）。
        # (3) batched_disc：real+fake 拼成 2B 一次过判别器 → 小 conv 填满 SM，
        #     TFLOPS 利用率提升（37.8%→46.1%）。wall time 持平但有效计算占比上升。
        if opts.get("channels_last_disc", False):
            import torch.nn as _nn
            for _mod in self.discriminator.modules():
                if isinstance(_mod, _nn.Conv2d):
                    _mod.weight.data = _mod.weight.data.to(
                        memory_format=torch.channels_last)
            # monkey-patch DiscriminatorP.forward 保持输入 channels_last
            from ..sovits import model as _sm
            _orig_dp_forward = _sm.DiscriminatorP.forward
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
            _sm.DiscriminatorP.forward = _cl_forward
        if opts.get("compile_enc_dec", False):
            self.generator.enc_p = torch.compile(
                self.generator.enc_p, mode=opts.get("compile_mode", "default"),
                dynamic=False)
            self.generator.dec = torch.compile(
                self.generator.dec, mode=opts.get("compile_mode", "default"),
                dynamic=False)
        # (3) batched_disc: real+fake 拼成 2B 一次过判别器。
        # ⚠ 实测 wall time 负优化（计算量翻倍 > 效率提升），保留开关但默认关、不推荐。
        # monkey-patch MultiPeriodDiscriminatorV2.forward（数学等价：conv 是 batch 维独立的）。
        if opts.get("batched_disc", False):
            from .model import MultiPeriodDiscriminatorV2 as _MPD
            def _batched_forward(_self, y, y_hat):
                B = y.shape[0]
                cat = torch.cat([y, y_hat], dim=0)
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
                for d in _self.discriminators:
                    out, fmap = d(cat)
                    y_d_rs.append(out[:B]); y_d_gs.append(out[B:])
                    fmap_rs.append([f[:B] for f in fmap]); fmap_gs.append([f[B:] for f in fmap])
                return y_d_rs, y_d_gs, fmap_rs, fmap_gs
            _MPD.forward = _batched_forward

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
            units, pitch_token, pitchf, spec, wave, spk_id, frame_len=batch.frame_len,
            vol=batch.rms,  # [B, T] volume (only used if vol_embedding=True)
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
        hop = self.cfg.backend_opts.get("hop_length", 480)

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
        hop = self.cfg.backend_opts.get("hop_length", 480)

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
            phone_lengths = torch.full(
                (units.shape[0],), units.shape[1], device=units.device, dtype=torch.long
            )
            out, *_ = self.generator.infer(
                units, phone_lengths, pitch_token, pitchf, spk_id,
                vol=batch.rms,  # [1, T] volume (only used if vol_embedding=True)
            )
        return out
