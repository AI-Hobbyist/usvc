"""rvc LightningModule (GAN: VITS + NSF-HiFiGAN, RVC v2)."""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.gan_module import BaseGANModule
from ...train.losses import gan_d_loss, gan_g_loss
from .model import MultiPeriodDiscriminatorV2, RVCModel

__all__ = ["RVCModule"]


# RVC upsample configs by (version, sr): prod(upsample_rates) == hop (sr/100)
# upsample 默认表共享于 backends._defaults(was 本文件内嵌副本)
from .._defaults import RVC_UPSAMPLE as _RVC_UPSAMPLE


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
        # (2) torch.compile enc_p/dec（SineGen 已用 Triton fused kernel，无 cumsum blocker）。
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
        the same `out`), then backward+step. `out` comes from
        forward_generator(batch); when None it is computed here (compat path).
        """
        if out is None:
            out = self.forward_generator(batch)
        return gan_g_loss(
            self.discriminator, out, batch, self._mel_args, self.c_mel, self.c_kl,
            self._hop, mel_clip_val=self._mel_clip_val, mel_eps=self._mel_eps,
        )

    def discriminator_loss(self, batch: SVCBatch, out=None):
        """D loss: LSGAN on real vs fake (strict alternating order, see above)."""
        if out is None:
            with torch.no_grad():
                out = self.forward_generator(batch)
        return gan_d_loss(self.discriminator, out, batch, self._hop)

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
