"""rvc LightningModule (GAN: VITS + NSF-HiFiGAN, RVC v2)."""
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.gan_module import BaseGANModule
from ...train.losses import gan_d_loss, gan_g_loss
from .._defaults import RVC_UPSAMPLE as _RVC_UPSAMPLE
from .model import MultiPeriodDiscriminatorV2, RVCModel


class RVCModule(BaseGANModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts
        version, sr = str(opts.get("rvc_version", "v2")), int(opts.get("sr", 48000))
        if "upsample_rates" not in opts or "upsample_kernel_sizes" not in opts:
            try:
                up, k = _RVC_UPSAMPLE[(version, sr)]
            except KeyError:
                raise ValueError(f"no RVC upsample defaults for ({version}, {sr}); "
                                 f"supported: {sorted(_RVC_UPSAMPLE)}")
        else:
            up, k = opts["upsample_rates"], opts["upsample_kernel_sizes"]
        # v1 = 256-d units（contentvec_proj），v2 = 768-d；默认版本原生 dim
        self.generator = RVCModel(
            inter_channels=opts.get("inter_channels", 192),
            hidden_channels=opts.get("hidden_channels", 192),
            filter_channels=opts.get("filter_channels", 768),
            n_units=opts.get("n_units", 256 if version == "v1" else 768),
            spk_embed_dim=opts.get("num_speaker", 1),
            segment_size=opts.get("segment_size", 20),
            sr=sr, upsample_rates=up, upsample_kernel_sizes=k,
            vol_embedding=opts.get("vol_embedding", False),
        )
        self.discriminator = MultiPeriodDiscriminatorV2(
            use_spectral_norm=opts.get("use_spectral_norm", False))
        # rvc 训练加速（不改语义）：
        # (1) D 的 Conv2d 权重转 channels_last + patch DiscriminatorP.forward 保持
        #     CL 输入 → 消除 nchw↔nhwc 转换胶水（实测 −5%）。
        # (2) compile enc_p/dec（SineGen 已 Triton fused kernel，无 cumsum blocker）。
        # (3) batched_disc 数学等价（conv 按 batch 维独立）但实测 wall time 负优化
        #     （计算翻倍 > 效率提升）；保留开关、默认关、不推荐。
        if opts.get("channels_last_disc", False):
            import torch.nn as _nn
            for _mod in self.discriminator.modules():
                if isinstance(_mod, _nn.Conv2d):
                    _mod.weight.data = _mod.weight.data.to(memory_format=torch.channels_last)
            from ..sovits import model as _sm
            _LRELU = 0.1
            def _cl_forward(_self, x):
                import torch.nn.functional as _F
                fmap = []; b, c, t = x.shape
                period = _self.period
                if t % period != 0:
                    x = _F.pad(x, (0, period - (t % period)), "reflect"); t += period - (t % period)
                x = x.view(b, c, t // period, period).to(memory_format=torch.channels_last)
                for l in _self.convs:
                    x = l(x).to(memory_format=torch.channels_last)
                    x = _F.leaky_relu(x, _LRELU); fmap.append(x)
                x = _self.conv_post(x); fmap.append(x)
                return torch.flatten(x, 1, -1), fmap
            _sm.DiscriminatorP.forward = _cl_forward
        if opts.get("compile_enc_dec", False):
            self.generator.enc_p = torch.compile(
                self.generator.enc_p, mode=opts.get("compile_mode", "default"), dynamic=False)
            self.generator.dec = torch.compile(
                self.generator.dec, mode=opts.get("compile_mode", "default"), dynamic=False)
        if opts.get("batched_disc", False):
            from .model import MultiPeriodDiscriminatorV2 as _MPD
            def _batched_forward(_self, y, y_hat):
                B = y.shape[0]
                cat = torch.cat([y, y_hat], dim=0)
                rs, gs, fr, fg = [], [], [], []
                for d in _self.discriminators:
                    out, fmap = d(cat)
                    rs.append(out[:B]); gs.append(out[B:])
                    fr.append([f[:B] for f in fmap]); fg.append([f[B:] for f in fmap])
                return rs, gs, fr, fg
            _MPD.forward = _batched_forward
        self.c_mel, self.c_kl = opts.get("c_mel", 45.0), opts.get("c_kl", 1.0)
        # RVC mel_processing：clip_val=2e-6, eps=2e-7（与 so-vits 1e-5/1e-6 不同）
        self._mel_args = (
            opts.get("filter_length", 2048), opts.get("n_mel_channels", 128),
            opts.get("sampling_rate", 48000), opts.get("hop_length", 480),
            opts.get("win_length", 2048), opts.get("mel_fmin", 0.0),
            opts.get("mel_fmax", None))
        self._mel_clip_val = opts.get("mel_clip_val", 2e-6)
        self._mel_eps = opts.get("mel_eps", 2e-7)
        self._hop = opts.get("hop_length", 480)

    def _prepare_inputs(self, batch: SVCBatch):
        return batch.units, batch.pitch_token, batch.f0, batch.mel, batch.wav, batch.spk_id

    def forward_generator(self, batch: SVCBatch):
        # 每 step 只跑一次 net_g，输出同时供 D loss（wave_gen.detach()）和 G loss
        # （wave_gen 带梯度）——见 BaseGANModule 严格交替顺序
        units, pitch_token, pitchf, spec, wave, spk_id = self._prepare_inputs(batch)
        return self.generator(units, pitch_token, pitchf, spec, wave, spk_id,
                              frame_len=batch.frame_len, vol=batch.rms)  # vol 仅 vol_embedding=True

    def generator_loss(self, batch: SVCBatch, out=None):
        if out is None:
            out = self.forward_generator(batch)
        return gan_g_loss(self.discriminator, out, batch, self._mel_args, self.c_mel,
                          self.c_kl, self._hop, mel_clip_val=self._mel_clip_val,
                          mel_eps=self._mel_eps)

    def discriminator_loss(self, batch: SVCBatch, out=None):
        if out is None:
            with torch.no_grad():
                out = self.forward_generator(batch)
        return gan_d_loss(self.discriminator, out, batch, self._hop)

    def sample(self, batch: SVCBatch):
        units, pitch_token, pitchf, spec, wave, spk_id = self._prepare_inputs(batch)
        with torch.no_grad():
            phone_lengths = torch.full(
                (units.shape[0],), units.shape[1], device=units.device, dtype=torch.long)
            out, *_ = self.generator.infer(units, phone_lengths, pitch_token, pitchf, spk_id,
                                           vol=batch.rms)  # vol 仅 vol_embedding=True
        return out