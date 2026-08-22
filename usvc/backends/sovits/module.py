# sovits LightningModule: GAN (VITS + NSF-HiFiGAN)，BaseGANModule 手动 G/D 交替优化
from __future__ import annotations

import torch

from ...train.config import TrainConfig
from ...train.dataset import SVCBatch
from ...train.gan_module import BaseGANModule
from ...train.losses import gan_d_loss, gan_g_loss
from .model import MultiPeriodDiscriminator, SoVITSModel

__all__ = ["SoVITSModule"]


class SoVITSModule(BaseGANModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        opts = cfg.backend_opts
        self.generator = SoVITSModel(
            n_units=opts.get("n_units", 768),  # contentvec768
            spec_channels=opts.get("spec_channels", 1025),  # n_fft//2+1
            inter_channels=opts.get("inter_channels", 192),
            hidden_channels=opts.get("hidden_channels", 192),
            filter_channels=opts.get("filter_channels", 768),
            n_heads=opts.get("n_heads", 2),
            n_layers_dec=opts.get("n_layers_dec", 6), n_layers_enc=opts.get("n_layers_enc", 6),
            n_sdp_layers0=opts.get("n_sdp_layers0", 6), n_sdp_layers=opts.get("n_sdp_layers", 6),
            n_speakers=opts.get("num_speaker", 1),
            segment_size=opts.get("segment_size", 20),  # FRAMES per segment (10240 samples // 512 hop)
            gin_channels=opts.get("gin_channels", 768),
            use_automatic_f0_prediction=opts.get("use_automatic_f0_prediction", True),
            vol_embedding=opts.get("vol_embedding", False))
        self.discriminator = MultiPeriodDiscriminator(use_spectral_norm=opts.get("use_spectral_norm", False))
        # 训练加速（与 rvc 同套策略，不改语义）：disc Conv2d 走 channels_last；
        # enc_p/dec/flow torch.compile（SineGen 已用 Triton fused kernel 可编译）
        if opts.get("channels_last_disc", False):
            import torch.nn as _nn
            for _mod in self.discriminator.modules():
                if isinstance(_mod, _nn.Conv2d):
                    _mod.weight.data = _mod.weight.data.to(memory_format=torch.channels_last)
            from .model import DiscriminatorP as _DP

            def _cl_forward(_self, x):
                import torch.nn.functional as _F
                fmap = []; b, c, t = x.shape; period = _self.period
                if t % period != 0:  # pad first
                    x = _F.pad(x, (0, period - t % period), "reflect"); t = t + period - t % period
                x = x.view(b, c, t // period, period).to(memory_format=torch.channels_last)
                for l in _self.convs:
                    x = _F.leaky_relu(l(x).to(memory_format=torch.channels_last), 0.1); fmap.append(x)
                x = _self.conv_post(x); fmap.append(x)
                return torch.flatten(x, 1, -1), fmap
            _DP.forward = _cl_forward
        if opts.get("compile_enc_dec", False):
            mode = opts.get("compile_mode", "default")
            for sub in ("enc_p", "dec", "flow"):
                setattr(self.generator, sub, torch.compile(getattr(self.generator, sub), mode=mode, dynamic=False))
        # Loss weights + mel 参数（so-vits config_template.json data 段）
        self.c_mel = opts.get("c_mel", 45.0); self.c_kl = opts.get("c_kl", 1.0)
        self._mel_args = (opts.get("filter_length", 2048), opts.get("n_mel_channels", 80),
                          opts.get("sampling_rate", 44100), opts.get("hop_length", 512),
                          opts.get("win_length", 2048), opts.get("mel_fmin", 0.0), opts.get("mel_fmax", 22050.0))
        self._hop = opts.get("hop_length", 512)

    def forward_generator(self, batch: SVCBatch):
        # net_g 每步仅 forward 一次，输出共享给 D loss（detach）与 G loss（so-vits train.py 原样）
        return self.generator(batch.units, batch.f0, batch.mel, batch.wav, batch.spk_id,
                              frame_len=batch.frame_len, vol=batch.rms, uv=batch.uv)

    def generator_loss(self, batch: SVCBatch, out=None):
        # 必须严格交替：forward_generator → D fwd/bwd/step → 本方法 fwd → bwd/step
        # （AdamW step 原地改 weight_norm 参数，第二次 D fwd 若在其后则 autograd version check 报错）
        if out is None: out = self.forward_generator(batch)
        return gan_g_loss(self.discriminator, out, batch, self._mel_args, self.c_mel, self.c_kl,
                          self._hop, use_lf0=True)

    def discriminator_loss(self, batch: SVCBatch, out=None):
        if out is None:
            with torch.no_grad(): out = self.forward_generator(batch)
        return gan_d_loss(self.discriminator, out, batch, self._hop)

    def sample(self, batch: SVCBatch):
        with torch.no_grad():
            out, _ = self.generator.infer(batch.units.transpose(1, 2), batch.f0, batch.uv,
                                          batch.spk_id.unsqueeze(-1))  # [B,D,T] so-vits 约定
        return out