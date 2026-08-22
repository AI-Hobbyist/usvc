"""Shared NSF-HiFiGAN vocoder (openvpi pretrained, 44.1k/hop512/128bin).

Extracted from openvpi/SingingVocoders ``models/nsf_HiFigan/models.py`` (same
weights used by Diffusion-SVC / DDSP-SVC / ReFlow-VAE-SVC). Two Generator
architectures, auto-detected from ckpt keys: mini_nsf=True (PC, 2025.02+):
fastsinegen + single ``source_conv`` after the 2nd upsample; mini_nsf=False
(classic): SourceModuleHnNSF (m_source) + per-stage noise_convs. Weight
formats: {'generator'|'state_dict'|'model': sd} | bare.
"""
import json
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

# channels_last 推理辅助统一在 usvc/nn/cl_helper.py（rvc/sovits/vocoder/wavenet 共享）
from ..cl_helper import conv2d_cl, convT2d_cl
from .._commons import get_padding, init_weights

LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *a, **k): dict.__init__(self, *a, **k)
    def __getstate__(self): return self.__dict__.items()
    def __setstate__(self, items):
        for k, v in items: self.__dict__[k] = v
    def __repr__(self): return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))
    def __getitem__(self, name):
        if name not in super().keys(): return None
        return super().__getitem__(name)
    __getattr__ = __getitem__; __setattr__ = dict.__setitem__
    def copy(self): return AttrDict(self)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                                       padding=get_padding(kernel_size, d))) for d in dilation); self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                                       padding=get_padding(kernel_size, 1))) for _ in range(3)); self.convs2.apply(init_weights)
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)) + x
        return x
    def forward_cl(self, x):  # channels_last 版（x: [B,C,1,T] CL，等价于 forward）
        for c1, c2 in zip(self.convs1, self.convs2):
            x = conv2d_cl(F.leaky_relu(conv2d_cl(F.leaky_relu(x, LRELU_SLOPE), c1), LRELU_SLOPE), c2) + x
        return x
    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                                      padding=get_padding(kernel_size, d))) for d in dilation)
        self.convs.apply(init_weights)
    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x
    def forward_cl(self, x):  # channels_last 版（x: [B,C,1,T] CL，等价于 forward）
        for c in self.convs:
            x = conv2d_cl(F.leaky_relu(x, LRELU_SLOPE), c) + x
        return x
    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(torch.nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp, self.noise_std, self.harmonic_num = sine_amp, noise_std, harmonic_num
        self.dim, self.sampling_rate, self.voiced_threshold = harmonic_num + 1, samp_rate, voiced_threshold
    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device, dtype=f0.dtype)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc, (0, 0, 1, -1))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype).reshape(1, 1, -1))
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device, dtype=f0.dtype) * (torch.arange(1, self.dim + 1, device=f0.device) > 1)
        sine_waves = torch.sin(2 * math.pi * (rad + rand_ini)) * self.sine_amp
        uv = F.interpolate((f0 > self.voiced_threshold).to(f0.dtype).transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        return sine_waves * uv + noise_amp * torch.randn_like(sine_waves)


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
    def forward(self, x, upp):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x, upp)))


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes); self.num_upsamples = len(h.upsample_rates)
        self.mini_nsf = h.mini_nsf; self.noise_sigma = h.noise_sigma
        if h.mini_nsf:
            self.source_sr = h.sampling_rate / int(math.prod(h.upsample_rates[2:]))
            self.upp = int(math.prod(h.upsample_rates[:2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(math.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
            self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        self._use_cl = False  # engine 推理时置 True 走 forward_cl
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = nn.ModuleList(); self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2)))
            self.resblocks += [resblock(h, ch, k2, d) for k2, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)]
            if h.mini_nsf:
                if i == 1:
                    self.source_conv = Conv1d(1, ch, 1)
                    self.source_conv.apply(init_weights)
            else:
                s = int(math.prod(h.upsample_rates[i + 1:])) if i + 1 < len(h.upsample_rates) else 1
                self.noise_convs.append(Conv1d(1, ch, kernel_size=s * 2 if s > 1 else 1, stride=s, padding=s // 2))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights); self.conv_post.apply(init_weights)
    def fastsinegen(self, f0):  # mini_nsf phase accumulation (no RNG)
        n = torch.arange(1, self.upp + 1, device=f0.device, dtype=f0.dtype)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        return torch.sin(2 * math.pi * rad.reshape(f0.shape[0], 1, -1))
    def forward(self, x, f0):
        return self.forward_cl(x, f0) if self._use_cl else self._forward(x, f0, cl=False)
    def forward_cl(self, x, f0):  # channels_last 版（x: [B,M,1,T] CL，等价于 forward）
        return self._forward(x, f0, cl=True)[:, :, 0, :]  # [B,1,T]（view，无拷贝）
    def _forward(self, x, f0, cl=False):
        har_source = self.fastsinegen(f0) if self.mini_nsf else self.m_source(f0, self.upp).transpose(1, 2)
        if cl:
            hs4 = har_source[:, :, None, :].to(memory_format=torch.channels_last)
            x = conv2d_cl(x[:, :, None, :].to(memory_format=torch.channels_last), self.conv_pre)
        else:
            x = self.conv_pre(x)
        if self.noise_sigma is not None and self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = convT2d_cl(x, self.ups[i]) if cl else self.ups[i](x)
            src = None
            if not self.mini_nsf:
                src = conv2d_cl(hs4, self.noise_convs[i]) if cl else self.noise_convs[i](har_source)
            elif i == 1:
                src = conv2d_cl(hs4, self.source_conv) if cl else self.source_conv(har_source)
            if src is not None:
                minl = min(x.size(-1), src.size(-1))
                x = x[..., :minl] + src[..., :minl]
            x = sum(rb.forward_cl(x) if cl else rb(x) for rb in self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]) / self.num_kernels
        x = F.leaky_relu(x)
        x = conv2d_cl(x, self.conv_post) if cl else self.conv_post(x)
        return torch.tanh(x)
    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre); remove_weight_norm(self.conv_post)


def load_config(config):
    if not isinstance(config, dict):
        with open(config) as f:
            config = json.load(f)
    return AttrDict(config)


class NSFHiFiGAN(nn.Module):
    """Shared NSF-HiFiGAN vocoder wrapper (config path/dict + ckpt path)."""
    def __init__(self, config, ckpt_path, device="cuda", is_half=False):
        super().__init__()
        self.device = device
        self.h = load_config(config)
        self.sampling_rate, self.hop_size, self.num_mels = self.h.sampling_rate, self.h.hop_size, self.h.num_mels
        if self.h.mini_nsf is None:  # not set in config -> infer from ckpt keys
            mini_nsf = self._detect_mini_nsf(ckpt_path)
            if mini_nsf is None:
                raise RuntimeError("cannot determine vocoder architecture: config has no 'mini_nsf' and ckpt has neither 'source_conv' nor 'm_source' keys")
            self.h["mini_nsf"] = mini_nsf
        self.generator = Generator(self.h).to(device)
        self.load_weights(ckpt_path)
        self.generator.eval()
        self.generator.remove_weight_norm()
        if is_half: self.generator.half()
    @staticmethod
    def _sd_of(ckpt):
        if isinstance(ckpt, dict):
            for key in ("generator", "state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
        return ckpt
    @staticmethod
    def _detect_mini_nsf(ckpt_path):
        """Peek ckpt keys: True (mini_nsf), False (classic), None (ambiguous)."""
        sd = NSFHiFiGAN._sd_of(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        if not isinstance(sd, dict):
            return None
        keys = {k[len("generator."):] for k in sd if k.startswith("generator.")} or set(sd)
        if any(k.startswith("source_conv") for k in keys):
            return True
        if any(k.startswith("m_source.") for k in keys):
            return False
        return None
    def load_weights(self, ckpt_path):
        sd = self._sd_of(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        if isinstance(sd, dict) and any(k.startswith("generator.") for k in sd):
            sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
        missing, unexpected = self.generator.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"NSF-HiFiGAN ckpt mismatch: missing={missing[:8]} (is this an openvpi 44.1k/128bin vocoder?)")
        if unexpected:
            raise RuntimeError(f"NSF-HiFiGAN ckpt has unexpected keys: {unexpected[:8]}")
    @torch.no_grad()
    def infer(self, mel, f0):  # mel [B,T,M] + f0 [B,T] (or [B,T,1]) -> wav [B,1,L]
        gen_dtype = next(self.generator.parameters()).dtype
        c = mel.transpose(1, 2).to(gen_dtype)  # [B, M, T]
        if f0.dim() == 3: f0 = f0.squeeze(-1)
        if f0.shape[-1] != c.shape[-1]: f0 = f0[:, : c.shape[-1]]
        if f0.dtype != c.dtype: f0 = f0.to(c.dtype)
        return self.generator(c, f0)
    def forward(self, mel, f0):
        return self.infer(mel, f0)