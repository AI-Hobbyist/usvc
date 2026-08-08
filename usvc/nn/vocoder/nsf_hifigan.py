"""Shared NSF-HiFiGAN vocoder (openvpi pretrained, 44.1k/hop512/128bin).

Extracted verbatim from openvpi/SingingVocoders ``models/nsf_HiFigan/models.py``
(the pretrained vocoder released at openvpi/SingingVocoders; same weights used
by Diffusion-SVC / DDSP-SVC / ReFlow-VAE-SVC / so-vits shallow diffusion).

Two Generator architectures are supported, auto-detected from the ckpt keys:

- ``mini_nsf=True`` (PC-NSF-HiFiGAN, openvpi 2025.02+): uses ``fastsinegen``
  + a single ``source_conv`` injected after the 2nd upsample stage. Keys:
  ``source_conv.*`` (no ``m_source`` / ``noise_convs``).
- ``mini_nsf=False`` (classic NSF-HiFiGAN): uses ``SourceModuleHnNSF``
  (``m_source``) + per-stage ``noise_convs``. Keys: ``m_source.*`` +
  ``noise_convs.*``.

The standalone vocoder differs from the VITS-built-in NSF generator inside
usvc/backends/sovits|rvc (which has a cond layer for speaker conditioning).

Weight formats accepted on load: {'generator': sd} | {'state_dict': sd} |
{'model': sd} | bare state_dict.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class AttrDict(dict):
    """A dictionary with attribute-style access. It maps attribute access to
    the real dictionary.
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        return super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, name):
        if name not in super(AttrDict, self).keys():
            return None
        return super(AttrDict, self).__getitem__(name)

    def __delitem__(self, name):
        return super(AttrDict, self).__delitem__(name)

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def copy(self):
        return AttrDict(self)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(torch.nn.Module):
    """Definition of sine generator

    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-waveform (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        """f0: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device, dtype=f0.dtype)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc, (0, 0, 1, -1))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype).reshape(1, 1, -1))
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device, dtype=f0.dtype)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    @torch.no_grad()
    def forward(self, f0, upp):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf

    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.mini_nsf = h.mini_nsf
        self.noise_sigma = h.noise_sigma

        if h.mini_nsf:
            self.source_sr = h.sampling_rate / int(np.prod(h.upsample_rates[2:]))
            self.upp = int(np.prod(h.upsample_rates[: 2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(np.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(
                sampling_rate=h.sampling_rate,
                harmonic_num=8
            )
            self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        ch = h.upsample_initial_channel
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2)))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
            if not h.mini_nsf:
                if i + 1 < len(h.upsample_rates):
                    stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                    self.noise_convs.append(Conv1d(
                        1, ch, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
                else:
                    self.noise_convs.append(Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = Conv1d(1, ch, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def fastsinegen(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device, dtype=f0.dtype)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        sines = torch.sin(2 * np.pi * rad)
        return sines

    def forward(self, x, f0):
        if self.mini_nsf:
            har_source = self.fastsinegen(f0)
        else:
            har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if self.noise_sigma is not None and self.noise_sigma > 0:
            x += self.noise_sigma * torch.randn_like(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if not self.mini_nsf:
                x_source = self.noise_convs[i](har_source)
                minl = min(x.size(2), x_source.size(2))
                x = x[:, :, :minl] + x_source[:, :, :minl]
            elif i == 1:
                x_source = self.source_conv(har_source)
                minl = min(x.size(2), x_source.size(2))
                x = x[:, :, :minl] + x_source[:, :, :minl]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def load_config(config):
    """Load config from a json path or dict; returns an AttrDict."""
    if isinstance(config, dict):
        h = AttrDict(config)
    else:
        with open(config) as f:
            h = AttrDict(json.loads(f.read()))
    return h


class NSFHiFiGAN(nn.Module):
    """Shared NSF-HiFiGAN vocoder wrapper.

    Args:
        config: openvpi ``config.json`` path or config dict (44.1k/hop512/128bin).
        ckpt_path: pretrained weights. ``mini_nsf`` is auto-detected from ckpt
            keys when the config does not set it explicitly
            (``source_conv.*`` -> mini_nsf=True; ``m_source.*`` -> mini_nsf=False).
        device: 'cuda' | 'cpu'.
        is_half: cast weights to fp16.
    """

    def __init__(self, config, ckpt_path, device="cuda", is_half=False):
        super().__init__()
        self.device = device
        self.h = load_config(config)
        self.sampling_rate = self.h.sampling_rate
        self.hop_size = self.h.hop_size
        self.num_mels = self.h.num_mels

        if self.h.mini_nsf is None:  # not set in config -> infer from ckpt keys
            mini_nsf = self._detect_mini_nsf(ckpt_path)
            if mini_nsf is None:
                raise RuntimeError(
                    "cannot determine vocoder architecture: config has no "
                    "'mini_nsf' and ckpt has neither 'source_conv' nor "
                    "'m_source' keys"
                )
            self.h["mini_nsf"] = mini_nsf

        self.generator = Generator(self.h).to(device)
        self.load_weights(ckpt_path)
        self.generator.eval()
        self.generator.remove_weight_norm()
        if is_half:
            self.generator.half()

    @staticmethod
    def _detect_mini_nsf(ckpt_path):
        """Peek ckpt state_dict keys to determine the Generator variant.

        Returns True (mini_nsf / PC-NSF-HiFiGAN), False (classic NSF) or None
        if the keys are ambiguous.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt
        if isinstance(ckpt, dict):
            for key in ("generator", "state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    sd = ckpt[key]
                    break
        if not isinstance(sd, dict):
            return None
        keys = set(sd.keys())
        # ckpts wrapped as {'state_dict': {'generator.*': ..., 'discriminator.*':}}
        # carry a prefix; keep only the generator sub-keys for detection.
        if any(k.startswith("generator.") for k in keys):
            keys = {k[len("generator."):] for k in keys if k.startswith("generator.")}
        if any(k.startswith("source_conv") for k in keys):
            return True
        if any(k.startswith("m_source.") for k in keys):
            return False
        return None

    def load_weights(self, ckpt_path):
        """Load pretrained weights, tolerant of common wrapper formats.

        Accepts {'generator': sd} | {'state_dict': sd} | {'model': sd} |
        bare state_dict; strips a leading ``generator.`` prefix if present.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt
        if isinstance(ckpt, dict):
            for key in ("generator", "state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    sd = ckpt[key]
                    break
        if isinstance(sd, dict):
            # openvpi release ckpts are {'state_dict': {'generator.*': ...,
            # 'discriminator.*': ...}}: load only the generator sub-keys.
            gen_keys = [k for k in sd if k.startswith("generator.")]
            if gen_keys:
                sd = {k[len("generator."):]: v for k, v in sd.items()
                      if k.startswith("generator.")}
        missing, unexpected = self.generator.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(
                f"NSF-HiFiGAN ckpt mismatch: missing={missing[:8]} "
                f"(is this an openvpi 44.1k/128bin vocoder?)"
            )
        if unexpected:
            raise RuntimeError(f"NSF-HiFiGAN ckpt has unexpected keys: {unexpected[:8]}")

    @torch.no_grad()
    def infer(self, mel, f0):
        """mel [B, T, M] + f0 [B, T] (or [B, T, 1]) -> wav [B, 1, L]."""
        gen_dtype = next(self.generator.parameters()).dtype
        c = mel.transpose(1, 2).to(gen_dtype)  # [B, M, T]
        if f0.dim() == 3:
            f0 = f0.squeeze(-1)
        if f0.shape[-1] != c.shape[-1]:
            f0 = f0[:, : c.shape[-1]]
        if f0.dtype != c.dtype:
            f0 = f0.to(c.dtype)
        audio = self.generator(c, f0)
        return audio

    def forward(self, mel, f0):
        return self.infer(mel, f0)
