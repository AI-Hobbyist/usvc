"""MLX 版 NSF-HiFiGAN(PC-NSF + classic 自动检测;镜像 nn/vocoder/nsf_hifigan.py)。

mini_nsf=True(PC-NSF): fastsinegen + source_conv(第2级后注入);
mini_nsf=False(classic): SourceModuleHnNSF + 每级 noise_convs。
torch 差异(映射手册):weight_norm 推理前已冻结,加载即普通 weight;
ConvTranspose1d padding=(k-u)//2;fastsinegen 相位 cumsum 用 %1.0 卷绕;
无 snake(全 LeakyReLU);forward_cl(channels_last)删除。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ._ops import interpolate_linear, interpolate_nearest

__all__ = ["NSFHiFiGAN", "AttrDict"]

LRELU_SLOPE = 0.1


class AttrDict(dict):
    """属性式访问(与 torch 版一致)。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def fastsinegen(f0, source_sr: float, upp: int):
    """PC-NSF 帧域 sine(无随机相位、无 UV 噪声、harmonic=0)。

    相位公式:rad = s0*n + 0.5*ds0*n*(n-1)/upp;s0=f0/source_sr,
    ds0 为帧间一阶差分;末帧卷绕 rad2=fmod(rad+0.5,1)-0.5 再 cumsum(axis=1)
    保证帧间相位连续;rad += shift(rad_acc)。返回 [B, T*upp, 1]。
    """
    B, T = f0.shape
    n = mx.arange(1, upp + 1)  # [upp]
    s0 = (f0 / source_sr)[:, :, None]  # [B, T, 1]
    ds0 = mx.zeros((B, T, 1))
    ds0[:, 1:, 0] = f0[:, 1:] / source_sr - f0[:, :-1] / source_sr
    rad = s0 * n + 0.5 * ds0 * n * (n - 1) / upp  # [B, T, upp]
    rad_last = rad[:, :, -1:]
    rad2 = (rad_last + 0.5) % 1.0 - 0.5  # fmod 卷绕
    rad_acc = mx.cumsum(rad2[:, :, 0], axis=1) % 1.0  # 帧间相位连续
    rad_acc_shifted = mx.zeros((B, T))
    rad_acc_shifted[:, 1:] = rad_acc[:, :-1]
    rad = (rad + rad_acc_shifted[:, :, None]).reshape(B, T * upp, 1)
    return mx.sin(2 * math.pi * rad)


class SineGen(nn.Module):
    """classic NSF SineGen(随机相位 + UV 噪声,harmonic=8)。⚠ 与 RVC 的
    SineGen 不同(RVC 用 linear 插值相位,classic 用 nearest)。"""

    def __init__(self, samp_rate: int, harmonic_num: int = 8,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).astype(f0.dtype)

    def __call__(self, f0, upp: int):
        """f0 [B, T] → (sine [B, L, dim], uv [B, L, 1], noise [B, L, dim])。"""
        f0 = f0[:, None, :].transpose(0, 2, 1)  # [B, T, 1]
        B, T, _ = f0.shape
        f0_buf = mx.zeros((B, T, self.dim))
        f0_buf[:, :, 0] = f0[:, :, 0]
        for idx in range(self.harmonic_num):
            f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
        rad_values = (f0_buf / self.sampling_rate) % 1.0
        rand_ini = mx.random.uniform(shape=(B, self.dim))
        rand_ini[:, 0] = 0.0  # 首谐波相位 0,其余随机
        rad_values = rad_values + rand_ini[:, None, :]
        tmp_over_one = mx.cumsum(rad_values, axis=1) * upp
        tmp_t = tmp_over_one.transpose(0, 2, 1)
        tmp_t = interpolate_linear(tmp_t, size=T * upp, align_corners=True)
        tmp_over_one = tmp_t.transpose(0, 2, 1)
        rad_values = interpolate_nearest(
            rad_values.transpose(0, 2, 1), scale_factor=float(upp)
        ).transpose(0, 2, 1)
        tmp_over_one = tmp_over_one % 1.0
        tmp_diff = tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]
        cumsum_shift = mx.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = (tmp_diff < 0) * -1.0  # 卷绕点相位 shift
        sine_waves = mx.sin(mx.cumsum(rad_values + cumsum_shift, axis=1)
                            * 2 * math.pi) * self.sine_amp
        uv = interpolate_nearest(self._f02uv(f0).transpose(0, 2, 1),
                                 scale_factor=float(upp)).transpose(0, 2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int, harmonic_num: int = 8,
                 sine_amp: float = 0.1, add_noise_std: float = 0.003,
                 voiced_threshod: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
                                 add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x, upp=None):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = mx.random.normal(shape=uv.shape) * self.sine_amp / 3
        return sine_merge, noise, uv


# --- ResBlock(权重已是合并后的普通 conv,无 weight_norm 运行时逻辑) ----------

class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1,
                      dilation=dilation[i], padding=get_padding(kernel_size, dilation[i]))
            for i in range(3)
        ]
        pad2 = get_padding(kernel_size, 1)
        self.convs2 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=pad2)
            for _ in range(3)
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, LRELU_SLOPE)
            xt = c2(nn.leaky_relu(c1(xt), LRELU_SLOPE))
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3)):
        super().__init__()
        self.convs = [
            nn.Conv1d(channels, channels, kernel_size, stride=1,
                      dilation=dilation[i], padding=get_padding(kernel_size, dilation[i]))
            for i in range(2)
        ]

    def __call__(self, x):
        for c in self.convs:
            xt = nn.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """NSF-HiFiGAN Generator;属性名 = ckpt 键面(mini_nsf/upp/source_sr/
    source_conv/m_source/conv_pre/cond/ups/resblocks/conv_post 不可改名)。"""

    def __init__(self, h: AttrDict, mini_nsf: bool = True):
        super().__init__()
        self.h = h
        self.mini_nsf = mini_nsf
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        upsample_rates = h["upsample_rates"]
        upsample_initial_channel = h["upsample_initial_channel"]

        if mini_nsf:
            source_sr = h["sampling_rate"] // math.prod(upsample_rates[2:])
            self.upp = int(math.prod(upsample_rates[:2]))
            self.source_sr = float(source_sr)
            self.source_conv = nn.Conv1d(1, h["upsample_initial_channel"] // 4, 1)
        else:
            self.upp = math.prod(upsample_rates)
            self.m_source = SourceModuleHnNSF(
                h["sampling_rate"], harmonic_num=8, sine_amp=0.1)
            self.noise_convs = []  # 普通 list(nn.ModuleList 语义相同,mlx 注册)

        self.conv_pre = nn.Conv1d(h["inter_channels"], upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2
        self.ups = []
        self.resblocks = []
        for i, (u, k) in enumerate(zip(upsample_rates, h["upsample_kernel_sizes"])):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i), c_cur, k,
                stride=u, padding=(k - u) // 2))  # 与 torch 一致
            if not mini_nsf:
                if i + 1 < self.num_upsamples:
                    stride_f0 = math.prod(upsample_rates[i + 1:])
                    self.noise_convs.append(nn.Conv1d(
                        1, c_cur, kernel_size=stride_f0 * 2,
                        stride=stride_f0, padding=(stride_f0 + 1) // 2))
                else:
                    self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
            for j, (rk, rd) in enumerate(zip(h["resblock_kernel_sizes"],
                                             h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(c_cur, rk, rd))

        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** self.num_upsamples), 1, 7, 1, padding=3)
        gin_channels = h.get("gin_channels", 0)  # 独立 vocoder 无 cond 层
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels else None

    def __call__(self, x, f0=None, g=None):
        """x [B, T, inter](mel/latent);f0 [B, T_frames];g [B, T, gin]→ wav [B, 1, L]。"""
        if self.mini_nsf:
            har_source = fastsinegen(f0, self.source_sr, self.upp)  # [B, L, 1]
        else:
            har_source, _, _ = self.m_source(f0, self.upp)

        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if self.mini_nsf:
                if i == 1:  # 第2级后注入 source_conv
                    hs = self.source_conv(har_source)
                    minl = min(x.shape[1], hs.shape[1])
                    x = x[:, :minl, :] + hs[:, :minl, :]
            else:
                x_source = self.noise_convs[i](har_source)
                minl = min(x.shape[1], x_source.shape[1])
                x = x[:, :minl, :] + x_source[:, :minl, :]
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels

        x = nn.leaky_relu(x, LRELU_SLOPE)
        return mx.tanh(self.conv_post(x))  # [B, T, 1]


class NSFHiFiGAN:
    """vocoder wrapper(非 nn.Module)。用法:vg=NSFHiFiGAN(config,ckpt);
    wav=vg.infer(mel, f0)。"""

    def __init__(self, config_path: str | Path, ckpt_path: str | Path,
                 device: str = "gpu", is_half: bool = False,
                 mini_nsf: bool | None = None):
        self.device = mx.gpu if device == "gpu" else mx.cpu
        if self.device is not None:
            mx.set_default_device(self.device)

        with open(config_path) as f:
            self.h = h = AttrDict(json.load(f))

        if mini_nsf is None:
            mini_nsf = self._detect_mini_nsf(ckpt_path)
        self.mini_nsf = mini_nsf

        self.generator = Generator(h, mini_nsf=mini_nsf)
        from ._convert import load_torch_ckpt_to_mlx
        dtype = mx.float16 if is_half else mx.float32
        ctranspose = {f"generator.ups.{i}.weight" for i in range(len(h["upsample_rates"]))}
        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype, convtranspose_keys=ctranspose)
        # ups 是 ConvTranspose1d ((I,O,K)->(O,K,I));剥 generator. 前缀
        sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
        self.generator.load_weights(list(sd.items()), strict=False)
        self.generator.eval()

    @staticmethod
    def _detect_mini_nsf(ckpt_path) -> bool:
        """窥探 ckpt 键:含 source_conv → True;含 m_source. → False;默认 True。"""
        from ._convert import load_torch_sd
        keys = set(load_torch_sd(ckpt_path).keys())
        if any("source_conv" in k for k in keys):
            return True
        if any(k.startswith("m_source.") or "m_source." in k for k in keys):
            return False
        return True

    def infer(self, mel: mx.array, f0: mx.array) -> mx.array:
        """mel [B, T, n_mels](NLC,mlx generator 直接吃)→ wav [B, 1, L]。"""
        return self.generator(mel, f0=f0)