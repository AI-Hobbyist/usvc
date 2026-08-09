"""MLX 版 NSF-HiFiGAN(PC-NSF-HiFiGAN + classic)。

镜像 ``usvc/nn/vocoder/nsf_hifigan.py``。两种架构自动检测:
  - mini_nsf=True(PC-NSF): fastsinegen + source_conv(第2级后注入)
  - mini_nsf=False(classic): SourceModuleHnNSF(m_source) + 每级 noise_convs

关键差异(torch → mlx,见映射手册):
  - weight_norm: 推理前 remove_weight_norm 已把权重冻结成普通 weight,
    MLX 直接加载合并后的 weight,运行时零开销
  - ConvTranspose1d: MLX 有,但 padding=(k-u)//2 不对称语义需核
  - LeakyReLU(0.1): mlx.nn.LeakyReLU(0.1)
  - fastsinegen 的 cumsum(dim=1).fmod(1.0): mx.cumsum + mx.floor(rem)
  - 无 snake(子 agent 确认本文件全 LeakyReLU)
  - channels_last 优化路径(forward_cl)直接删除(Metal 不需 NHWC 优化)
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


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

class AttrDict(dict):
    """与 torch 版 AttrDict 一致:属性式访问。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """与 torch 版 get_padding 一致: (kernel*dilation - dilation) // 2。"""
    return int((kernel_size * dilation - dilation) / 2)


# ---------------------------------------------------------------------------
# fastsinegen(PC-NSF,mini_nsf=True)—— 相位累积核心
# ---------------------------------------------------------------------------

def fastsinegen(f0, source_sr: float, upp: int):
    """PC-NSF 的帧域 sine 生成(无随机相位、无 UV 噪声、harmonic=0)。

    复现 ``nsf_hifigan.py::Generator.fastsinegen``:
      rad = s0*n + 0.5*ds0*n*(n-1)/upp      # 线性内插相位公式
      rad2 = fmod(rad[..., -1:] + 0.5, 1.0) - 0.5  # 末帧相位卷绕
      rad_acc = fmod(cumsum(rad2, dim=1), 1.0)     # 帧间相位连续
      rad += pad(rad_acc[:, :-1], left=1)          # 累积相位拼回
      sines = sin(2π · rad)

    Args:
        f0: [B, T] Hz at frame rate。
        source_sr: source 采样率(sampling_rate / prod(upsample_rates[2:]))。
        upp: 源域上采样因子(prod(upsample_rates[:2]))。
    Returns:
        sine_waves [B, 1, L] where L = T * upp。
    """
    B, T = f0.shape
    n = mx.arange(1, upp + 1)                       # [upp]
    s0 = (f0 / source_sr)[:, :, None]               # [B, T, 1]
    # 一阶差分(帧间 f0 变化)
    ds0 = mx.zeros((B, T, 1))
    ds0[:, 1:, 0] = f0[:, 1:] / source_sr - f0[:, :-1] / source_sr
    # 线性内插相位公式
    # rad[b,t,j] = s0[b,t] * n[j] + 0.5 * ds0[b,t] * n[j] * (n[j]-1) / upp
    rad = s0 * n + 0.5 * ds0 * n * (n - 1) / upp    # [B, T, upp]
    # 末帧相位卷绕到 [-0.5, 0.5)
    rad_last = rad[:, :, -1:]                        # [B, T, 1]
    rad2 = mx.floor(rad_last + 0.5) - 0.5            # fmod(x, 1.0) 等价?需核
    # 严格 fmod(rad_last+0.5, 1.0) - 0.5:MLX 用 mod
    rad2 = (rad_last + 0.5) % 1.0 - 0.5
    # 帧间相位累积(cumsum 沿时间轴)
    rad_acc = mx.cumsum(rad2[:, :, 0], axis=1)       # [B, T]
    rad_acc = (rad_acc) % 1.0                   # fmod(.,1.0)
    # 拼回:rad_acc[:, :-1] 前补 0(shift 一帧)
    rad_acc_shifted = mx.zeros((B, T))
    rad_acc_shifted[:, 1:] = rad_acc[:, :-1]
    rad = rad + rad_acc_shifted[:, :, None]          # [B, T, upp]
    rad = rad.reshape(B, T * upp, 1)                 # [B, L, 1] (NLC)
    return mx.sin(2 * math.pi * rad)


# ---------------------------------------------------------------------------
# classic SineGen + SourceModuleHnNSF(mini_nsf=False)
# ---------------------------------------------------------------------------

class SineGen(nn.Module):
    """classic NSF SineGen(随机相位 + UV 噪声,harmonic_num=8)。

    复现 ``nsf_hifigan.py::SineGen``。注意与 RVC 的 SineGen 不同(RVC 用
    linear 插值相位,classic 用 nearest)。
    """

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
        # uv = f0 > threshold(与 torch ones_like * (f0 > thresh) 等价)
        return (f0 > self.voiced_threshold).astype(f0.dtype)

    def __call__(self, f0, upp: int):
        """f0: [B, T_frames] -> (sine_waves [B, L, dim], uv [B,L,1], noise [B,L,dim])。"""
        f0 = f0[:, None, :].transpose(0, 2, 1)       # [B, T, 1]
        B, T, _ = f0.shape
        f0_buf = mx.zeros((B, T, self.dim))
        f0_buf[:, :, 0] = f0[:, :, 0]
        for idx in range(self.harmonic_num):
            f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
        rad_values = (f0_buf / self.sampling_rate) % 1.0
        # rand_ini:首谐波为 0,其余随机
        rand_ini = mx.random.uniform(shape=(B, self.dim))
        rand_ini[:, 0] = 0.0
        rad_values = rad_values + rand_ini[:, None, :]
        # cumsum 相位
        tmp_over_one = mx.cumsum(rad_values, axis=1)  # [B, T, dim]
        tmp_over_one = tmp_over_one * upp
        # linear interpolate phase to sample rate
        # transpose -> [B, dim, T] interpolate -> [B, dim, L]
        tmp_t = tmp_over_one.transpose(0, 2, 1)
        tmp_t = interpolate_linear(tmp_t, size=T * upp, align_corners=True)
        tmp_over_one = tmp_t.transpose(0, 2, 1)
        rad_values = interpolate_nearest(
            rad_values.transpose(0, 2, 1), scale_factor=float(upp)
        ).transpose(0, 2, 1)
        tmp_over_one = (tmp_over_one) % 1.0
        # 检测相位卷绕(<0 处需 shift)
        tmp_diff = tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]
        tmp_over_one_idx = tmp_diff < 0
        cumsum_shift = mx.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        sine_waves = mx.sin(
            mx.cumsum(rad_values + cumsum_shift, axis=1) * 2 * math.pi
        )
        sine_waves = sine_waves * self.sine_amp
        uv = self._f02uv(f0)
        uv = interpolate_nearest(uv.transpose(0, 2, 1), scale_factor=float(upp)).transpose(0, 2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """classic source module: Linear(harmonic+1→1) + Tanh。"""

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


# ---------------------------------------------------------------------------
# ResBlock(MLX 权重已是合并后的普通 conv,无 weight_norm 运行时逻辑)
# ---------------------------------------------------------------------------

class ResBlock1(nn.Module):
    """3 组 dilated conv 残差栈(对应 torch ResBlock1)。"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        pad = [get_padding(kernel_size, d) for d in dilation]
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1,
                      dilation=dilation[i], padding=pad[i])
            for i in range(3)
        ]
        pad2 = get_padding(kernel_size, 1)
        self.convs2 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1,
                      dilation=1, padding=pad2)
            for _ in range(3)
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = nn.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    """2 组 dilated conv 残差栈(对应 torch ResBlock2)。"""

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


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """NSF-HiFiGAN Generator。

    mini_nsf=True: fastsinegen + source_conv(第2级后注入)
    mini_nsf=False: SourceModuleHnNSF + 每级 noise_convs

    注:推理前 torch 版 remove_weight_norm() 已把权重冻结,MLX 直接加载
    合并后的普通 weight。conv_pre/conv_post/ups/resblocks 全是普通 Conv1d。
    """

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
                h["sampling_rate"], harmonic_num=8, sine_amp=0.1
            )
            self.noise_convs = []  # mlx 用普通 list 替代 nn.ModuleList

        self.conv_pre = nn.Conv1d(h["inter_channels"], upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2
        self.ups = []
        self.resblocks = []
        for i, (u, k) in enumerate(zip(upsample_rates, h["upsample_kernel_sizes"])):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            # ConvTranspose1d:MLX padding=(k-u)//2,与 torch 一致
            self.ups.append(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                c_cur, k, stride=u, padding=(k - u) // 2,
            ))
            if not mini_nsf and i + 1 < self.num_upsamples:
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2,
                    stride=stride_f0, padding=(stride_f0 + 1) // 2,
                ))
            elif not mini_nsf:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
            for j, (rk, rd) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(c_cur, rk, rd))

        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** self.num_upsamples), 1, 7, 1, padding=3
        )
        # cond 层仅在有 speaker 条件时创建(独立 vocoder 无 gin_channels)
        gin_channels = h.get("gin_channels", 0)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels else None

    def __call__(self, x, f0=None, g=None):
        """x: [B, T, inter_channels](mel 或 latent);f0: [B, T_frames];g: [B, T, gin]。

        Returns wav [B, 1, L]。
        """
        # 谐波源
        if self.mini_nsf:
            har_source = fastsinegen(f0, self.source_sr, self.upp)  # [B, 1, L_har]
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
                    # 对齐长度
                    minl = min(x.shape[1], hs.shape[1])
                    x = x[:, :minl, :] + hs[:, :minl, :]
            else:
                x_source = self.noise_convs[i](har_source)
                minl = min(x.shape[1], x_source.shape[1])
                x = x[:, :minl, :] + x_source[:, :minl, :]
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                rb = self.resblocks[idx](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels

        x = nn.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return mx.tanh(x)                           # [B, T, 1]


# ---------------------------------------------------------------------------
# NSFHiFiGAN wrapper(对应 torch NSFHiFiGAN 类)
# ---------------------------------------------------------------------------

class NSFHiFiGAN:
    """MLX 版 NSF-HiFiGAN vocoder(非 nn.Module,持有 Generator + 配置)。

    用法::
        vg = NSFHiFiGAN(config_json_path, ckpt_path, device='gpu')
        wav = vg.infer(mel_mx, f0_mx)   # mel [B, T, n_mels], f0 [B, T_frames]
    """

    def __init__(
        self,
        config_path: str | Path,
        ckpt_path: str | Path,
        device: str = "gpu",
        is_half: bool = False,
        mini_nsf: bool | None = None,
    ):
        self.device = mx.gpu if device == "gpu" else mx.cpu
        if self.device is not None:
            mx.set_default_device(self.device)

        with open(config_path) as f:
            h = AttrDict(json.load(f))
        self.h = h

        # 自动检测 mini_nsf(若未指定)
        if mini_nsf is None:
            mini_nsf = self._detect_mini_nsf(ckpt_path)
        self.mini_nsf = mini_nsf

        self.generator = Generator(h, mini_nsf=mini_nsf)
        # 加载权重(vocoder ckpt 的 ups.* 是 weight_norm,merge_weight_norm
        # 合并后成 ups.*.weight;这些是 ConvTranspose1d,需 (I,O,K)->(O,K,I) 转置)
        from ._convert import load_torch_ckpt_to_mlx
        dtype = mx.float16 if is_half else mx.float32
        ctranspose = {f"generator.ups.{i}.weight" for i in range(len(h["upsample_rates"]))}
        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype, convtranspose_keys=ctranspose)
        # 剥 generator. 前缀(NSFHiFiGAN ckpt 外层)
        sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
        self.generator.load_weights(list(sd.items()), strict=False)
        self.generator.eval()

    @staticmethod
    def _detect_mini_nsf(ckpt_path) -> bool:
        """窥探 ckpt 键:含 source_conv → True;含 m_source. → False。"""
        from ._convert import load_torch_sd
        sd = load_torch_sd(ckpt_path)
        keys = set(sd.keys())
        if any("source_conv" in k for k in keys):
            return True
        if any(k.startswith("m_source.") or "m_source." in k for k in keys):
            return False
        return True  # 默认 mini_nsf

    def infer(self, mel: mx.array, f0: mx.array) -> mx.array:
        """mel: [B, T_frames, n_mels] → wav [B, 1, L]。

        与 torch 版 infer 一致:mel 转 NLC,f0 对齐到帧数。
        """
        # mlx Generator 是 NLC,接受 mel [B, T, n_mels] 直接传入(与 torch infer
        # 的 [B, T, M] 约定一致,torch 内部转 [B, M, T] 是因 torch generator 是 NCL)
        return self.generator(mel, f0=f0)
