"""MLX 版 RVC GeneratorNSF(wav-domain 生成器)。

镜像 ``usvc/backends/rvc/model.py`` 的 ``GeneratorNSF``(推理最小集)。
RVC 的 GeneratorNSF 是 NSF-HiFiGAN 变体:
  - conv_pre/conv_post **无 weight_norm**(与 vocoder 不同)—— 加载即普通 weight
  - conv_post bias=False
  - ups 是 weight_norm ConvTranspose1d(加载时由 _convert 合并)
  - 每级上采样后加 noise_convs[i](har_source)
  - RVC SineGen 与 vocoder classic 不同:RVC 的 rand_ini 只加到第一帧
    ``rad_values[:, 0, :]``,且 m_source 返回 (sine_merge, None, None)
  - f0_to_coarse:mel 刻度量化到 [1,255]

与 torch 版的差异(见 MLX 映射手册):
  - 删除 channels_last 优化路径(forward_cl),只用 forward
  - MLX 输入 x: [B, T, C](NLC),torch 是 [B, C, T]
  - g(speaker):mlx 接受 [B, gin](自动转 [B,1,gin])或 [B, 1, gin]
  - weight_norm 权重由 ``_convert.load_torch_ckpt_to_mlx`` 在加载时合并

TextEncoder/flow(RVCModel 全量)暂不在本文件 —— 见 backends/__init__ 的
加载分发。本文件只提供 generator 部分(推理最常走的 dec)。
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .._ops import interpolate_linear, interpolate_nearest
from ..nsf_hifigan import ResBlock1, ResBlock2  # 复用 vocoder 的 ResBlock

__all__ = [
    "RVCSineGen",
    "RVCSourceModuleHnNSF",
    "GeneratorNSF",
    "f0_to_coarse",
]

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# RVC SineGen —— 与 vocoder classic 不同:
#   - rand_ini 只加到第一帧 rad_values[:, 0, :](vocoder 加到所有帧)
#   - _f02uv 由 f0 直接计算(无 DirectML 特判,mlx 不需要)
#   - forward 全程 @torch.no_grad(mlx 天然无梯度,无需装饰器)
# ---------------------------------------------------------------------------

class RVCSineGen(nn.Module):
    """RVC sine generator(镜像 rvc/model.py::SineGen,harmonic_num=0)。"""

    def __init__(self, samp_rate: int, harmonic_num: int = 0,
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
        # f0[:, None, :].transpose(1,2) -> [B, T, 1]
        f0 = f0[:, None, :].transpose(0, 2, 1)
        B, T, _ = f0.shape
        f0_buf = mx.zeros((B, T, self.dim))
        f0_buf[:, :, 0] = f0[:, :, 0]
        for idx in range(self.harmonic_num):
            f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
        rad_values = (f0_buf / self.sampling_rate) % 1.0
        # rand_ini 只加到第一帧(torch: rad_values[:, 0, :] += rand_ini)
        rand_ini = mx.random.uniform(shape=(B, self.dim))
        rand_ini[:, 0] = 0.0
        rad_values = rad_values.at[:, 0, :].add(rand_ini)
        tmp_over_one = mx.cumsum(rad_values, axis=1)  # [B, T, dim]
        tmp_over_one = tmp_over_one * upp
        # linear interpolate phase to sample rate (align_corners=True)
        tmp_t = tmp_over_one.transpose(0, 2, 1)       # [B, dim, T]
        tmp_t = interpolate_linear(tmp_t, size=T * upp, align_corners=True)
        tmp_over_one = tmp_t.transpose(0, 2, 1)       # [B, L, dim]
        rad_values = interpolate_nearest(
            rad_values.transpose(0, 2, 1), scale_factor=float(upp)
        ).transpose(0, 2, 1)
        tmp_over_one = tmp_over_one % 1.0
        # 相位卷绕检测(<0 处需 shift)
        tmp_diff = tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]
        tmp_over_one_idx = tmp_diff < 0
        cumsum_shift = mx.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        sine_waves = mx.sin(
            mx.cumsum(rad_values + cumsum_shift, axis=1) * 2 * math.pi
        )
        sine_waves = sine_waves * self.sine_amp
        uv = self._f02uv(f0)
        uv = interpolate_nearest(
            uv.transpose(0, 2, 1), scale_factor=float(upp)
        ).transpose(0, 2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class RVCSourceModuleHnNSF(nn.Module):
    """RVC source module(镜像 rvc/model.py::SourceModuleHnNSF)。

    返回 (sine_merge, None, None)(noise/uv 为 None,torch 版如此)。
    """

    def __init__(self, sampling_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, add_noise_std: float = 0.003,
                 voiced_threshod: float = 0, is_half: bool = True):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = RVCSineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x, upp=1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


# ---------------------------------------------------------------------------
# GeneratorNSF —— RVC wav-domain 生成器
# ---------------------------------------------------------------------------

class GeneratorNSF(nn.Module):
    """RVC GeneratorNSF(镜像 rvc/model.py::GeneratorNSF,推理 forward)。

    删除 channels_last 优化路径;MLX Conv1d 输入 NLC [B, T, C]。

    Args:
        g 接受 [B, gin](2D)自动转 [B,1,gin],或 [B, 1, gin](NLC)。
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        is_half: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.gin_channels = gin_channels
        self.sr = sr

        self.m_source = RVCSourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = []
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    c_cur, k, stride=u, padding=(k - u) // 2,
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    nn.Conv1d(
                        1, c_cur, kernel_size=stride_f0 * 2,
                        stride=stride_f0, padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    # -- ckpt 加载 ---------------------------------------------------------

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str | Path,
        *,
        config: dict | None = None,
        key_prefix: str | None = "generator.dec",
        strict: bool = False,
    ):
        """从 RVC torch ckpt 加载生成器权重(推理最小集)。

        RVC ckpt 中生成器权重位于 ``generator.dec.*``(usvc 训练格式)或
        ``dec.*``(裸 RVC)。``config`` 提供构造参数;None 时从 ckpt 反推
        (见 ``_infer_config``)。

        Args:
            ckpt_path: torch ckpt(lightning/RVC wrapper)。
            config: GeneratorNSF 构造参数 dict;None 自动推断。
            key_prefix: ckpt 中 dec 权重的键前缀(不含尾部点)。
            strict: 是否严格匹配权重键。
        """
        from .._convert import load_torch_ckpt_to_mlx

        sd = load_torch_ckpt_to_mlx(ckpt_path)
        cfg = config if config is not None else cls._infer_config(ckpt_path, key_prefix, sd=sd)
        obj = cls(**cfg)
        # 剥 dec 前缀
        prefix = (key_prefix + ".") if key_prefix else ""
        dec_sd = {}
        for k, v in sd.items():
            if prefix and k.startswith(prefix):
                dec_sd[k[len(prefix):]] = v
            elif not prefix and (k.startswith("dec.") or ".dec." in k):
                # 容忍 ".dec." 中间前缀(如 generator.dec.*)
                dec_sd[k.split(".dec.")[-1]] = v
        if not dec_sd:
            raise ValueError(
                f"no '{prefix or 'dec'}' keys found in {ckpt_path}; "
                f"got keys like: {list(sd)[:5]}"
            )
        obj.load_weights(list(dec_sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(ckpt_path: str | Path | None = None, key_prefix: str | None = None,
                      sd: dict | None = None) -> dict:
        """从 ckpt 权重 shape 反推 GeneratorNSF 构造参数。

        仅推断初等参数:initial_channel(conv_pre 输入)、upsample_initial_channel、
        gin_channels(cond)、upsample kernel sizes。upsample_rates 无法从权重
        shape 推断(在 stride 里),用 RVC 默认 [10,8,2,2]。

        Args:
            ckpt_path: torch ckpt;与 sd 二选一。
            key_prefix: ckpt 中 dec 权重的键前缀。
            sd: 预加载的 mlx state_dict(测试/复用用);None 则从 ckpt 加载。
        """
        if sd is None:
            if ckpt_path is None:
                raise ValueError("either ckpt_path or sd must be given")
            from .._convert import load_torch_ckpt_to_mlx
            sd = load_torch_ckpt_to_mlx(ckpt_path)
        prefix = (key_prefix + ".") if key_prefix else ""
        cand = {}
        for k, v in sd.items():
            if prefix and k.startswith(prefix):
                cand[k[len(prefix):]] = v
            elif not prefix and ".dec." in k:
                cand[k.split(".dec.")[-1]] = v
        shape = lambda k: cand[k].shape if k in cand else None  # noqa: E731
        conv_pre_w = shape("conv_pre.weight")
        conv_post_w = shape("conv_post.weight")
        cond_w = shape("cond.weight")
        if conv_pre_w is None or conv_post_w is None:
            raise ValueError(
                f"cannot infer GeneratorNSF config from {ckpt_path}: "
                f"missing conv_pre/conv_post"
            )
        initial_channel = conv_pre_w[2]   # mlx (O, K, C)
        upsample_initial_channel = conv_pre_w[0]
        # upsample 层数 = ups.*.weight 数量(每层一个)
        ups = sorted([k for k in cand if k.startswith("ups.") and k.endswith(".weight")],
                     key=lambda k: int(k.split(".")[1]))
        num_ups = len(ups)
        # 从每层 in/out 通道推断 upsample_initial_channel 的 /2^i 结构
        # mlx ConvTranspose1d weight: (out, k, in);torch 是 (in, out, k)
        # 已由 _convert 转置,out = 2^(i+1) 通道,in = 2^i 通道
        kernel_sizes = []
        for uk in ups:
            kernel_sizes.append(int(cand[uk].shape[1]))  # mlx (out, k, in) 的 k
        # upsample_rates 无法从权重 shape 推断(在 stride 里);
        # 用 RVC 默认 [10,8,2,2],kernel 取实际(可能含 padding 影响)
        if num_ups == 4:
            upsample_rates = [10, 8, 2, 2]
        elif num_ups == 3:
            upsample_rates = [10, 8, 2]
        else:
            upsample_rates = [8, 6, 4, 2][:num_ups]
        # kernel_sizes: RVC 常用 [20,16,4,4](与 stride 匹配)
        upsample_kernel_sizes = kernel_sizes or [20, 16, 4, 4][:num_ups]
        gin_channels = cond_w[2] if cond_w is not None else 0  # mlx (O, 1, C) 的 C=gin
        return dict(
            initial_channel=initial_channel,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=48000,
        )

    def __call__(self, x, f0, g=None, n_res=None):
        """x: [B, T, initial_channel]; f0: [B, T_frames] Hz; g: [B, gin] or [B,1,gin]。

        Returns wav [B, 1, L]。
        """
        har_source, _, _ = self.m_source(f0, self.upp)   # [B, L, 1] (NLC)
        # torch: har_source.transpose(1,2) 后 [B, 1, L];mlx 已 NLC [B, L, 1]
        if n_res is not None:
            n = int(n_res)
            if n * self.upp != har_source.shape[1]:
                har_source = interpolate_linear(
                    har_source.transpose(0, 2, 1), size=n * self.upp, align_corners=True
                ).transpose(0, 2, 1)
            if n != x.shape[1]:
                x = interpolate_linear(
                    x.transpose(0, 2, 1), size=n, align_corners=True
                ).transpose(0, 2, 1)

        x = self.conv_pre(x)
        if g is not None:
            # torch: g [B, gin, 1];mlx: 接受 [B, gin] -> [B,1,gin],或 [B,1,gin]
            if g.ndim == 2:
                g = g[:, None, :]  # [B, 1, gin]
            x = x + self.cond(g)

        for i, (ups, noise_conv) in enumerate(zip(self.ups, self.noise_convs)):
            if i < self.num_upsamples:
                x = nn.leaky_relu(x, self.lrelu_slope)
                x = ups(x)
                x_source = noise_conv(har_source)
                minl = min(x.shape[1], x_source.shape[1])
                x = x[:, :minl, :] + x_source[:, :minl, :]
                xs = None
                l = [i * self.num_kernels + j for j in range(self.num_kernels)]
                for j, resblock in enumerate(self.resblocks):
                    if j in l:
                        rb = resblock(x)
                        xs = rb if xs is None else xs + rb
                x = xs / self.num_kernels
        x = nn.leaky_relu(x, self.lrelu_slope)
        x = self.conv_post(x)
        return mx.tanh(x)  # [B, T_out, 1]


# ---------------------------------------------------------------------------
# f0_to_coarse —— mel 刻度量化(镜像 rvc/model.py::f0_to_coarse)
# ---------------------------------------------------------------------------

def f0_to_coarse(f0, f0_min: float = 50.0, f0_max: float = 1100.0):
    """f0: [B, T] Hz -> [B, T] tokens in [1, 255]。

    torch 版用 ``1127 * log(1 + f/700)``(mel 刻度)。
    """
    f0 = f0.astype(mx.float32)
    f0_mel_min = 1127.0 * mx.log(1.0 + f0_min / 700.0)
    f0_mel_max = 1127.0 * mx.log(1.0 + f0_max / 700.0)
    f0_mel = 1127.0 * mx.log(1.0 + f0 / 700.0)
    f0_mel = mx.where(
        f0_mel > 0,
        (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0,
        f0_mel,
    )
    f0_mel = mx.clip(f0_mel, 1.0, 255.0)
    return mx.round(f0_mel).astype(mx.int32)
