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
    "RVCTextEncoder",
    "RVCModel",
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

        sd = load_torch_ckpt_to_mlx(
            ckpt_path, convtranspose_keys=cls._ups_weights(ckpt_path, key_prefix)
        )
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
    def _ups_weights(ckpt_path: str | Path, key_prefix: str | None = None) -> set[str]:
        """从 ckpt 收集 ConvTranspose1d 权重键(ups.*.weight,含前缀)。

        RVC GeneratorNSF 的 ups 是 weight_norm ConvTranspose1d,merge_weight_norm
        后键为 ``{prefix}.ups.{i}.weight``,需要 (I,O,K)→(O,K,I) 转置。
        """
        from .._convert import load_torch_sd
        from .._ops import merge_weight_norm

        merged = merge_weight_norm(load_torch_sd(ckpt_path))
        pref = (key_prefix + ".") if key_prefix else ""
        return {
            k for k in merged
            if k.startswith(pref + "ups.") and k.endswith(".weight")
        }

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


# ---------------------------------------------------------------------------
# RVC TextEncoder(prior:units + pitch → m, logs, x_mask)
# 镜像 rvc/model.py::TextEncoder;attention 组件复用 sovits 的 Encoder
# (RVC window_size=10,与 so-vits 4 不同)。
# ---------------------------------------------------------------------------

def sequence_mask(lengths, max_length=None):
    """lengths: [B] int -> [B, T] bool(T = max_length)。镜像 model.py::sequence_mask。"""
    if max_length is None:
        max_length = int(lengths.max())
    x = mx.arange(max_length, dtype=lengths.dtype)[None, :]
    return x < lengths[:, None]


class RVCTextEncoder(nn.Module):
    """RVC prior encoder:emb_phone(Linear n_units→hidden)+ emb_pitch(Embedding 256)
    + Encoder(attention)+ proj(Conv1d hidden→inter*2)。

    Args(镜像 rvc/model.py::TextEncoder):
        in_channels: units 维(768)。
        out_channels: inter_channels(192),proj 输出 2*out。
        hidden_channels / filter_channels / n_heads / n_layers / kernel_size /
        p_dropout: Encoder 参数。
        f0: True 时建 emb_pitch(推理恒 True)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        # RVC 默认 window_size=10(so-vits 默认 4;影响 rel-pos emb 形状)
        from .sovits import Encoder

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            window_size=10,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(self, phone, pitch, lengths, skip_head=None):
        """phone: [B, T, n_units];pitch: [B, T] token;lengths: [B] int。

        Returns:
            m: [B, T, out] 均值
            logs: [B, T, out] 对数方差
            x_mask: [B, T, 1]
        """
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)   # [b, t, h]
        x = self.lrelu(x)
        # torch 转 [B, h, T] → mlx 保持 NLC [B, T, h]
        x_mask = sequence_mask(lengths, x.shape[1])[..., None].astype(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        if skip_head is not None:
            head = int(skip_head)
            x = x[:, head:, :]
            x_mask = x_mask[:, head:, :]
        stats = self.proj(x) * x_mask
        m = stats[..., : self.out_channels]
        logs = stats[..., self.out_channels:]
        return m, logs, x_mask


# ---------------------------------------------------------------------------
# RVCModel = SynthesizerTrnMs768NSFsid(RVC v2,推理路径)
#   units → enc_p → (m, logs) → z_p → flow(reverse)→ z → dec → wav
# 镜像 rvc/model.py::RVCModel::infer(确定性版:z = m,无采样噪声,便于数值验证)。
# 注意 enc_q(训练 posterior)不参与推理,未移植。
# ---------------------------------------------------------------------------

class RVCModel(nn.Module):
    """RVC v2 推理模型(emb_g + enc_p + flow + dec)。

    默认参数对应 RVC configs/v2/48k.json:inter=192, hidden=192, filter=768,
    n_heads=2, n_layers=6, kernel=3, upsample_rates=[12,10,2,2],
    upsample_initial_channel=512, upsample_kernel_sizes=[24,20,4,4],
    gin_channels=256, sr=48k。
    """

    def __init__(
        self,
        n_units=768,
        spec_channels=1025,
        segment_size=20,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(12, 10, 2, 2),
        upsample_initial_channel=512,
        upsample_kernel_sizes=(24, 20, 4, 4),
        spk_embed_dim=1,
        gin_channels=256,
        sr=48000,
        is_half=False,
        vol_embedding=False,
        **kwargs,
    ):
        super().__init__()
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim

        self.enc_p = RVCTextEncoder(
            n_units,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            list(resblock_kernel_sizes),
            [list(d) for d in resblock_dilation_sizes],
            list(upsample_rates),
            upsample_initial_channel,
            list(upsample_kernel_sizes),
            gin_channels=gin_channels,
            sr=sr,
            is_half=is_half,
        )
        # flow:同 so-vits(mean_only=True, n_flows=4),复用 sovits 实现
        from .sovits import ResidualCouplingBlock

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        self.vol_embedding = vol_embedding
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)

    # -- 推理 ---------------------------------------------------------------

    def infer(
        self,
        phone,
        f0,
        pitch,
        lengths,
        spk_id=0,
        skip_head=None,
        return_length=None,
        vol=None,
    ):
        """RVC 推理链路(确定性,镜像 rvc/model.py::RVCModel.infer)。

        Args:
            phone: [B, T, n_units] content 特征(NLC)。
            f0: [B, T] 连续 f0(Hz)。
            pitch: [B, T] f0 token(1-255,f0_to_coarse 输出)。
            lengths: [B] 帧长。
            spk_id: int 或 [B] speaker id。
            skip_head / return_length: 流式推理(可选)。
            vol: [B, T] 音量(仅 vol_embedding)。

        Returns:
            wav: [B, L, 1](确定性 z = m,无采样噪声)。
        """
        g = self.emb_g(spk_id)                    # [B, gin] 或 [gin]
        if g.ndim == 1:
            g = g[None, :]                        # [1, gin]
        g = g[:, None, :]                         # [B, 1, gin](NLC)

        vol_emb = (
            self.emb_vol(vol[:, :, None])
            if vol is not None and self.vol_embedding
            else None
        )
        if skip_head is not None and return_length is not None:
            head = int(skip_head)
            length = int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m, logs, x_mask = self.enc_p(phone, pitch, lengths, skip_head=flow_head)
            z_p = (m + mx.exp(logs) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, dec_head : dec_head + length, :]
            x_mask = x_mask[:, dec_head : dec_head + length, :]
            f0 = f0[:, head : head + length]
        else:
            m, logs, x_mask = self.enc_p(phone, pitch, lengths)
            z_p = (m + mx.exp(logs) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, f0, g=g)
        return o

    def f0_to_coarse(self, f0, f0_min=50.0, f0_max=1100.0):
        """f0 → coarse token(静态方法委托)。"""
        return f0_to_coarse(f0, f0_min, f0_max)

    # -- ckpt 加载 ----------------------------------------------------------

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str | Path,
        *,
        config: dict | None = None,
        key_prefix: str | None = None,
        strict: bool = False,
    ):
        """从 RVC torch ckpt 加载完整推理模型。

        RVC ckpt 权重在 ``generator.*``(usvc 训练格式)或裸键 ``dec.*``/
        ``enc_p.*``。

        Args:
            ckpt_path: torch ckpt(lightning/RVC wrapper)。
            config: RVCModel 构造参数 dict;None 用默认(48k v2)。
            key_prefix: 权重前缀(如 "generator");None 时自动探测。
            strict: 是否严格匹配权重键。
        """
        from .._convert import load_torch_ckpt_to_mlx

        cfg = config if config is not None else {}
        obj = cls(**cfg)

        # 探测前缀:优先 generator.(usvc/RVC 训练格式)
        prefix = key_prefix
        if prefix is None:
            from .._convert import load_torch_sd

            raw = load_torch_sd(ckpt_path)
            if any(k.startswith("generator.") for k in raw):
                prefix = "generator"
            else:
                prefix = ""
        pref = (prefix + ".") if prefix else ""

        sd = load_torch_ckpt_to_mlx(
            ckpt_path,
            convtranspose_keys=GeneratorNSF._ups_weights(
                ckpt_path,
                key_prefix=(prefix + ".dec") if prefix else "dec",
            ),
        )
        model_sd = {}
        for k, v in sd.items():
            mk = k
            if pref and k.startswith(pref):
                mk = k[len(pref):]
            elif not pref and ("." in k and (k.startswith("enc_p.") or k.startswith("dec.")
                                             or k.startswith("flow.") or k.startswith("emb_g."))):
                mk = k.split(".", 1)[-1]
            # 只保留推理需要的模块(enc_q 训练专用)
            if mk.startswith("enc_q."):
                continue
            model_sd[mk] = v
        obj.load_weights(list(model_sd.items()), strict=strict)
        obj.eval()
        return obj
