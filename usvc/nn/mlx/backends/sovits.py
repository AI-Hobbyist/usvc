"""MLX 版 so-vits(wav-domain VITS + NSF-HiFiGAN;镜像 usvc/backends/sovits/model.py 推理集)。
NLC 布局;weight_norm 由 _convert 加载时合并;relative-attention 因相对位置项加在
scores 与 output 上,手写 matmul 与 torch 逐位一致(SDPA 无法表达)。
F0Decoder(predict_f0=True)未移植(推理默认 predict_f0=False)。"""
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..nsf_hifigan import ResBlock1, ResBlock2, get_padding
from .._gan import DiscriminatorP, DiscriminatorS, MultiPeriodDiscriminator
from .._ops import sequence_mask as _sequence_mask  # 单一实现(原本地 copy 删除)

__all__ = [
    "LayerNorm", "WN", "Flip", "ResidualCouplingLayer", "ResidualCouplingBlock",
    "MultiHeadAttention", "FFN", "Encoder", "TextEncoder", "SourceModuleHnNSF",
    "Generator", "SoVITSModel", "SoVITSGenerator", "f0_to_coarse",
]

LRELU_SLOPE = 0.1
# so-vits f0 量化常量(model.py::f0_to_coarse)
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * math.log(1 + f0_min / 700)
f0_mel_max = 1127 * math.log(1 + f0_max / 700)


def n_fft_default():
    return 1025  # spec_channels = n_fft//2+1


def f0_to_coarse(f0):
    """f0 → coarse token(1..255)。⚠ 与 rvc.f0_to_coarse 数学等价但 FP 舍入不同
    (220k 样本 1 个 round 边界 token 差);两份实现必须共存,不可合并。"""
    f0 = f0.astype(mx.float32)
    f0_mel = 1127.0 * mx.log(1.0 + f0 / 700.0)
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = mx.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = mx.round(f0_mel).astype(mx.int32)
    b_gt0 = (f0_coarse > 0).astype(mx.int32); b_lt1 = (f0_coarse < 1).astype(mx.int32)
    b_lt_bin = (f0_coarse < f0_bin).astype(mx.int32); b_ge_bin = (f0_coarse >= f0_bin).astype(mx.int32)
    f0_coarse = f0_coarse * b_gt0 + b_lt1
    f0_coarse = f0_coarse * b_lt_bin + b_ge_bin * (f0_bin - 1)
    return f0_coarse


class LayerNorm(nn.Module):
    """gamma/beta 命名与 torch ckpt 键一致(so-vits 不用 weight/bias)。"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones(channels)
        self.beta = mx.zeros(channels)

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.gamma, self.beta, self.eps)  # NLC 天然按 C


class WN(nn.Module):
    """weight_norm dilated Conv1d + fused_add_tanh_sigmoid_multiply。"""

    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int,
                 n_layers: int, gin_channels: int = 0, p_dropout: int = 0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.in_layers = []
        self.res_skip_layers = []
        self.drop = nn.Dropout(p_dropout)
        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = get_padding(kernel_size, dilation)
            self.in_layers.append(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                          dilation=dilation, padding=padding)
            )
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(nn.Conv1d(hidden_channels, res_skip_channels, 1))

    def __call__(self, x, x_mask, g=None):
        output = mx.zeros_like(x); h = self.hidden_channels
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, :, i * 2 * h:(i + 1) * 2 * h] if g is not None else mx.zeros_like(x_in)
            in_act = x_in + g_l
            acts = self.drop(mx.tanh(in_act[..., :h]) * mx.sigmoid(in_act[..., h:]))
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[..., :h]) * x_mask
                output = output + res_skip_acts[..., h:]
            else:
                output = output + res_skip_acts
        return output * x_mask


class Flip(nn.Module):
    """沿通道维翻转(mean_only flow 的奇偶交替)。"""

    def __call__(self, x, *args, reverse=False, **kwargs):
        x = x[..., ::-1]
        if not reverse:
            return x, mx.zeros(x.shape[0])
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, p_dropout: int = 0,
                 gin_channels: int = 0, mean_only: bool = False,
                 wn_sharing_parameter=None):
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = (
            WN(hidden_channels, kernel_size, dilation_rate, n_layers,
               p_dropout=p_dropout, gin_channels=gin_channels)
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

    def __call__(self, x, x_mask, g=None, reverse=False):
        h = self.half_channels
        x0 = x[..., :h]; x1 = x[..., h:]
        stats = self.post(self.enc(self.pre(x0) * x_mask, x_mask, g=g)) * x_mask
        if not self.mean_only:
            m = stats[..., :h]; logs = stats[..., h:]
        else:
            m = stats; logs = mx.zeros_like(m)
        if not reverse:
            x1 = m + x1 * mx.exp(logs) * x_mask
            return mx.concatenate([x0, x1], axis=-1), mx.sum(logs, axis=(1, 2))
        x1 = (x1 - m) * mx.exp(-logs) * x_mask
        return mx.concatenate([x0, x1], axis=-1)


class ResidualCouplingBlock(nn.Module):
    """n_flows × (ResidualCouplingLayer + Flip);推理用 reverse=True。"""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, n_flows: int = 4,
                 gin_channels: int = 0, share_parameter: bool = False):
        super().__init__()
        self.n_flows = n_flows
        self.flows = []
        self.wn = (
            WN(hidden_channels, kernel_size, dilation_rate, n_layers,
               p_dropout=0, gin_channels=gin_channels)
            if share_parameter
            else None
        )
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                gin_channels=gin_channels, mean_only=True,
                wn_sharing_parameter=self.wn,
            ))
            self.flows.append(Flip())

    def __call__(self, x, x_mask, g=None, reverse=False):
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            if reverse:
                x = flow(x, x_mask, g=g, reverse=True)
            else:
                x, _ = flow(x, x_mask, g=g, reverse=False)
        return x


class MultiHeadAttention(nn.Module):
    """window_size=4 相对位置嵌入;相对项加在 scores 与 output 上,
    按 torch matmul 路径逐位移植(scale = k_channels**-0.5)。"""

    def __init__(self, channels: int, out_channels: int, n_heads: int,
                 p_dropout: float = 0.0, window_size=None, heads_share: bool = True,
                 block_length=None, proximal_bias: bool = False,
                 proximal_init: bool = False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels, self.out_channels, self.n_heads = channels, out_channels, n_heads
        self.p_dropout, self.window_size = p_dropout, window_size
        self.heads_share, self.block_length = heads_share, block_length
        self.proximal_bias, self.proximal_init = proximal_bias, proximal_init
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = mx.random.normal(
                shape=(n_heads_rel, window_size * 2 + 1, self.k_channels)
            ) * rel_stddev
            self.emb_rel_v = mx.random.normal(
                shape=(n_heads_rel, window_size * 2 + 1, self.k_channels)
            ) * rel_stddev

    def __call__(self, x, c, attn_mask=None):
        return self.conv_o(self._attention(self.conv_q(x), self.conv_k(c), self.conv_v(c), mask=attn_mask))

    def _attention(self, query, key, value, mask=None):
        B, t_t, C = query.shape
        t_s = key.shape[1]
        h, kc = self.n_heads, self.k_channels

        def _to_heads(a):
            return a.reshape(B, a.shape[1], h, kc).transpose(0, 2, 1, 3)

        q = _to_heads(query); k = _to_heads(key); v = _to_heads(value)
        scale = math.sqrt(kc)
        scores = mx.matmul(q / scale, k.transpose(0, 1, 3, 2))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)
            scores = scores + self._relative_position_to_absolute_position(
                self._matmul_with_relative_keys(q / scale, key_rel)
            )
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s)
        if mask is not None:
            scores = mx.where(mask == 0, mx.full(scores.shape, -1e4, scores.dtype), scores)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                scores = mx.where(self._triu_tril(t_s, self.block_length) == 0,
                                  mx.full(scores.shape, -1e4, scores.dtype), scores)
        p_attn = self.drop(mx.softmax(scores, axis=-1))
        output = mx.matmul(p_attn, v)
        if self.window_size is not None:
            rel_w = self._absolute_position_to_relative_position(p_attn)
            output = output + self._matmul_with_relative_values(
                rel_w, self._get_relative_embeddings(self.emb_rel_v, t_s)
            )
        return output.transpose(0, 2, 1, 3).reshape(B, t_t, C)

    @staticmethod
    def _triu_tril(length: int, block_length: int):
        d = mx.arange(length)[None, :] - mx.arange(length)[:, None]
        return ((d <= block_length) & (d >= -block_length)).astype(mx.float32)

    @staticmethod
    def _matmul_with_relative_values(x, y):
        return mx.matmul(x, y[None])

    @staticmethod
    def _matmul_with_relative_keys(x, y):
        return mx.matmul(x, y[None].transpose(0, 1, 3, 2))

    def _get_relative_embeddings(self, relative_embeddings, length):
        # [hr, 2W+1, kc] → [hr, 2*length-1, kc](pad 与 slice 二选一)
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        padded = relative_embeddings if pad_length == 0 else mx.pad(
            relative_embeddings, [(0, 0), (pad_length, pad_length), (0, 0)]
        )
        return padded[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        # [B,h,L,2L-1] → [B,h,L,L](torch view+pad 等价)
        batch, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)])
        x_flat = x.reshape(batch, heads, length * 2 * length)
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (0, length - 1)])
        x_final = x_flat.reshape(batch, heads, length + 1, 2 * length - 1)
        return x_final[:, :, :length, length - 1:]

    def _absolute_position_to_relative_position(self, x):
        # [B,h,L,L] → [B,h,L,2L-1](torch view+pad 等价)
        batch, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, length - 1)])
        x_flat = x.reshape(batch, heads, length * length + length * (length - 1))
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (length, 0)])
        x_final = x_flat.reshape(batch, heads, length, 2 * length)
        return x_final[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = mx.arange(length, dtype=mx.float32)
        return (-mx.log1p(mx.abs(r[None, :] - r[:, None])))[None, None]


class FFN(nn.Module):
    """same padding(same 分支 kernel//2 对称;kernel==1 或 causal 不同)。"""

    def __init__(self, in_channels: int, out_channels: int, filter_channels: int,
                 kernel_size: int, p_dropout: float = 0.0,
                 activation: str | None = None, causal: bool = False):
        super().__init__()
        self.kernel_size, self.activation, self.causal = kernel_size, activation, causal
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def __call__(self, x, x_mask):
        x = self.conv_1(self._pad(x * x_mask))
        if self.activation == "gelu":
            x = x * mx.sigmoid(1.702 * x)
        else:
            x = mx.maximum(x, 0.0)
        x = self.drop(x)
        x = self.conv_2(self._pad(x * x_mask))
        return x * x_mask

    def _pad(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1 if self.causal else (self.kernel_size - 1) // 2
        pad_r = 0 if self.causal else self.kernel_size // 2
        return mx.pad(x, [(0, 0), (pad_l, pad_r), (0, 0)])


class Encoder(nn.Module):
    """FFT 编码器(window_size=4 相对注意力,TextEncoder 用)。"""

    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int,
                 n_layers: int, kernel_size: int = 1, p_dropout: float = 0.0,
                 window_size: int = 4, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = [
            MultiHeadAttention(hidden_channels, hidden_channels, n_heads,
                               p_dropout=p_dropout, window_size=window_size)
            for _ in range(n_layers)
        ]
        self.norm_layers_1 = [LayerNorm(hidden_channels) for _ in range(n_layers)]
        self.ffn_layers = [
            FFN(hidden_channels, hidden_channels, filter_channels, kernel_size,
                p_dropout=p_dropout)
            for _ in range(n_layers)
        ]
        self.norm_layers_2 = [LayerNorm(hidden_channels) for _ in range(n_layers)]

    def __call__(self, x, x_mask):
        xm = x_mask[..., 0]  # [B, T]
        attn_mask = (xm[:, None, :] * xm[:, :, None])[:, None]  # [B,1,T,T]
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class TextEncoder(nn.Module):
    """f0_emb + Encoder + proj → z, m, logs。"""

    def __init__(self, out_channels: int, hidden_channels: int, kernel_size: int,
                 n_layers: int, gin_channels: int = 0,
                 filter_channels: int | None = None, n_heads: int | None = None,
                 p_dropout: float | None = None):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)
        self.enc_ = Encoder(hidden_channels, filter_channels, n_heads, n_layers,
                            kernel_size, p_dropout)

    def __call__(self, x, x_mask, f0=None, noice_scale=1):
        x = (x + self.f0_emb(f0)) * x_mask  # Embedding → [B, T, hidden],先加后 mask
        x = self.enc_(x, x_mask)
        stats = self.proj(x) * x_mask  # [B, T, 2*out]
        m = stats[..., :self.out_channels]; logs = stats[..., self.out_channels:]
        z = (m + mx.random.normal(shape=m.shape) * mx.exp(logs) * noice_scale) * x_mask
        return z, m, logs, x_mask


class SineGen(nn.Module):
    """**属性容器**(fast_sinegen 只用其属性;逐样本 forward 不需要)。
    保留属性以匹配 ckpt 键(m_source.l_sin_gen.*)。"""

    def __init__(self, samp_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1,
                 noise_std: float = 0.003, voiced_threshold: float = 0,
                 flag_for_pulse: bool = False):
        super().__init__()
        self.sine_amp, self.noise_std, self.harmonic_num = sine_amp, noise_std, harmonic_num
        self.dim, self.sampling_rate = harmonic_num + 1, samp_rate
        self.voiced_threshold, self.flag_for_pulse = voiced_threshold, flag_for_pulse


class SourceModuleHnNSF(nn.Module):
    """l_sin_gen + l_linear + l_tanh(谐波合并)。"""

    def __init__(self, sampling_rate: int, harmonic_num: int = 8, sine_amp: float = 0.1,
                 add_noise_std: float = 0.003, voiced_threshod: float = 0):
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
                                 add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()


class Generator(nn.Module):
    """so-vits NSF-HiFiGAN Generator(恒 fast_sinegen,harmonic=8;有 cond(speaker)
    层;最终 leaky_relu 用 **0.01**(F.leaky_relu 默认,循环内是 0.1))。"""

    def __init__(self, h):
        super().__init__()
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.upsample_rates = h["upsample_rates"]
        self.m_source = SourceModuleHnNSF(sampling_rate=h["sampling_rate"],
                                          harmonic_num=8, sine_amp=0.1)
        self.noise_convs = []
        self.conv_pre = nn.Conv1d(h["inter_channels"], h["upsample_initial_channel"],
                                  7, 1, padding=3)
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2

        self.ups = []
        self.resblocks = []
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            c_cur = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(h["upsample_initial_channel"] // (2 ** i), c_cur,
                                   k, stride=u, padding=(k - u + 1) // 2)
            )
            if i + 1 < self.num_upsamples:
                stride_f0 = math.prod(h["upsample_rates"][i + 1:])
                self.noise_convs.append(
                    nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2,
                              stride=stride_f0, padding=(stride_f0 + 1) // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
            for j, (rk, rd) in enumerate(
                zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])
            ):
                self.resblocks.append(resblock(c_cur, rk, rd))

        self.conv_post = nn.Conv1d(
            h["upsample_initial_channel"] // (2 ** self.num_upsamples), 1, 7, 1, padding=3
        )
        self.cond = nn.Conv1d(h["gin_channels"], h["upsample_initial_channel"], 1)
        self.upp = math.prod(h["upsample_rates"])

    def fast_sinegen(self, f0):
        """帧域 sine 生成([B,T] Hz → [B, T*upp, dim=9])。"""
        B, T = f0.shape
        upp = self.upp
        dim = self.m_source.l_sin_gen.dim
        sr = self.m_source.l_sin_gen.sampling_rate
        sine_amp = self.m_source.l_sin_gen.sine_amp
        noise_std = self.m_source.l_sin_gen.noise_std
        f0_harm = f0[:, :, None] * mx.arange(1, dim + 1, dtype=f0.dtype)[None, None, :]
        rad = (f0_harm / sr) % 1.0
        rand_ini = mx.random.uniform(shape=(B, dim))
        rand_ini = rand_ini.at[:, 0].subtract(rand_ini[:, 0])  # rand_ini[:, 0] = 0
        frame_boundary = mx.cumsum(rad * upp, axis=1)          # [B, T, dim]
        prev = mx.zeros_like(frame_boundary)
        prev = prev.at[:, 1:, :].add(frame_boundary[:, :-1, :])
        arange = mx.arange(1, upp + 1, dtype=f0.dtype)         # [upp]
        phase = prev[:, :, None, :] + rad[:, :, None, :] * arange[None, None, :, None]
        phase = phase.reshape(B, T * upp, dim) + rand_ini[:, None, :]
        sines = mx.sin(phase * 2 * mx.pi) * sine_amp
        uv = (f0[:, :, None] > 0).astype(sines.dtype)          # [B, T, 1]
        uv = mx.broadcast_to(uv[:, :, None, :], (B, T, upp, 1)).reshape(B, T * upp, 1)
        noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=(B, T * upp, dim))
        return sines * uv + noise

    def __call__(self, x, f0=None, g=None):
        har_source = self.m_source.l_tanh(self.m_source.l_linear(self.fast_sinegen(f0)))
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)  # g [B,T,gin] 或 [B,1,gin]
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            minl = min(x.shape[1], x_source.shape[1])
            x = x[:, :minl, :] + x_source[:, :minl, :]
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels
        x = nn.leaky_relu(x, 0.01)  # F.leaky_relu 默认 slope(与循环内 0.1 不同)
        return mx.tanh(self.conv_post(x))  # [B, L, 1]

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, key_prefix: str = "dec",
                  config: dict | None = None, strict: bool = False):
        from .._convert import load_torch_ckpt_to_mlx, load_torch_sd, strip_prefix
        from .._ops import merge_weight_norm
        if config is None:
            config = cls._infer_config(ckpt_path, key_prefix)
        obj = cls(config)
        raw = load_torch_sd(ckpt_path)
        merged = merge_weight_norm(raw)
        pref = (key_prefix + ".") if key_prefix else ""
        ups_keys = {
            k for k in merged
            if k.startswith(pref + "ups.") and k.endswith(".weight")
        }
        sd = load_torch_ckpt_to_mlx(ckpt_path, convtranspose_keys=ups_keys)
        obj.load_weights(list(strip_prefix(sd, pref).items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(ckpt_path: str | Path, key_prefix: str = "dec") -> dict:
        """从权重 shape 反推 h(upsample_rates 无法推断,用默认 [8,8,2,2,2])。"""
        from .._convert import load_torch_ckpt_to_mlx, load_torch_sd
        from .._ops import merge_weight_norm
        raw = load_torch_sd(ckpt_path)
        merged = merge_weight_norm(raw)
        sd = load_torch_ckpt_to_mlx(ckpt_path)
        pref = (key_prefix + ".") if key_prefix else ""
        cand = {}
        for k, v in sd.items():
            if pref and k.startswith(pref):
                cand[k[len(pref):]] = v
            elif not pref:
                cand[k] = v
        shape = lambda k: cand[k].shape if k in cand else None  # noqa: E731
        conv_pre_w = shape("conv_pre.weight")
        cond_w = shape("cond.weight")
        if conv_pre_w is None:
            raise ValueError(f"cannot infer so-vits Generator config from {ckpt_path}")
        inter_channels = conv_pre_w[2]
        upsample_initial_channel = conv_pre_w[0]
        ups = sorted(
            [k for k in cand if k.startswith("ups.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        num_ups = len(ups)
        kernel_sizes = [int(cand[k].shape[1]) for k in ups] if ups else None
        gin_channels = cond_w[2] if cond_w is not None else 0
        return dict(
            resblock="1",
            upsample_rates=[8, 8, 2, 2, 2][:num_ups],
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=kernel_sizes or [16, 16, 4, 4, 4][:num_ups],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            inter_channels=inter_channels,
            gin_channels=gin_channels,
            sampling_rate=44100,
        )


class SoVITSModel(nn.Module):
    """SynthesizerTrn(emb_g + pre + enc_p + dec + flow;config_template 默认值)。"""

    def __init__(
        self,
        n_units: int = 768,
        spec_channels=None,
        segment_size: int = 20,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock: str = "1",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(8, 8, 2, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes=(16, 16, 4, 4, 4),
        gin_channels: int = 768,
        ssl_dim: int | None = None,
        n_speakers: int = 200,
        sampling_rate: int = 44100,
        vol_embedding: bool = False,
        use_automatic_f0_prediction: bool = True,
        flow_share_parameter: bool = False,
        n_flow_layer: int = 4,
        n_layers_trans_flow: int = 3,
        use_transformer_flow: bool = False,
        **kwargs,
    ):
        super().__init__()
        if ssl_dim is None:
            ssl_dim = n_units
        if use_transformer_flow:
            raise NotImplementedError("use_transformer_flow=True 未移植")
        self.segment_size = segment_size
        self.vol_embedding = vol_embedding
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.resblock_kernel_sizes = list(resblock_kernel_sizes)
        self.resblock_dilation_sizes = [list(d) for d in resblock_dilation_sizes]
        self.upsample_rates = list(upsample_rates)
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = list(upsample_kernel_sizes)
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)
        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)
        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels=filter_channels,
            n_heads=n_heads, n_layers=n_layers, kernel_size=kernel_size,
            p_dropout=p_dropout,
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": self.resblock_kernel_sizes,
            "resblock_dilation_sizes": self.resblock_dilation_sizes,
            "upsample_rates": self.upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": self.upsample_kernel_sizes,
            "gin_channels": gin_channels,
        }
        self.dec = Generator(h=hps)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, n_flow_layer,
            gin_channels=gin_channels, share_parameter=flow_share_parameter,
        )
        self.emb_uv = nn.Embedding(2, hidden_channels)
        spec_channels = spec_channels if spec_channels else n_fft_default()
        self.enc_q = EncoderQ(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16,
            gin_channels=gin_channels,
        )
        self.spec_channels = spec_channels
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1, hidden_channels, filter_channels, n_heads, n_layers,
                kernel_size, p_dropout, spk_channels=gin_channels,
            )

    def infer(self, c, f0, uv, g=None, noice_scale: float = 0.35, seed: int = 52468,
              predict_f0: bool = False, vol=None):
        """NetGSvc.infer;predict_f0 未移植(默认 False)。"""
        if predict_f0:
            raise NotImplementedError(
                "so-vits predict_f0(F0Decoder)未移植;默认 predict_f0=False"
            )
        if g.ndim == 1:
            g = g[:, None]
        g_emb = self.emb_g(g)                                  # [B, 1, gin]
        B, T = c.shape[0], c.shape[1]
        x_mask = mx.ones((B, T, 1), dtype=c.dtype)
        vol_emb = (
            self.emb_vol(vol[:, :, None]) if vol is not None and self.vol_embedding
            else 0
        )
        x = self.pre(c) * x_mask + self.emb_uv(uv.astype(mx.int32)) + vol_emb
        z_p, m_p, logs_p, c_mask = self.enc_p(
            x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale
        )
        z = self.flow(z_p, c_mask, g=g_emb, reverse=True)
        return self.dec(z * c_mask, g=g_emb, f0=f0), f0

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, config: dict | None = None,
                  key_prefix: str | None = None, strict: bool = False):
        from .._convert import load_torch_ckpt_to_mlx, load_torch_sd
        from .._ops import merge_weight_norm
        obj = cls(**(config if config is not None else {}))
        raw = load_torch_sd(ckpt_path)
        merged = merge_weight_norm(raw)
        pref = (key_prefix + ".") if key_prefix else ""
        ups_keys = {
            k for k in merged
            if (pref + "dec.ups.") in k and k.endswith(".weight")
        }
        sd = load_torch_ckpt_to_mlx(ckpt_path, convtranspose_keys=ups_keys)
        model_sd = {}
        for k, v in sd.items():
            mk = k[len(pref):] if pref and k.startswith(pref) else k
            model_sd[mk] = v
        obj.load_weights(list(model_sd.items()), strict=strict)
        obj.eval()
        return obj


SoVITSGenerator = SoVITSModel  # 别名(任务命名约定)


def _sovits_forward_train(self, units, f0, spec, wave, spk_id, frame_len=None,
                          vol=None, uv=None, segment_size=None):
    return train_forward_sovits(self, units, f0, spec, wave, spk_id,
                                frame_len=frame_len, vol=vol, uv=uv,
                                segment_size=segment_size)


SoVITSModel.forward_train = _sovits_forward_train


class EncoderQ(nn.Module):
    """Posterior encoder: spec → z, m_q, logs_q(pre Conv1d + WN + proj,NLC)。"""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 kernel_size: int, dilation_rate: int, n_layers: int,
                 gin_channels: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(self, x, x_lengths, g=None):
        B, T, _ = x.shape
        x_mask = (mx.arange(T, dtype=mx.int32)[None, :] < x_lengths[:, None]) \
            .astype(x.dtype)[..., None]
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask          # [B, T, 2*out]
        m = stats[..., :self.out_channels]; logs = stats[..., self.out_channels:]
        z = (m + mx.random.normal(shape=m.shape) * mx.exp(logs)) * x_mask
        return z, m, logs, x_mask


class _FFTForF0Decoder(nn.Module):
    """F0Decoder 的 FFT(无 cond/isflow 分支;attn 用 subsequent_mask 因果掩码)。"""

    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int,
                 n_layers: int = 1, kernel_size: int = 1, p_dropout: float = 0.0):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = [
            MultiHeadAttention(hidden_channels, hidden_channels, n_heads,
                               p_dropout=p_dropout, window_size=None)
            for _ in range(n_layers)
        ]
        self.norm_layers_0 = [LayerNorm(hidden_channels) for _ in range(n_layers)]
        self.ffn_layers = [
            FFN(hidden_channels, hidden_channels, filter_channels, kernel_size,
                p_dropout=p_dropout, causal=True)
            for _ in range(n_layers)
        ]
        self.norm_layers_1 = [LayerNorm(hidden_channels) for _ in range(n_layers)]

    def __call__(self, x, x_mask):
        T = x.shape[1]
        attn_mask = mx.tril(mx.ones((1, 1, T, T), dtype=x.dtype))  # 下三角 1=attend
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.drop(self.self_attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_0[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_1[i](x + y)
        return x * x_mask


class F0Decoder(nn.Module):
    """prenet + FFT + proj + f0_prenet + cond → normalized f0。"""

    def __init__(self, out_channels: int, hidden_channels: int, filter_channels: int,
                 n_heads: int, n_layers: int, kernel_size: int, p_dropout: float,
                 spk_channels: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.spk_channels = spk_channels
        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = _FFTForF0Decoder(hidden_channels, filter_channels, n_heads,
                                        n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        if spk_channels > 0:
            self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def __call__(self, x, norm_f0, x_mask, spk_emb=None):
        x = mx.stop_gradient(x)
        if spk_emb is not None and self.spk_channels > 0:
            x = x + self.cond(spk_emb)
        x = x + self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        return self.proj(x) * x_mask


def _normalize_f0(f0, x_mask, uv, random_scale=True):
    """f0 按浊音均值归一(f0: [B,1,T];uv: [B,T];x_mask: [B,T,1])。"""
    uv_sum = uv.sum(axis=1, keepdims=True)              # [B, 1]
    uv_sum = mx.where(uv_sum == 0, mx.full_like(uv_sum, 9999.0), uv_sum)
    f0_b = f0[..., 0]                                    # [B, T]
    means = (f0_b * uv).sum(axis=1, keepdims=True) / uv_sum   # [B, 1]
    if random_scale:
        factor = mx.random.uniform(low=0.8, high=1.2, shape=(f0.shape[0], 1))
    else:
        factor = mx.ones((f0.shape[0], 1))
    return ((f0_b - means) * factor * x_mask[..., 0])[..., None]  # [B, T, 1]


def _rand_slice_segments_with_pitch(x, pitch, x_lengths, segment_size):
    """随机 slice 段 → (x_slice, pitch_slice, ids_str)。"""
    B, T, C = x.shape
    ids_str_max = (x_lengths - segment_size + 1).astype(mx.int32)
    r = mx.random.uniform(shape=(B,)).astype(mx.float32) * ids_str_max.astype(mx.float32)
    ids_str = r.astype(mx.int32)
    idx = ids_str[:, None] + mx.arange(segment_size)[None, :]   # [B, seg]
    return mx.take_along_axis(x, idx[..., None], axis=1), \
        mx.take_along_axis(pitch, idx, axis=1), ids_str


def train_forward_sovits(self, units, f0, spec, wave, spk_id, frame_len=None,
                         vol=None, uv=None, segment_size=None):
    """so-vits 训练 forward → dict(wave_gen, z_p, m_p, logs_p, m_q, logs_q,
    z_mask, ids_slice, pred_lf0, lf0, c_lengths)。"""
    B = units.shape[0]
    if segment_size is None:
        segment_size = self.segment_size
    if spk_id is None:
        spk_id = mx.zeros((B,), dtype=mx.int32)
    if spk_id.ndim == 1:
        spk_id = spk_id[:, None]
    g = self.emb_g(spk_id)                                   # [B, 1, gin]
    c_lengths = (
        frame_len if frame_len is not None
        else mx.full((B,), units.shape[1], dtype=mx.int32)
    )
    spec_lengths = c_lengths
    if uv is None:
        uv = (f0 > 0).astype(mx.float32)
    x_mask = _sequence_mask(c_lengths, units.shape[1])[..., None].astype(units.dtype)
    vol_emb = (
        self.emb_vol(vol[:, :, None]) if vol is not None and self.vol_embedding
        else 0
    )
    x = self.pre(units) * x_mask + self.emb_uv(uv.astype(mx.int32)) + vol_emb

    lf0 = None
    pred_lf0 = None
    if self.use_automatic_f0_prediction:
        lf0 = 2595.0 * mx.log10(1.0 + f0[:, :, None] / 700.0) / 500.0   # [B, T, 1]
        norm_lf0 = _normalize_f0(lf0, x_mask, uv)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g) if hasattr(self, "f0_decoder") else None

    z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
    z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
    z_p = self.flow(z, spec_mask, g=g)
    z_slice, pitch_slice, ids_slice = _rand_slice_segments_with_pitch(
        z, f0, spec_lengths, segment_size
    )
    o = self.dec(z_slice, f0=pitch_slice, g=g)               # [B, seg*hop, 1]
    return {
        "wave_gen": o,
        "z_p": z_p,
        "m_p": m_p,
        "logs_p": logs_p,
        "m_q": m_q,
        "logs_q": logs_q,
        "z_mask": spec_mask,
        "ids_slice": ids_slice,
        "pred_lf0": pred_lf0,
        "lf0": lf0,
        "c_lengths": c_lengths,
    }