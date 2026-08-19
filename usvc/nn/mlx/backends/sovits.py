"""MLX 版 so-vits(classic wav-domain VITS + NSF-HiFiGAN)。

镜像 ``usvc/backends/sovits/model.py`` 的推理最小集:
  - WN:weighted conv(dilated)+ fused_add_tanh_sigmoid_multiply
  - Flip + ResidualCouplingLayer + ResidualCouplingBlock(flow,reverse 推理)
  - Generator:conv_pre + ups(ConvTranspose1d)+ resblocks + SourceModuleHnNSF
    (f0→sine,harmonic=8,fast_sinegen)+ conv_post + cond(speaker)
  - MultiHeadAttention(window_size=4 相对位置)+ FFN + Encoder + LayerNorm
  - TextEncoder:f0_emb(Embedding 256)+ Encoder + proj
  - SoVITSModel(SynthesizerTrn):emb_g + pre + enc_p + dec + flow,infer()

与 torch 版的差异(见 MLX 映射手册):
  - 输入 NLC [B, T, C];torch 是 NCL
  - 删除 channels_last 优化路径(forward_cl)
  - weight_norm 权重由 ``_convert.load_torch_ckpt_to_mlx`` 在加载时合并
  - ConvTranspose1d 权重 (I,O,K)→(O,K,I) 由 convtranspose_keys 处理
  - relative-attention 因相对位置项加在 scores 与 output 上,
    用手写 matmul(与 torch 逐位一致),不用 mx.fast.scaled_dot_product_attention
  - F0Decoder(predict_f0=True)未移植(推理默认 predict_f0=False)
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..nsf_hifigan import ResBlock1, ResBlock2, get_padding
from .._gan import DiscriminatorP, DiscriminatorS, MultiPeriodDiscriminator

__all__ = [
    "LayerNorm",
    "WN",
    "Flip",
    "ResidualCouplingLayer",
    "ResidualCouplingBlock",
    "MultiHeadAttention",
    "FFN",
    "Encoder",
    "TextEncoder",
    "SourceModuleHnNSF",
    "Generator",
    "SoVITSModel",
    "SoVITSGenerator",
    "f0_to_coarse",
]

LRELU_SLOPE = 0.1

# so-vits f0 量化常量(model.py::f0_to_coarse)
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * math.log(1 + f0_min / 700)
f0_mel_max = 1127 * math.log(1 + f0_max / 700)


def n_fft_default():
    """so-vits 默认 spec_channels = n_fft//2+1 = 1025。"""
    return 1025


def f0_to_coarse(f0):
    """f0: [B, T] Hz → [B, T] coarse token(1..255)。镜像 model.py::f0_to_coarse。"""
    f0 = f0.astype(mx.float32)
    f0_mel = 1127.0 * mx.log(1.0 + f0 / 700.0)
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = mx.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = mx.round(f0_mel).astype(mx.int32)
    # torch 的 bool*int 自动提升为 int;mlx 需显式转 int32
    b_gt0 = (f0_coarse > 0).astype(mx.int32)
    b_lt1 = (f0_coarse < 1).astype(mx.int32)
    b_lt_bin = (f0_coarse < f0_bin).astype(mx.int32)
    b_ge_bin = (f0_coarse >= f0_bin).astype(mx.int32)
    f0_coarse = f0_coarse * b_gt0
    f0_coarse = f0_coarse + b_lt1
    f0_coarse = f0_coarse * b_lt_bin
    f0_coarse = f0_coarse + b_ge_bin * (f0_bin - 1)
    return f0_coarse


# ---------------------------------------------------------------------------
# LayerNorm(torch so-vits 用 gamma/beta 命名,MLX 自定义以匹配 ckpt 键)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """so-vits LayerNorm:gamma/beta 命名与 torch ckpt 键一致。"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones(channels)
        self.beta = mx.zeros(channels)

    def __call__(self, x):
        # x [B, T, C];torch 先 transpose 到 [B, T, C] 再按最后一维 norm,
        # mlx NLC 天然最后一维是 C
        return mx.fast.layer_norm(x, self.gamma, self.beta, self.eps)


# ---------------------------------------------------------------------------
# WN(weighted conv)
# ---------------------------------------------------------------------------

class WN(nn.Module):
    """so-vits WN(镜像 model.py::WN)。

    - in_layers:n_layers 层 dilated weight_norm Conv1d
    - res_skip_layers:n_layers 层 weight_norm Conv1d(1x1)
    - cond_layer:gin_channels != 0 时存在
    - fused_add_tanh_sigmoid_multiply:split 两半,tanh(left)*sigmoid(right)
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: int = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

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
            self.res_skip_layers.append(
                nn.Conv1d(hidden_channels, res_skip_channels, 1)
            )

    def __call__(self, x, x_mask, g=None):
        """x: [B, T, hidden];x_mask: [B, T, 1];g: [B, T, gin] 或 None。"""
        output = mx.zeros_like(x)
        h = self.hidden_channels

        if g is not None:
            g = self.cond_layer(g)  # [B, T, 2*h*n_layers]

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * h
                g_l = g[:, :, cond_offset:cond_offset + 2 * h]
            else:
                g_l = mx.zeros_like(x_in)
            # fused_add_tanh_sigmoid_multiply
            in_act = x_in + g_l
            t_act = mx.tanh(in_act[..., :h])
            s_act = mx.sigmoid(in_act[..., h:])
            acts = t_act * s_act
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[..., :h]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[..., h:]
            else:
                output = output + res_skip_acts
        return output * x_mask


# ---------------------------------------------------------------------------
# Flow:Flip + ResidualCouplingLayer + ResidualCouplingBlock
# ---------------------------------------------------------------------------

class Flip(nn.Module):
    """沿通道维翻转(so-vits mean_only flow 的奇偶交替)。"""

    def __call__(self, x, *args, reverse=False, **kwargs):
        x = x[..., ::-1]
        if not reverse:
            logdet = mx.zeros(x.shape[0])
            return x, logdet
        return x


class ResidualCouplingLayer(nn.Module):
    """Residual coupling(镜像 model.py::ResidualCouplingLayer)。"""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: int = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
        wn_sharing_parameter=None,
    ):
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
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
        x0 = x[..., :h]
        x1 = x[..., h:]
        hh = self.pre(x0) * x_mask
        hh = self.enc(hh, x_mask, g=g)
        stats = self.post(hh) * x_mask
        if not self.mean_only:
            m = stats[..., :h]
            logs = stats[..., h:]
        else:
            m = stats
            logs = mx.zeros_like(m)
        if not reverse:
            x1 = m + x1 * mx.exp(logs) * x_mask
            x = mx.concatenate([x0, x1], axis=-1)
            logdet = mx.sum(logs, axis=(1, 2))
            return x, logdet
        else:
            x1 = (x1 - m) * mx.exp(-logs) * x_mask
            x = mx.concatenate([x0, x1], axis=-1)
            return x


class ResidualCouplingBlock(nn.Module):
    """n_flows × (ResidualCouplingLayer + Flip)。推理用 reverse=True。"""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = []
        self.wn = (
            WN(hidden_channels, kernel_size, dilation_rate, n_layers,
               p_dropout=0, gin_channels=gin_channels)
            if share_parameter
            else None
        )
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate,
                    n_layers, gin_channels=gin_channels, mean_only=True,
                    wn_sharing_parameter=self.wn,
                )
            )
            self.flows.append(Flip())

    def __call__(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


# ---------------------------------------------------------------------------
# MultiHeadAttention + FFN + Encoder(TextEncoder 的 prior 编码器)
# ---------------------------------------------------------------------------

def convert_pad_shape(pad_shape):
    """torch convert_pad_shape(反转 + 展平)等价,用于 F.pad 的维度对齐。"""
    l = pad_shape[::-1]
    return [item for sublist in l for item in sublist]


class MultiHeadAttention(nn.Module):
    """so-vits MHA(window_size=4 相对位置嵌入)。

    因相对位置项加在 scores 与 output 上,SDPA 无法表达,按 torch
    matmul 路径逐位移植(scale = k_channels**-0.5)。
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size=None,
        heads_share: bool = True,
        block_length=None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = (
                mx.random.normal(
                    shape=(n_heads_rel, window_size * 2 + 1, self.k_channels)
                ) * rel_stddev
            )
            self.emb_rel_v = (
                mx.random.normal(
                    shape=(n_heads_rel, window_size * 2 + 1, self.k_channels)
                ) * rel_stddev
            )

    def __call__(self, x, c, attn_mask=None):
        """x, c: [B, T, channels];attn_mask: [B, 1, T, T] float(1=attend)。"""
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self._attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    # -- 内部 attention 数学(torch 逐位移植) ---------------------------------

    def _attention(self, query, key, value, mask=None):
        # query/key/value: [B, T, C] NLC → 内部 [B, h, T, kc]
        B, t_t, C = query.shape
        t_s = key.shape[1]
        h, kc = self.n_heads, self.k_channels

        def _to_heads(a):
            return a.reshape(B, a.shape[1], h, kc).transpose(0, 2, 1, 3)

        q = _to_heads(query)   # [B, h, t_t, kc]
        k = _to_heads(key)     # [B, h, t_s, kc]
        v = _to_heads(value)   # [B, h, t_s, kc]

        scale = math.sqrt(kc)
        scores = mx.matmul(q / scale, k.transpose(0, 1, 3, 2))  # [B, h, t_t, t_s]

        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)  # [hr, 2t_s-1, kc]
            rel_logits = self._matmul_with_relative_keys(q / scale, key_rel)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s)

        if mask is not None:
            scores = mx.where(mask == 0, mx.full(scores.shape, -1e4, scores.dtype), scores)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = self._triu_tril(t_s, self.block_length)
                scores = mx.where(block_mask == 0, mx.full(scores.shape, -1e4, scores.dtype), scores)

        p_attn = mx.softmax(scores, axis=-1)  # [B, h, t_t, t_s]
        p_attn = self.drop(p_attn)

        output = mx.matmul(p_attn, v)  # [B, h, t_t, kc]
        if self.window_size is not None:
            rel_w = self._absolute_position_to_relative_position(p_attn)  # [B, h, t_t, 2t_s-1]
            v_rel = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(rel_w, v_rel)

        output = output.transpose(0, 2, 1, 3).reshape(B, t_t, C)
        return output

    @staticmethod
    def _triu_tril(length: int, block_length: int):
        """torch 的 triu(-b).tril(b) 等价 → [L, L] float 0/1。"""
        r = mx.arange(length)[:, None]
        c = mx.arange(length)[None, :]
        d = c - r
        m = (d <= block_length) & (d >= -block_length)
        return m.astype(mx.float32)

    def _matmul_with_relative_values(self, x, y):
        # x: [B, h, t_t, 2t_s-1], y: [hr, 2t_s-1, kc]
        return mx.matmul(x, y[None])

    def _matmul_with_relative_keys(self, x, y):
        # x: [B, h, t_t, kc], y: [hr, 2t_s-1, kc]
        return mx.matmul(x, y[None].transpose(0, 1, 3, 2))

    def _get_relative_embeddings(self, relative_embeddings, length):
        # [hr, 2W+1, kc] → [hr, 2*length-1, kc]
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            padded = mx.pad(
                relative_embeddings,
                [(0, 0), (pad_length, pad_length), (0, 0)],
            )
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        # x: [B, h, L, 2L-1] → [B, h, L, L](torch view+pad 等价)
        batch, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)])          # [B, h, L, 2L]
        x_flat = x.reshape(batch, heads, length * 2 * length)     # [B, h, 2L²]
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (0, length - 1)])  # [B, h, 2L²+L-1]
        x_final = x_flat.reshape(batch, heads, length + 1, 2 * length - 1)
        return x_final[:, :, :length, length - 1:]               # [B, h, L, L]

    def _absolute_position_to_relative_position(self, x):
        # x: [B, h, L, L] → [B, h, L, 2L-1](torch view+pad 等价)
        batch, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, length - 1)])  # [B, h, L, 2L-1]
        x_flat = x.reshape(batch, heads, length * length + length * (length - 1))
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (length, 0)])
        x_final = x_flat.reshape(batch, heads, length, 2 * length)
        return x_final[:, :, :, 1:]                              # [B, h, L, 2L-1]

    def _attention_bias_proximal(self, length):
        r = mx.arange(length, dtype=mx.float32)
        diff = r[None, :] - r[:, None]
        return (-mx.log1p(mx.abs(diff)))[None, None]


class FFN(nn.Module):
    """so-vits FFN(same padding,激活可选 gelu/relu)。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str | None = None,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

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
        if self.causal:
            pad_l = self.kernel_size - 1
            pad_r = 0
        else:
            pad_l = (self.kernel_size - 1) // 2
            pad_r = self.kernel_size // 2
        return mx.pad(x, [(0, 0), (pad_l, pad_r), (0, 0)])


class Encoder(nn.Module):
    """so-vits FFT 编码器(window_size=4 相对注意力,TextEncoder 用)。"""

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = [
            MultiHeadAttention(
                hidden_channels, hidden_channels, n_heads,
                p_dropout=p_dropout, window_size=window_size,
            )
            for _ in range(n_layers)
        ]
        self.norm_layers_1 = [LayerNorm(hidden_channels) for _ in range(n_layers)]
        self.ffn_layers = [
            FFN(hidden_channels, hidden_channels, filter_channels,
                kernel_size, p_dropout=p_dropout)
            for _ in range(n_layers)
        ]
        self.norm_layers_2 = [LayerNorm(hidden_channels) for _ in range(n_layers)]

    def __call__(self, x, x_mask):
        # x: [B, T, C];x_mask: [B, T, 1]
        xm = x_mask[..., 0]                                    # [B, T]
        attn_mask = xm[:, None, :] * xm[:, :, None]            # [B, T, T]
        attn_mask = attn_mask[:, None]                         # [B, 1, T, T]
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


# ---------------------------------------------------------------------------
# TextEncoder(prior:units + f0 → z_p, m_p, logs_p)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """so-vits TextEncoder(f0_emb + Encoder + proj)。"""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        gin_channels: int = 0,
        filter_channels: int | None = None,
        n_heads: int | None = None,
        p_dropout: float | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)
        self.enc_ = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout,
        )

    def __call__(self, x, x_mask, f0=None, noice_scale=1):
        """x: [B, T, hidden];x_mask: [B, T, 1];f0: [B, T] coarse token。"""
        x = x + self.f0_emb(f0)                     # Embedding → [B, T, hidden]
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask               # [B, T, 2*out]
        m = stats[..., :self.out_channels]
        logs = stats[..., self.out_channels:]
        z = (m + mx.random.normal(shape=m.shape) * mx.exp(logs) * noice_scale) * x_mask
        return z, m, logs, x_mask


# ---------------------------------------------------------------------------
# SourceModuleHnNSF + Generator(so-vits NSF-HiFiGAN)
# ---------------------------------------------------------------------------

class SineGen(nn.Module):
    """so-vits SineGen 的**属性容器**(fast_sinegen 只用其属性)。

    推理实际走 Generator.fast_sinegen(torch 非 onnx 路径),SineGen 的
    逐样本 forward 不需要 —— 保留属性以匹配 ckpt 键(m_source.l_sin_gen.*)。
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        flag_for_pulse: bool = False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).astype(f0.dtype)


class SourceModuleHnNSF(nn.Module):
    """so-vits source module(l_linear + l_tanh;谐波合并)。"""

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp,
            add_noise_std, voiced_threshod,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()


class Generator(nn.Module):
    """so-vits NSF-HiFiGAN Generator(镜像 model.py::Generator)。

    与 vocoder/RVC 的 Generator 区别:
      - 恒用 fast_sinegen(f0 → [B, T*upp, dim],harmonic=8)
      - 有 cond(speaker)层,``x = conv_pre(x) + cond(g)``
      - 每级 ups 后 noise_convs[i] 处理全速率 har_source
      - 最终 leaky_relu 用 **0.01**(torch F.leaky_relu 默认值)
    """

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.upsample_rates = h["upsample_rates"]

        self.m_source = SourceModuleHnNSF(
            sampling_rate=h["sampling_rate"], harmonic_num=8, sine_amp=0.1
        )
        self.noise_convs = []
        self.conv_pre = nn.Conv1d(
            h["inter_channels"], h["upsample_initial_channel"], 7, 1, padding=3
        )
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2

        self.ups = []
        self.resblocks = []
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            c_cur = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(
                    h["upsample_initial_channel"] // (2 ** i),
                    c_cur, k, stride=u, padding=(k - u + 1) // 2,
                )
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

    # -- 帧域 sine 生成(torch fast_sinegen,harmonic=8) -----------------------

    def fast_sinegen(self, f0):
        """f0: [B, T] Hz → [B, T*upp, dim](dim=harmonic+1=9)。"""
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

        # 帧域相位边界
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
        """x: [B, T, inter];f0: [B, T_frames] Hz;g: [B, T, gin] 或 [B, 1, gin]。

        Returns wav [B, L, 1]。
        """
        har_source = self.fast_sinegen(f0)                     # [B, L, dim]
        har_source = self.m_source.l_tanh(self.m_source.l_linear(har_source))
        # [B, L, 1](NLC;torch 同路径 transpose(1,2) 到 [B, 1, L] 是因为 NCL)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)                               # g [B,T,gin] 或 [B,1,gin]

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            minl = min(x.shape[1], x_source.shape[1])
            x = x[:, :minl, :] + x_source[:, :minl, :]
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                rb = self.resblocks[idx](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels

        # torch: F.leaky_relu(x) —— 默认 slope 0.01(与循环内的 0.1 不同)
        x = nn.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        return mx.tanh(x)                                      # [B, L, 1]

    # -- ckpt 加载 -----------------------------------------------------------

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, key_prefix: str = "dec",
                  config: dict | None = None, strict: bool = False):
        """从 so-vits ckpt 加载 Generator(权重键在 ``dec.*`` 下)。

        Args:
            ckpt_path: torch ckpt。
            key_prefix: Generator 权重在 ckpt 中的前缀(so-vits 默认 ``dec``)。
            config: Generator 构造参数(h dict);None 时从权重反推。
            strict: 是否严格匹配权重键。
        """
        from .._convert import load_torch_ckpt_to_mlx, load_torch_sd
        from .._ops import merge_weight_norm

        if config is None:
            config = cls._infer_config(ckpt_path, key_prefix)
        obj = cls(config)

        # 识别 ups 的 ConvTranspose 权重键(需特殊转置)
        raw = load_torch_sd(ckpt_path)
        merged = merge_weight_norm(raw)
        pref = (key_prefix + ".") if key_prefix else ""
        ups_keys = {
            k for k in merged
            if k.startswith(pref + "ups.") and k.endswith(".weight")
        }
        sd = load_torch_ckpt_to_mlx(ckpt_path, convtranspose_keys=ups_keys)
        dec_sd = {}
        for k, v in sd.items():
            if pref and k.startswith(pref):
                dec_sd[k[len(pref):]] = v
            elif not pref:
                dec_sd[k] = v
        obj.load_weights(list(dec_sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(ckpt_path: str | Path, key_prefix: str = "dec") -> dict:
        """从权重 shape 反推 Generator 构造参数(h dict)。

        推断:inter_channels、upsample_initial_channel、gin_channels、
        upsample_kernel_sizes、resblock 类型。upsample_rates 无法从 shape
        推断(在 stride 里),用 so-vits 默认 [8,8,2,2,2]。
        """
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
        resblock = "1"
        return dict(
            resblock=resblock,
            upsample_rates=[8, 8, 2, 2, 2][:num_ups],
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=kernel_sizes or [16, 16, 4, 4, 4][:num_ups],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            inter_channels=inter_channels,
            gin_channels=gin_channels,
            sampling_rate=44100,
        )


# ---------------------------------------------------------------------------
# SoVITSModel(SynthesizerTrn,推理最小集)
# ---------------------------------------------------------------------------

class SoVITSModel(nn.Module):
    """so-vits SynthesizerTrn(emb_g + pre + enc_p + dec + flow)。

    Args(so-vits config_template 默认值):
        n_units/ssl_dim: content 特征维(768 for vec768l12)
        inter_channels / hidden_channels / filter_channels / n_heads / n_layers
        gin_channels / n_speakers / sampling_rate / vol_embedding
        upsample_*: dec 上采样配置
        n_flow_layer: flow 层数(默认 4)
    """

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

        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = list(resblock_kernel_sizes)
        self.resblock_dilation_sizes = [list(d) for d in resblock_dilation_sizes]
        self.upsample_rates = list(upsample_rates)
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = list(upsample_kernel_sizes)
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.sampling_rate = sampling_rate

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

        # 训练组件(posterior encoder + f0 预测头)
        self.enc_q = EncoderQ(
            spec_channels if spec_channels else n_fft_default(),
            inter_channels, hidden_channels, 5, 1, 16,
            gin_channels=gin_channels,
        )
        self.spec_channels = spec_channels if spec_channels else n_fft_default()
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1, hidden_channels, filter_channels, n_heads, n_layers,
                kernel_size, p_dropout, spk_channels=gin_channels,
            )

    # -- 推理 ---------------------------------------------------------------

    def infer(
        self,
        c,
        f0,
        uv,
        g=None,
        noice_scale: float = 0.35,
        seed: int = 52468,
        predict_f0: bool = False,
        vol=None,
    ):
        """so-vits NetGSvc.infer(镜像 model.py::SoVITSModel.infer)。

        Args:
            c: [B, T, ssl_dim] content 特征(NLC;torch 是 [B, ssl_dim, T])。
            f0: [B, T] 连续 f0(Hz)。
            uv: [B, T] 0/1 浊音标志。
            g: [B, 1] 或 [B] speaker id。
            noice_scale: prior 噪声尺度。
            predict_f0: 用 f0 预测头(未移植,默认 False)。
            vol: [B, T] 音量(仅 vol_embedding)。

        Returns:
            (o, f0):wav [B, L, 1] 和(可能的预测)f0。
        """
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
        o = self.dec(z * c_mask, g=g_emb, f0=f0)
        return o, f0

    # -- ckpt 加载 -----------------------------------------------------------

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str | Path,
        *,
        config: dict | None = None,
        key_prefix: str | None = None,
        strict: bool = False,
    ):
        """从 so-vits ckpt 加载完整模型。

        Args:
            ckpt_path: torch ckpt。
            config: SoVITSModel 构造参数;None 时用默认(config_template)。
            key_prefix: 权重前缀(so-vits 多为裸键;若 ckpt 有 ``model.``/
                ``net_g.`` 前缀可传入)。
            strict: 是否严格匹配权重键。
        """
        from .._convert import load_torch_ckpt_to_mlx, load_torch_sd
        from .._ops import merge_weight_norm

        cfg = config if config is not None else {}
        obj = cls(**cfg)
        kwargs = {}
        if key_prefix is not None:
            kwargs["key_remap"] = {f"{key_prefix}.{k}": k for k in ()}

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
            mk = k
            if pref and k.startswith(pref):
                mk = k[len(pref):]
            model_sd[mk] = v
        obj.load_weights(list(model_sd.items()), strict=strict)
        obj.eval()
        return obj


# 别名(任务命名约定)
SoVITSGenerator = SoVITSModel


# 训练 forward 绑定(镜像 torch SoVITSModel.forward)
def _sovits_forward_train(self, units, f0, spec, wave, spk_id, frame_len=None,
                          vol=None, uv=None, segment_size=None):
    return train_forward_sovits(
        self, units, f0, spec, wave, spk_id,
        frame_len=frame_len, vol=vol, uv=uv, segment_size=segment_size,
    )


SoVITSModel.forward_train = _sovits_forward_train


# ---------------------------------------------------------------------------
# 训练组件:EncoderQ / F0Decoder / MultiPeriodDiscriminator / 训练 forward
# ---------------------------------------------------------------------------


class EncoderQ(nn.Module):
    """Posterior encoder: spec → z, m_q, logs_q。

    镜像 torch model.py::EncoderQ(pre Conv1d + WN + proj)。NLC 布局。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(self, x, x_lengths, g=None):
        """x: [B, T, spec_channels] spec;g: [B, 1, gin] 或 None。

        Returns (z, m_q, logs_q, x_mask);x_mask: [B, T, 1]。
        """
        import mlx.core as _mx
        B, T, _ = x.shape
        seq = _mx.arange(T, dtype=_mx.int32)[None, :]
        x_mask = (seq < x_lengths[:, None]).astype(x.dtype)[..., None]
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask          # [B, T, 2*out]
        m = stats[..., : self.out_channels]
        logs = stats[..., self.out_channels:]
        z = (m + _mx.random.normal(shape=m.shape) * _mx.exp(logs)) * x_mask
        return z, m, logs, x_mask


class _FFTForF0Decoder(nn.Module):
    """F0Decoder 的 FFT(镜像 torch model.py::FFT,无 cond/isflow 分支)。

    用 mlx MultiHeadAttention + FFN + LayerNorm 组装(causal=True 仅
    ffn;attn 用 subsequent_mask 因果掩码)。
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int = 1,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = [
            MultiHeadAttention(
                hidden_channels, hidden_channels, n_heads,
                p_dropout=p_dropout, window_size=None,
            )
            for _ in range(n_layers)
        ]
        self.norm_layers_0 = [LayerNorm(hidden_channels) for _ in range(n_layers)]
        self.ffn_layers = [
            FFN(hidden_channels, hidden_channels, filter_channels,
                kernel_size, p_dropout=p_dropout, causal=True)
            for _ in range(n_layers)
        ]
        self.norm_layers_1 = [LayerNorm(hidden_channels) for _ in range(n_layers)]

    def __call__(self, x, x_mask):
        import mlx.core as _mx
        T = x.shape[1]
        # subsequent_mask:下三角 [1, 1, T, T](1=attend)
        attn_mask = _mx.tril(_mx.ones((1, 1, T, T), dtype=x.dtype))
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        return x * x_mask


class F0Decoder(nn.Module):
    """Predict normalized f0 from hidden states(so-vits F0Decoder)。

    镜像 torch model.py::F0Decoder(prenet + FFT + proj + f0_prenet + cond)。
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        spk_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.spk_channels = spk_channels
        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = _FFTForF0Decoder(
            hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        if spk_channels > 0:
            self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def __call__(self, x, norm_f0, x_mask, spk_emb=None):
        import mlx.core as _mx
        x = _mx.stop_gradient(x)
        if spk_emb is not None and self.spk_channels > 0:
            x = x + self.cond(spk_emb)
        x = x + self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


def _normalize_f0(f0, x_mask, uv, random_scale=True):
    """Normalize f0 by voiced mean(so-vits utils.normalize_f0)。

    f0: [B, 1, T];uv: [B, T];x_mask: [B, T, 1]。
    """
    import mlx.core as _mx
    uv_sum = uv.sum(axis=1, keepdims=True)              # [B, 1]
    uv_sum = _mx.where(uv_sum == 0, _mx.full_like(uv_sum, 9999.0), uv_sum)
    f0_b = f0[..., 0]                                    # [B, T]
    means = (f0_b * uv).sum(axis=1, keepdims=True) / uv_sum   # [B, 1]
    if random_scale:
        factor = _mx.random.uniform(low=0.8, high=1.2, shape=(f0.shape[0], 1))
    else:
        factor = _mx.ones((f0.shape[0], 1))
    f0_norm = (f0_b - means) * factor
    return (f0_norm * x_mask[..., 0])[..., None]        # [B, T, 1](NLC)


def _rand_slice_segments_with_pitch(x, pitch, x_lengths, segment_size):
    """随机 slice 段(so-vits commons.rand_slice_segments_with_pitch)。

    x: [B, T, C];pitch: [B, T];返回 (x_slice, pitch_slice, ids_str)。
    """
    import mlx.core as _mx
    B, T, C = x.shape
    ids_str_max = (x_lengths - segment_size + 1).astype(_mx.int32)
    r = _mx.random.uniform(shape=(B,)).astype(_mx.float32) * ids_str_max.astype(_mx.float32)
    ids_str = r.astype(_mx.int32)
    idx = ids_str[:, None] + _mx.arange(segment_size)[None, :]   # [B, seg]
    x_slice = _mx.take_along_axis(x, idx[..., None], axis=1)
    pitch_slice = _mx.take_along_axis(pitch, idx, axis=1)
    return x_slice, pitch_slice, ids_str


def _sequence_mask(lengths, max_length=None):
    """[B] lengths → [B, max_length] bool。"""
    import mlx.core as _mx
    if max_length is None:
        max_length = int(lengths.max().item())
    seq = _mx.arange(max_length, dtype=_mx.int32)[None, :]
    return seq < lengths[:, None]


def train_forward_sovits(self, units, f0, spec, wave, spk_id, frame_len=None,
                         vol=None, uv=None, segment_size=None):
    """so-vits 训练 forward(镜像 torch SoVITSModel.forward)。

    Args:
        units: [B, T, ssl_dim]
        f0: [B, T] Hz
        spec: [B, T_spec, spec_channels] linear spec(NLC)
        wave: [B, L] 或 [B, L, 1](仅 API parity)
        spk_id: [B]
        frame_len: [B]
        vol: [B, T] 音量(仅 vol_embedding)
        uv: [B, T] 浊音

    Returns dict(wave_gen, z_p, m_p, logs_p, m_q, logs_q, z_mask,
                 ids_slice, pred_lf0, lf0, c_lengths)。
    """
    import mlx.core as _mx
    B = units.shape[0]
    if segment_size is None:
        segment_size = self.segment_size
    if spk_id is None:
        spk_id = _mx.zeros((B,), dtype=_mx.int32)
    if spk_id.ndim == 1:
        spk_id = spk_id[:, None]
    g = self.emb_g(spk_id)                                   # [B, 1, gin]

    c = units
    c_lengths = (
        frame_len if frame_len is not None
        else _mx.full((B,), c.shape[1], dtype=_mx.int32)
    )
    spec_lengths = c_lengths
    if uv is None:
        uv = (f0 > 0).astype(_mx.float32)

    x_mask = _sequence_mask(c_lengths, c.shape[1])[..., None].astype(c.dtype)
    vol_emb = (
        self.emb_vol(vol[:, :, None]) if vol is not None and self.vol_embedding
        else 0
    )
    x = self.pre(c) * x_mask + self.emb_uv(uv.astype(_mx.int32)) + vol_emb

    lf0 = None
    pred_lf0 = None
    if self.use_automatic_f0_prediction:
        lf0 = 2595.0 * _mx.log10(1.0 + f0[:, :, None] / 700.0) / 500.0   # [B, T, 1]
        norm_lf0 = _normalize_f0(lf0, x_mask, uv)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g) if hasattr(self, "f0_decoder") else None

    z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
    z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
    z_p = self.flow(z, spec_mask, g=g)

    # rand slice:z [B, T, inter], f0 [B, T]
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
