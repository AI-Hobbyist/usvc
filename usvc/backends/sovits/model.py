"""so-vits-svc classic model: VITS + NSF-HiFiGAN.

Extracted from so-vits-svc (SynthesizerTrn + MultiPeriodDiscriminator),
with torchaudio removed. Inline dependencies: commons, modules (WN/attention),
attentions (FFT/MHA), vdecoder/hifigan (Generator/MPD/DiscriminatorS/P).

Architecture (so-vits classic, configs_template/config_template.json):
  inter_channels=192, hidden_channels=192, filter_channels=768, n_heads=2,
  n_layers=6, kernel_size=3, p_dropout=0.1, resblock=1,
  resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5]x3],
  upsample_rates=[8,8,2,2,2], upsample_initial_channel=512,
  upsample_kernel_sizes=[16,16,4,4,4], gin_channels=768, ssl_dim=768,
  n_speakers=200, use_automatic_f0_prediction=True.

Generator forward returns dict:
  {wave_gen, z_p, m_p, logs_p, m_q, logs_q, z_mask, ids_slice,
   pred_lf0, norm_lf0, lf0, uv, c_lengths}
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

LRELU_SLOPE = 0.1

# channels_last 推理辅助统一在 usvc/nn/cl_helper.py（rvc/sovits/vocoder/wavenet 共享）
from ...nn.cl_helper import conv2d_cl, convT2d_cl

# ---------------------------------------------------------------------------
# mel_spectrogram / spec_to_mel (from so-vits-svc/modules/mel_processing.py)
# ---------------------------------------------------------------------------

_mel_basis_cache: dict = {}
_hann_window_cache: dict = {}


def _spectrogram_torch(
    y: torch.Tensor, n_fft: int, sampling_rate: int, hop_size: int,
    win_size: int, center: bool = False, eps: float = 1e-6,
) -> torch.Tensor:
    """Linear spectrogram from waveform.

    Bit-matches so-vits (eps=1e-6 via view_as_real) or RVC (eps=2e-7 via
    real/imag) mel_processing. y: [B, samples] or [samples].
    Returns [B, n_fft//2+1, T] magnitudes.
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)

    dtype_device = str(y.dtype) + "_" + str(y.device)
    key = f"{win_size}_{dtype_device}"
    if key not in _hann_window_cache:
        _hann_window_cache[key] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    pad = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
    y = y.squeeze(1)

    y_dtype = y.dtype
    if y.dtype == torch.bfloat16:
        y = y.to(torch.float32)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window_cache[key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    if eps == 1e-6:
        # so-vits: view_as_real -> sqrt(re^2+im^2+1e-6)
        spec = torch.view_as_real(spec).to(y_dtype)
        spec = torch.sqrt(spec.pow(2).sum(-1) + eps)
    else:
        # RVC: sqrt(real^2 + imag^2 + 2e-7)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + eps)
        spec = spec.to(y_dtype)
    return spec


def spec_to_mel(
    spec: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int,
    fmin: float, fmax: float | None, clip_val: float = 1e-5,
) -> torch.Tensor:
    """Linear spec -> log mel.

    Bit-matches so-vits (clip_val=1e-5) or RVC (clip_val=2e-6) mel_processing.
    """
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}_{dtype_device}"
    if key not in _mel_basis_cache:
        import librosa

        mel = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        _mel_basis_cache[key] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    mel = torch.matmul(_mel_basis_cache[key], spec)
    mel = torch.log(torch.clamp(mel, min=clip_val))
    return mel


def mel_spectrogram(
    y: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int,
    hop_size: int, win_size: int, fmin: float, fmax: float | None,
    center: bool = False, clip_val: float = 1e-5, eps: float = 1e-6,
) -> torch.Tensor:
    """Waveform -> log mel.

    Bit-matches so-vits (default clip_val=1e-5/eps=1e-6) or RVC
    (clip_val=2e-6/eps=2e-7) mel_processing. y: [B, samples] or [samples].
    Returns [B, num_mels, T].
    """
    spec = _spectrogram_torch(
        y, n_fft, sampling_rate, hop_size, win_size, center, eps=eps
    )
    return spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax, clip_val=clip_val)


# ---------------------------------------------------------------------------
# commons helpers (from so-vits-svc/modules/commons.py)
# ---------------------------------------------------------------------------


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_pitch_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
    return ret, ret_pitch, ids_str


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


# ---------------------------------------------------------------------------
# f0 helpers (from so-vits-svc/utils.py)
# ---------------------------------------------------------------------------

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


def normalize_f0(f0, x_mask, uv, random_scale=True):
    """Normalize f0 by voiced mean, optional random scaling (so-vits utils)."""
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    return f0_norm * x_mask


# ---------------------------------------------------------------------------
# modules (from so-vits-svc/modules/modules.py)
# NOTE: use_depthwise_conv=False is the default so-vits config; depthwise
# conv (DSConv) is not extracted. weight_norm_modules == weight_norm here.
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WN(torch.nn.Module):
    def __init__(
        self, hidden_channels, kernel_size, dilation_rate, n_layers,
        gin_channels=0, p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = Conv1d(
                hidden_channels, 2 * hidden_channels, kernel_size,
                dilation=dilation, padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self, channels, hidden_channels, kernel_size, dilation_rate, n_layers,
        p_dropout=0, gin_channels=0, mean_only=False, wn_sharing_parameter=None,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
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
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self, channels, hidden_channels, kernel_size, dilation_rate, n_layers,
        n_flows=4, gin_channels=0, share_parameter=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            WN(hidden_channels, kernel_size, dilation_rate, n_layers,
               p_dropout=0, gin_channels=gin_channels)
            if share_parameter
            else None
        )

        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate,
                    n_layers, gin_channels=gin_channels, mean_only=True,
                    wn_sharing_parameter=self.wn,
                )
            )
            self.flows.append(Flip())

    def remove_weight_norm(self):
        for flow in self.flows:
            if hasattr(flow, "remove_weight_norm"):
                flow.remove_weight_norm()
        if self.wn is not None:
            self.wn.remove_weight_norm()

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


# ---------------------------------------------------------------------------
# attentions (from so-vits-svc/modules/attentions.py)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(
        self, channels, out_channels, n_heads, p_dropout=0.0,
        window_size=None, heads_share=True, block_length=None,
        proximal_bias=False, proximal_init=False,
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
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(
        self, in_channels, out_channels, filter_channels, kernel_size,
        p_dropout=0.0, activation=None, causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x


class Encoder(nn.Module):
    """FFT encoder used by TextEncoder (prior)."""

    def __init__(
        self, hidden_channels, filter_channels, n_heads, n_layers,
        kernel_size=1, p_dropout=0.0, window_size=4, **kwargs,
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
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads,
                    p_dropout=p_dropout, window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels,
                    kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
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


class FFT(nn.Module):
    """FFT with optional speaker conditioning (used by F0Decoder)."""

    def __init__(
        self, hidden_channels, filter_channels, n_heads, n_layers=1,
        kernel_size=1, p_dropout=0.0, proximal_bias=False,
        proximal_init=True, isflow=False, **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        if isflow:
            cond_layer = torch.nn.Conv1d(
                kwargs["gin_channels"], 2 * hidden_channels * n_layers, 1
            )
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")
            self.gin_channels = kwargs["gin_channels"]
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads,
                    p_dropout=p_dropout, proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels,
                    kernel_size, p_dropout=p_dropout, causal=True)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            g = self.cond_layer(g)

        self_attn_mask = subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        x = x * x_mask
        for i in range(self.n_layers):
            if g is not None:
                x = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x = fused_add_tanh_sigmoid_multiply(
                    x, g_l, torch.IntTensor([self.hidden_channels])
                )
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        x = x * x_mask
        return x


# ---------------------------------------------------------------------------
# VITS components (from so-vits-svc/models.py)
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Prior encoder: units + f0_emb → z_p, m_p, logs_p."""

    def __init__(
        self, out_channels, hidden_channels, kernel_size, n_layers,
        gin_channels=0, filter_channels=None, n_heads=None, p_dropout=None,
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

    def forward(self, x, x_mask, f0=None, noice_scale=1):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask
        return z, m, logs, x_mask


class EncoderQ(nn.Module):
    """Posterior encoder: spec → z, m_q, logs_q."""

    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size,
        dilation_rate, n_layers, gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class F0Decoder(nn.Module):
    """Predict normalized f0 from hidden states (so-vits F0Decoder)."""

    def __init__(
        self, out_channels, hidden_channels, filter_channels, n_heads,
        n_layers, kernel_size, p_dropout, spk_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = FFT(hidden_channels, filter_channels, n_heads, n_layers,
                           kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if spk_emb is not None:
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


# ---------------------------------------------------------------------------
# NSF-HiFiGAN decoder (from so-vits-svc/vdecoder/hifigan/models.py)
# ---------------------------------------------------------------------------


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

    def forward_cl(self, x):
        """channels_last 版（x: [B,C,1,T] CL，等价于 forward）。"""
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv2d_cl(xt, c1)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2d_cl(xt, c2)
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

    def forward_cl(self, x):
        """channels_last 版（x: [B,C,1,T] CL，等价于 forward）。"""
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv2d_cl(xt, c)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


def padDiff(x):
    return F.pad(F.pad(x, (0, 0, -1, 1), "constant", 0) - x, (0, 0, 0, -1), "constant", 0)


@torch.compiler.disable
class SineGen(torch.nn.Module):
    """Sine generator for NSF-HiFiGAN."""

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0, flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.onnx = False

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1

        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2],
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        if not self.flag_for_pulse:
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0, upp=None):
        if self.onnx:
            with torch.no_grad():
                f0 = f0[:, None].transpose(1, 2)
                f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
                f0_buf[:, :, 0] = f0[:, :, 0]
                for idx in np.arange(self.harmonic_num):
                    f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
                rad_values = (f0_buf / self.sampling_rate) % 1
                rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
                rand_ini[:, 0] = 0
                rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
                tmp_over_one = torch.cumsum(rad_values, 1)  # % 1
                tmp_over_one *= upp
                tmp_over_one = F.interpolate(
                    tmp_over_one.transpose(2, 1), scale_factor=upp,
                    mode="linear", align_corners=True,
                ).transpose(2, 1)
                rad_values = F.interpolate(
                    rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
                ).transpose(2, 1)
                tmp_over_one %= 1
                tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
                cumsum_shift = torch.zeros_like(rad_values)
                cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
                sine_waves = torch.sin(
                    torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
                )
                sine_waves = sine_waves * self.sine_amp
                uv = self._f02uv(f0)
                uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise
        else:
            with torch.no_grad():
                fn = torch.multiply(
                    f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
                )
                sine_waves = self._f02sine(fn) * self.sine_amp
                uv = self._f02uv(f0)
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """Source module for hn-nsf."""

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
                                 add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(torch.nn.Module):
    """NSF-HiFiGAN generator (so-vits vdecoder/hifigan/models.py)."""

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(h["upsample_rates"]))
        self.m_source = SourceModuleHnNSF(sampling_rate=h["sampling_rate"], harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(
            Conv1d(h["inter_channels"], h["upsample_initial_channel"], 7, 1, padding=3)
        )
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            c_cur = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h["upsample_initial_channel"] // (2 ** i),
                        h["upsample_initial_channel"] // (2 ** (i + 1)),
                        k, u, padding=(k - u + 1) // 2,
                    )
                )
            )
            if i + 1 < len(h["upsample_rates"]):
                stride_f0 = np.prod(h["upsample_rates"][i + 1:])
                self.noise_convs.append(
                    Conv1d(1, c_cur, kernel_size=stride_f0 * 2,
                           stride=stride_f0, padding=(stride_f0 + 1) // 2)
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.cond = nn.Conv1d(h["gin_channels"], h["upsample_initial_channel"], 1)
        self.upp = np.prod(h["upsample_rates"])
        self.onnx = False
        self._arange_cache = None  # cached [1..upp] for fast_sinegen

    @torch.compiler.disable
    def fast_sinegen(self, f0):
        """Frame-domain sine generator (mathematically equivalent to the
        sample-domain cumsum path, but ~200x faster).

        f0: [B, T] Hz at frame rate. Returns sine_waves [B, L, dim] where
        L = T * upp.

        The original path upsamples f0 to audio rate (T*upp) then does a
        sequential cumsum over L samples — a scan that cannot be parallelized
        and dominates sovits inference (~290ms for 10s).

        With nearest upsample, f0 is piecewise-constant, so cumulative phase
        decomposes into:
          frame boundary phase (frame-domain cumsum, length T)
          + intra-frame linear ramp (broadcast arange, length upp)
        Random initial phase propagates as a constant offset (cumsum of a
        delta). sin(2π·x) is periodic, so integer-wrap shifts have no effect.
        """
        B, T = f0.shape
        upp, dim = self.upp, self.m_source.l_sin_gen.dim
        sr = self.m_source.l_sin_gen.sampling_rate
        sine_amp = self.m_source.l_sin_gen.sine_amp
        noise_std = self.m_source.l_sin_gen.noise_std

        # harmonics: [B, T, dim]
        f0_harm = f0.unsqueeze(-1) * torch.arange(
            1, dim + 1, device=f0.device, dtype=f0.dtype
        )
        rad = (f0_harm / sr) % 1

        rand_ini = torch.rand(B, dim, device=f0.device)
        rand_ini[:, 0] = 0

        # frame-domain phase boundary
        frame_boundary = torch.cumsum(rad * upp, dim=1)  # [B, T, dim]
        prev = torch.zeros_like(frame_boundary)
        prev[:, 1:, :] = frame_boundary[:, :-1, :]

        if self._arange_cache is None or self._arange_cache.shape[0] != upp:
            self._arange_cache = torch.arange(
                1, upp + 1, dtype=f0.dtype, device=f0.device
            )
        arange = self._arange_cache

        phase = prev.unsqueeze(2) + rad.unsqueeze(2) * arange[None, None, :, None]
        phase = phase.reshape(B, T * upp, dim) + rand_ini[:, None, :]

        sines = torch.sin(phase * 2 * np.pi) * sine_amp
        uv = (f0[:, :, None] > 0).to(sines.dtype)
        uv = uv.unsqueeze(2).expand(B, T, upp, 1).reshape(B, T * upp, 1)
        noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
        noise = noise_amp * torch.randn(B, T * upp, dim, device=f0.device)
        return sines * uv + noise

    def forward(self, x, f0, g=None):
        if getattr(self, "_use_cl", False):
            return self.forward_cl(x, f0, g)
        if not self.onnx:
            # Frame-domain fast sinegen: avoids the T*upp sequential cumsum
            # (~290ms -> ~1ms for 10s audio). f0 stays at frame rate [B, T].
            har_source = self.fast_sinegen(f0)  # [B, T*upp, dim]
            # merge harmonics -> [B, 1, T*upp]
            har_source = self.m_source.l_tanh(
                self.m_source.l_linear(
                    har_source.to(self.m_source.l_linear.weight.dtype)
                )
            ).transpose(1, 2)
        else:
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, noi_source, uv = self.m_source(f0, self.upp)
            har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
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

    def forward_cl(self, x, f0, g=None):
        """channels_last 版（x: [B,inter,1,T] CL，等价于 forward）。"""
        if not self.onnx:
            har_source = self.fast_sinegen(f0)
            har_source = self.m_source.l_tanh(
                self.m_source.l_linear(
                    har_source.to(self.m_source.l_linear.weight.dtype)
                )
            ).transpose(1, 2)
        else:
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, noi_source, uv = self.m_source(f0, self.upp)
            har_source = har_source.transpose(1, 2)
        hs4 = har_source[:, :, None, :].to(memory_format=torch.channels_last)
        x4 = x[:, :, None, :].to(memory_format=torch.channels_last)
        x = conv2d_cl(x4, self.conv_pre)
        g4 = g[:, :, None, :].to(memory_format=torch.channels_last)
        x = x + conv2d_cl(g4, self.cond)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = convT2d_cl(x, self.ups[i])
            x_source = conv2d_cl(hs4, self.noise_convs[i])
            minl = min(x.size(3), x_source.size(3))
            x = x[:, :, :, :minl] + x_source[:, :, :, :minl]
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = rb.forward_cl(x)
                else:
                    xs = xs + rb.forward_cl(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = conv2d_cl(x, self.conv_post)
        x = torch.tanh(x)
        return x[:, :, 0, :]  # [B,1,T]（view，无拷贝，与 forward 同形状）

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# ---------------------------------------------------------------------------
# Discriminators (from so-vits-svc/models.py — DiscriminatorS + MPD)
# ---------------------------------------------------------------------------


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1,
                          padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """MPD with periods [2,3,5,7,11] + DiscriminatorS (so-vits models.py)."""

    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
                         for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ---------------------------------------------------------------------------
# SoVITSModel (from so-vits-svc/models.py SynthesizerTrn)
# ---------------------------------------------------------------------------


class SoVITSModel(nn.Module):
    """so-vits classic generator (SynthesizerTrn).

    Args (so-vits config_template defaults):
        spec_channels: linear spec channels (filter_length//2+1 = 1025)
        segment_size:  frames per training segment (train.segment_size//hop)
        inter_channels / hidden_channels / filter_channels / n_heads / n_layers:
            VITS transformer dims
        ssl_dim: content feature dim (768 for vec768l12)
        n_speakers: speaker embedding count
        gin_channels: speaker embedding dim
        use_automatic_f0_prediction: enable F0Decoder
    """

    def __init__(
        self,
        n_units=768,                    # alias for ssl_dim (accepted for API compat)
        spec_channels=None,             # None -> inter_channels (small-model smoke test);
                                        # real so-vits: filter_length//2+1 = 1025
        segment_size=20,                # 10240 // 512
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
        upsample_rates=(8, 8, 2, 2, 2),
        upsample_initial_channel=512,
        upsample_kernel_sizes=(16, 16, 4, 4, 4),
        gin_channels=768,
        ssl_dim=None,                   # if given, overrides n_units
        n_speakers=200,
        sampling_rate=44100,
        vol_embedding=False,
        use_depthwise_conv=False,
        use_automatic_f0_prediction=True,
        flow_share_parameter=False,
        n_flow_layer=4,
        n_layers_trans_flow=3,
        use_transformer_flow=False,
        # RVC-style aliases accepted for API compatibility (task smoke test)
        n_sdp_layers0=None,
        n_sdp_layers=None,
        **kwargs,
    ):
        super().__init__()
        if ssl_dim is None:
            ssl_dim = n_units
        if n_sdp_layers is not None and n_sdp_layers != n_layers:
            n_layers = n_sdp_layers
        if n_sdp_layers0 is not None and n_sdp_layers0 != n_layers:
            n_layers = n_sdp_layers0
        if use_depthwise_conv:
            raise NotImplementedError(
                "use_depthwise_conv=True is not extracted (so-vits default is False)"
            )

        if spec_channels is None:
            spec_channels = inter_channels  # small-model default; real so-vits: 1025
        self.spec_channels = spec_channels
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
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
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
            "use_depthwise_conv": use_depthwise_conv,
        }
        self.dec = Generator(h=hps)

        self.enc_q = EncoderQ(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            raise NotImplementedError(
                "use_transformer_flow=True is not extracted (so-vits default is False)"
            )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, n_flow_layer,
            gin_channels=gin_channels, share_parameter=flow_share_parameter,
        )
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1, hidden_channels, filter_channels, n_heads, n_layers,
                kernel_size, p_dropout, spk_channels=gin_channels,
            )
        self.emb_uv = nn.Embedding(2, hidden_channels)

    def forward(
        self,
        units: torch.Tensor,   # [B, T, ssl_dim]
        f0: torch.Tensor,      # [B, T]
        spec: torch.Tensor,    # [B, spec_channels, T_spec]
        wave: torch.Tensor,    # [B, 1, L] real wave (unused in gen; for API parity)
        spk_id: torch.Tensor,  # [B]
        frame_len: torch.Tensor | None = None,  # [B] units frame lengths
        vol: torch.Tensor | None = None,  # [B, T] volume (only if vol_embedding)
        uv: torch.Tensor | None = None,  # [B, T] voicing from raw f0 (pre-interp)
    ) -> dict:
        """Training forward. Returns dict of tensors for loss computation.

        Returns:
            wave_gen: [B, 1, segment_size * prod(upsample_rates)]
            z_p: flow output [B, inter, T_spec]
            m_p/logs_p: prior stats [B, inter, T_spec]
            m_q/logs_q: posterior stats [B, inter, T_spec]
            z_mask: [B, 1, T_spec] posterior mask
            ids_slice: [B] frame slice indices for wave/mel slicing
            pred_lf0/norm_lf0/lf0: f0 prediction outputs (if enabled)
            uv: [B, T] voiced flag (from raw f0, pre-interpolation)
            c_lengths: [B] unit lengths
        """
        # speaker embedding: so-vits passes spk_id [B, 1]; accept [B] too
        if spk_id.dim() == 1:
            spk_id = spk_id.unsqueeze(-1)
        g = self.emb_g(spk_id).transpose(1, 2)  # [B, 1, gin] -> [B, gin, 1]
        c = units.transpose(1, 2)
        c_lengths = (
            frame_len if frame_len is not None
            else torch.full((c.size(0),), c.size(2), device=c.device, dtype=torch.long)
        )
        spec_lengths = c_lengths  # so-vits aligns spec to units length
        if uv is None:
            uv = (f0 > 0).float()  # fallback; only correct for raw (uninterp) f0

        x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        vol_emb = (
            self.emb_vol(vol[:, :, None]).transpose(1, 2)
            if vol is not None and self.vol_embedding
            else 0
        )
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol_emb

        # f0 prediction
        if self.use_automatic_f0_prediction:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
            norm_lf0 = normalize_f0(lf0, x_mask, uv)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        else:
            lf0 = 0
            norm_lf0 = 0
            pred_lf0 = 0

        # prior encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
        # posterior encoder
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        # flow
        z_p = self.flow(z, spec_mask, g=g)
        z_slice, pitch_slice, ids_slice = rand_slice_segments_with_pitch(
            z, f0, spec_lengths, self.segment_size
        )
        # NSF decoder
        o = self.dec(z_slice, g=g, f0=pitch_slice)

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
            "norm_lf0": norm_lf0,
            "lf0": lf0,
            "uv": uv,
            "c_lengths": c_lengths,
        }

    def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468,
              predict_f0=False, vol=None):
        """so-vits NetGSvc.infer (verbatim from so-vits-svc/models.py).

        Args:
            c: [B, ssl_dim, T] content features (transposed).
            f0: [B, T] continuous f0 (Hz).
            uv: [B, T] voicing flag (0/1).
            g: [B, 1] speaker ids (so-vits convention; [B] also accepted).
            noice_scale: prior noise scale.
            seed: RNG seed for the stochastic prior.
            predict_f0: use the f0 decoder head instead of ground f0.
            vol: [B, T] volume (only if vol_embedding).

        Returns:
            (o, f0): wav [B, 1, L] and (possibly predicted) f0 [B, T].
        """
        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if g.dim() == 1:
            g = g.unsqueeze(0)
        g = self.emb_g(g).transpose(1, 2)

        x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)

        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0

        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        if self.use_automatic_f0_prediction and predict_f0:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o, f0
