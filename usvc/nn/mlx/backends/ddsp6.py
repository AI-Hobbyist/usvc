"""MLX 版 ddsp6 backend(mel-domain Rectified Flow + DDSP)。

镜像 ``usvc/backends/ddsp6/model.py`` 的推理最小集:
  - LYNXNet2 / LYNXNet:reflow velocity network(6.3 / 6.1-6.2)
  - RectifiedFlow:sample_euler / sample_rk4 ODE 采样
  - Unit2Control:units/f0/volume → DDSP 频谱 filter 参数
  - CombSubSuperFast:DDSP 频谱路径(harmonic+noise 激励经复 STFT 滤波 → ISTFT)
  - ConformerNaiveEncoder:Unit2Control 条件 encoder(6.3 linear blocks,SDPA)
  - DDSP6Model:sample(units, f0, volume) → [B, T, out_dims] mel

与 torch 版的差异(见 MLX 映射手册 / MLX_PITFALLS.md):
  - MLX Conv1d 输入 NLC [B, T, C],torch 是 NCL —— 权重加载时由 _convert
    转置 (O,C,K)→(O,K,C),forward 里不再需要 Transpose 层(用 no-op 占位
    保持 state_dict 键位对齐)
  - ``nn.ModuleList`` → 普通 list
  - ``nn.GELU`` 用类(approx='none'),函数版 ``nn.gelu`` 不接受 approx
  - ``torch.fmod(x, 1.0)`` → ``x - mx.trunc(x)``(% 是 floor-mod,负数语义不同)
  - ``torch.sinc`` → ``mx.where(x==0, 1, sin(pi*x)/(pi*x))``
  - ``torch.stft/istft(center=True, return_complex)`` → 本模块手写
    ``_stft_center_complex`` / ``_istft_center_complex``(mx.fft.rfft/irfft,
    帧网格 + window envelop 归一化逐位对齐 torch)
  - weight_norm(Unit2Control stack/stack2/dense_out)在 from_ckpt 时由
    ``_convert.load_torch_ckpt_to_mlx`` 合并,运行时是普通 Conv/Linear
  - PReLU 用 ``nn.PReLU(num_parameters=inner)``
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .._convert import load_torch_ckpt_to_mlx
from .._ops import hann_window

__all__ = [
    "SwiGLU",
    "SinusoidalPosEmb",
    "LYNXNet2",
    "LYNXNet",
    "ConformerNaiveEncoder",
    "Unit2Control",
    "CombSubSuperFast",
    "RectifiedFlow",
    "DDSP6Model",
]


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU:out, gate = split(x, 2, -1);out * silu(gate)。"""

    def __call__(self, x):
        out, gate = mx.split(x, 2, axis=-1)
        return out * nn.silu(gate)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        # x: [B] -> [B, dim] (float32 计算,与 torch 一致)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        freqs = mx.exp(mx.arange(half_dim) * -emb)  # [half_dim]
        emb = x[:, None].astype(mx.float32) * freqs[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)



class _Seq(list):
    """可调用的普通 list(替代 mlx nn.Sequential)。

    MLX ``nn.Sequential`` 把子模块注册在 ``.layers`` 下,state_dict 键变成
    ``net.layers.0.weight``,与 torch ckpt 的 ``net.0.weight`` 错位。用普通
    list(MLX nn.Module 自动按索引注册子模块)保持键位一致。
    """

    def __init__(self, *items):
        super().__init__(items)

    def __call__(self, x):
        for m in self:
            x = m(x)
        return x


class _Noop(nn.Module):
    """占位模块:对应 torch 的 Transpose((1,2))。

    MLX Conv1d 是 NLC,无需转置;但 torch 的 nn.Sequential 里 Transpose
    占用了 net.1 / net.3 等索引,去掉会让 state_dict 键位错位,from_ckpt
    无法直接 load。保留为 no-op 以对齐键位。
    """

    def __call__(self, x):
        return x


# ---------------------------------------------------------------------------
# LYNXNet2 (reflow/lynxnet2.py, DDSP-SVC-6.3 velocity network)
# ---------------------------------------------------------------------------

class LYNXNet2Block(nn.Module):
    def __init__(self, dim: int, expansion_factor: float, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = _Seq(
            nn.LayerNorm(dim),
            _Noop(),  # torch Transpose((1,2))
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            _Noop(),  # torch Transpose((1,2))
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def __call__(self, x):
        return x + self.net(x)


class LYNXNet2(nn.Module):
    """LYNXNet2(Linear Gated Depthwise Separable Convolution Network V2,6.3)。

    forward(spec [B,1,M,T] 或 [B,M,T], diffusion_step [B], cond [B,M,T])
    → [B,1,M,T] 或 [B,M,T](use_4_dim 保持输入维度)。
    """

    def __init__(self, in_dims: int, dim_cond: int, n_layers: int = 6,
                 n_chans: int = 512, n_dilates: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)
        self.diffusion_embedding = _Seq(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans),
        )
        self.residual_layers = [
            LYNXNet2Block(n_chans, n_dilates, kernel_size=31, dropout=dropout)
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
        # torch nn.init.zeros_(output_projection.weight)
        self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond):
        x = spec
        use_4_dim = False
        if x.ndim == 4:
            x = x[:, 0]  # [B, 1, M, T] -> [B, M, T]
            use_4_dim = True
        x = self.input_projection(x.transpose(0, 2, 1))  # [B, T, M] -> [B, T, n_chans]
        x = x + self.conditioner_projection(cond.transpose(0, 2, 1))  # [B, T, dim_cond]
        # diffusion_step [B] (或 [B,1]) -> [B, 1, n_chans]
        if diffusion_step.ndim == 2:
            diffusion_step = diffusion_step.reshape(-1)
        x = x + self.diffusion_embedding(diffusion_step)[:, None, :]
        for layer in self.residual_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_projection(x).transpose(0, 2, 1)  # [B, T, in_dims] -> [B, in_dims, T]
        return x[:, None] if use_4_dim else x


# ---------------------------------------------------------------------------
# LYNXNet (reflow/lynxnet.py, DDSP-SVC-6.1/6.2 velocity network)
# ---------------------------------------------------------------------------

class LYNXConvModule(nn.Module):
    @staticmethod
    def calc_same_padding(kernel_size):
        pad = kernel_size // 2
        return pad, pad - (kernel_size + 1) % 2

    def __init__(self, dim: int, expansion_factor: float, kernel_size: int = 31,
                 in_norm: bool = False, activation: str = "PReLU", dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        if activation == "PReLU":
            act = nn.PReLU(inner_dim)
        elif activation == "SiLU":
            act = nn.SiLU()
        elif activation == "ReLU":
            act = nn.ReLU()
        else:
            raise ValueError(f"{activation} is not a valid activation")
        padding = self.calc_same_padding(kernel_size)
        self.net = _Seq(
            nn.LayerNorm(dim),
            _Noop(),  # torch Transpose((1,2))
            nn.Conv1d(dim, inner_dim * 2, 1),
            SwiGLU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=padding[0], groups=inner_dim),
            act,
            nn.Conv1d(inner_dim, dim, 1),
            _Noop(),  # torch Transpose((1,2))
            nn.Dropout(dropout),
        )

    def __call__(self, x):
        return self.net(x)


class LYNXNetResidualLayer(nn.Module):
    def __init__(self, dim_cond: int, dim: int, expansion_factor: float,
                 kernel_size: int = 31, in_norm: bool = False,
                 activation: str = "PReLU", dropout: float = 0.0):
        super().__init__()
        self.diffusion_projection = nn.Conv1d(dim, dim, 1)
        self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
        self.convmodule = LYNXConvModule(
            dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size,
            in_norm=in_norm, activation=activation, dropout=dropout,
        )

    def __call__(self, x, conditioner, diffusion_step):
        # x / conditioner: NLC [B, T, dim];diffusion_step: [B, 1, dim]
        res_x = x
        x = x + self.diffusion_projection(diffusion_step) + self.conditioner_projection(conditioner)
        x = self.convmodule(x)
        return x + res_x


class LYNXNet(nn.Module):
    """LYNXNet(Linear Gated Depthwise Separable Convolution Network,6.1/6.2)。"""

    def __init__(self, in_dims: int, dim_cond: int, n_feats: int = 1, n_layers: int = 6,
                 n_chans: int = 512, n_dilates: int = 2, in_norm: bool = False,
                 activation: str = "PReLU", dropout: float = 0.0):
        super().__init__()
        self.input_projection = nn.Conv1d(in_dims * n_feats, n_chans, 1)
        self.diffusion_embedding = _Seq(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans),
        )
        self.residual_layers = [
            LYNXNetResidualLayer(
                dim_cond=dim_cond, dim=n_chans, expansion_factor=n_dilates,
                kernel_size=31, in_norm=in_norm, activation=activation, dropout=dropout,
            )
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Conv1d(n_chans, in_dims * n_feats, 1)
        self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond):
        x = spec
        use_4_dim = False
        if x.ndim == 4:
            x = x[:, 0]
            use_4_dim = True
        # NCL -> NLC(torch 的 input_projection 吃 NCL,mlx 吃 NLC)
        x = nn.gelu(self.input_projection(x.transpose(0, 2, 1)))  # [B, T, n_chans]
        if diffusion_step.ndim == 2:
            diffusion_step = diffusion_step.reshape(-1)
        step = self.diffusion_embedding(diffusion_step)[:, None, :]  # [B, 1, n_chans]
        cond_t = cond.transpose(0, 2, 1)  # [B, T, dim_cond]
        for layer in self.residual_layers:
            x = layer(x, cond_t, step)
        x = self.norm(x)
        x = self.output_projection(x).transpose(0, 2, 1)  # [B, in_dims, T]
        return x[:, None] if use_4_dim else x


# ---------------------------------------------------------------------------
# ConformerNaiveEncoder (ddsp/model_conformer_naive.py;6.3 linear blocks)
# ---------------------------------------------------------------------------

class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, expansion_factor=None, kernel_size: int = 31,
                 dropout: float = 0.0, use_norm: bool = False,
                 conv_model_type: str = "mode1", conv_block: str = "linear"):
        super().__init__()
        if conv_block not in ("linear", "conv1d"):
            raise ValueError(f"{conv_block} is not a valid conv_block")
        if expansion_factor is None:
            expansion_factor = 2 if conv_block == "conv1d" else 1
        inner_dim = dim * expansion_factor

        norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        if conv_model_type == "mode1":
            if conv_block == "linear":
                # 6.3: depthwise Conv1d + Linear/SwiGLU 块
                self.net = _Seq(
                    norm,
                    _Noop(),  # Transpose
                    nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
                    _Noop(),  # Transpose
                    nn.Linear(dim, inner_dim * 2),
                    SwiGLU(),
                    nn.Linear(inner_dim, inner_dim * 2),
                    SwiGLU(),
                    nn.Linear(inner_dim, dim),
                    nn.Dropout(dropout),
                )
            else:
                # 6.1/6.2: depthwise Conv1d + PReLU 块
                padding = kernel_size // 2
                self.net = _Seq(
                    norm,
                    _Noop(),  # Transpose
                    nn.Conv1d(dim, inner_dim * 2, 1),
                    SwiGLU(),
                    nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=padding, groups=inner_dim),
                    nn.PReLU(inner_dim),
                    nn.Conv1d(inner_dim, dim, 1),
                    _Noop(),  # Transpose
                    nn.Dropout(dropout),
                )
        else:
            raise ValueError(f"{conv_model_type} is not a valid conv_model_type")

    def __call__(self, x):
        return self.net(x)


class _TransformerEncoderLayer(nn.Module):
    """torch nn.TransformerEncoderLayer(post-LN, norm_first=False) 的 MLX 版。

    用 ``mx.fast.scaled_dot_product_attention`` 替代 fused MHA。
    推理(conv_only=True)时不会被构造;保留给 use_attention=True 配置。
    """

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward=None,
                 dropout: float = 0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4
        self.self_attn = _MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        # post-LN(norm_first=False)
        x = self.norm1(x + self.dropout1(self.self_attn(x, mask)))
        x = self.norm2(x + self.dropout2(self.linear2(self.dropout(nn.gelu(self.linear1(x))))))
        return x


class _MultiheadAttention(nn.Module):
    """MHA(in_proj_weight 布局与 torch 一致:[3d, d] q/k/v 堆叠)。"""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.in_proj_weight = mx.zeros((3 * d_model, d_model))
        self.in_proj_bias = mx.zeros((3 * d_model,))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def __call__(self, x, mask=None):
        B, T, D = x.shape
        h, hd = self.num_heads, self.head_dim
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias  # [B, T, 3D]
        q, k, v = mx.split(qkv, [D, 2 * D], axis=-1)
        q = q.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        sdpa_mask = None
        if mask is not None:
            sdpa_mask = mask[:, None, None, :]  # [B, 1, 1, T]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=sdpa_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)


class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, num_heads: int = 8, use_norm: bool = False,
                 conv_only: bool = False, conv_dropout: float = 0.0,
                 atten_dropout: float = 0.1, conv_block: str = "linear"):
        super().__init__()
        self.conformer = ConformerConvModule(
            dim_model, use_norm=use_norm, dropout=conv_dropout, conv_block=conv_block,
        )
        if not conv_only:
            self.attn = _TransformerEncoderLayer(
                d_model=dim_model, nhead=num_heads,
                dim_feedforward=dim_model * 4, dropout=atten_dropout,
            )
            self.norm = nn.LayerNorm(dim_model)
        else:
            self.attn = None

    def __call__(self, x, mask=None):
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask)
        x = x + self.conformer(x)
        return x


class ConformerNaiveEncoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, dim_model: int,
                 use_norm: bool = False, conv_only: bool = False,
                 conv_dropout: float = 0.0, atten_dropout: float = 0.0,
                 conv_block: str = "linear"):
        super().__init__()
        self.encoder_layers = [
            CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only,
                            conv_dropout, atten_dropout, conv_block=conv_block)
            for _ in range(num_layers)
        ]

    def __call__(self, x, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


# ---------------------------------------------------------------------------
# Unit2Control (ddsp/unit2control.py)
# ---------------------------------------------------------------------------

def _split_to_dict(tensor, tensor_splits):
    """Split last dim by sizes -> dict(torch.split sizes 语义)。"""
    sizes = list(tensor_splits.values())
    idx = []
    acc = 0
    for s in sizes[:-1]:
        acc += s
        idx.append(acc)
    parts = mx.split(tensor, idx, axis=-1)
    return dict(zip(tensor_splits.keys(), parts))


class Unit2Control(nn.Module):
    def __init__(self, input_channel: int, block_size: int, n_spk,
                 output_splits: dict, num_layers: int = 3, dim_model: int = 256,
                 use_norm: bool = False, use_attention: bool = False,
                 use_pitch_aug: bool = False, conv_block: str = "linear"):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, dim_model)
        self.phase_embed = nn.Linear(1, dim_model)
        self.volume_embed = nn.Linear(1, dim_model)
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, dim_model)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, dim_model, bias=False)
        else:
            self.aug_shift_embed = None
        self.stack = _Seq(
            nn.Conv1d(input_channel, 512, 3, 1, 1),
            nn.PReLU(512),
            nn.Conv1d(512, dim_model, 3, 1, 1),
        )
        self.stack2 = _Seq(
            nn.Conv1d(2 * block_size, 512, 3, 1, 1),
            nn.PReLU(512),
            nn.Conv1d(512, dim_model, 3, 1, 1),
        )
        self.decoder = ConformerNaiveEncoder(
            num_layers=num_layers, num_heads=8, dim_model=dim_model,
            use_norm=use_norm, conv_only=not use_attention,
            conv_dropout=0, atten_dropout=0.1, conv_block=conv_block,
        )
        self.norm = nn.LayerNorm(dim_model)
        self.n_out = sum(output_splits.values())
        self.dense_out = nn.Linear(dim_model, self.n_out)

    def __call__(self, units, source, noise, volume, spk_id=None,
                 spk_mix_dict=None, aug_shift=None):
        # 全部 NLC [B, T, C];torch 的 transpose(1,2) 在 mlx 不需要
        exciter = mx.concatenate([source, noise], axis=-1)  # [B, T, 2*block]
        x = self.stack(units) + self.stack2(exciter)        # [B, T, dim_model]
        x = x + self.volume_embed(volume)                    # volume [B, T, 1]
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    x = x + v * self.spk_embed(mx.array([[k]]))[:, 0:1, :]
            elif spk_id is not None:
                x = x + self.spk_embed(spk_id)[:, None, :]
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)[:, None, :]
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = _split_to_dict(e, self.output_splits)
        return controls, x


# ---------------------------------------------------------------------------
# DDSP 频谱路径:复杂 STFT / ISTFT(center=True, window envelop 归一化)
# ---------------------------------------------------------------------------

def _pad_reflect_2d(x, left: int, right: int):
    """F.pad(x, (left, right), mode='reflect') 等价(沿最后一维,批量)。"""
    if left == 0 and right == 0:
        return x
    parts = [x]
    if left > 0:
        parts.insert(0, x[:, left:0:-1])
    if right > 0:
        parts.append(x[:, -2:-right - 2:-1])
    return mx.concatenate(parts, axis=-1)


def _stft_center_complex(wav, n_fft: int, hop: int, window, pad_mode: str = "reflect"):
    """torch.stft(center=True, return_complex=True) 等价。

    wav [B, L] -> [B, n_fft//2+1, n_frames],n_frames = L//hop + 1
    (center=True 两端各补 n_fft//2)。
    """
    B, L = wav.shape
    pad = n_fft // 2
    if pad_mode == "reflect" and pad < L:
        padded = _pad_reflect_2d(wav, pad, pad)
    else:
        padded = mx.pad(wav, [(0, 0), (pad, pad)])
    n_frames = (padded.shape[1] - n_fft) // hop + 1
    idx = mx.arange(n_frames)[:, None] * hop + mx.arange(n_fft)[None, :]
    frames = mx.take(padded, idx, axis=1)  # [B, n_frames, n_fft]
    frames = frames * window
    spec = mx.fft.rfft(frames, n=n_fft, axis=-1)  # [B, n_frames, F]
    return spec.transpose(0, 2, 1)  # [B, F, n_frames]


def _istft_center_complex(spec, n_fft: int, hop: int, window):
    """torch.istft(center=True) 等价(复频谱 -> 时域,window envelop 归一化)。

    spec [B, n_fft//2+1, n_frames] -> [B, hop*(n_frames-1)]。
    """
    B, F, n_frames = spec.shape
    frames = mx.fft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1)  # [B, n_frames, n_fft]
    frames = frames * window
    expected = n_fft + hop * (n_frames - 1)
    y = mx.zeros((B, expected))
    win_env = mx.zeros((B, expected))
    w2 = window * window
    for i in range(n_frames):
        sl = slice(i * hop, i * hop + n_fft)
        y = y.at[:, sl].add(frames[:, i])
        win_env = win_env.at[:, sl].add(w2)
    start = n_fft // 2
    end = expected - n_fft // 2
    return y[:, start:end] / win_env[:, start:end]


def _sinc(x):
    """torch.sinc:sin(pi*x)/(pi*x),sinc(0)=1。"""
    return mx.where(x == 0, mx.ones_like(x), mx.sin(mx.pi * x) / (mx.pi * x))


def _fmod1(x):
    """torch.fmod(x, 1.0) 等价(x - trunc(x);% 是 floor-mod,负数不同)。"""
    return x - mx.trunc(x)


# ---------------------------------------------------------------------------
# CombSubSuperFast (ddsp/vocoder.py) —— DDSP 频谱路径
# ---------------------------------------------------------------------------

class CombSubSuperFast(nn.Module):
    def __init__(self, sampling_rate: int, block_size: int, win_length: int,
                 n_unit: int = 256, n_spk=1, num_layers: int = 3,
                 dim_model: int = 256, use_norm: bool = False,
                 use_attention: bool = False, use_pitch_aug: bool = False,
                 conv_block: str = "linear"):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.win_length = win_length
        self.window = hann_window(win_length)
        split_map = {
            "harmonic_magnitude": win_length // 2 + 1,
            "harmonic_phase": win_length // 2 + 1,
            "noise_magnitude": win_length // 2 + 1,
            "noise_phase": win_length // 2 + 1,
        }
        self.unit2ctrl = Unit2Control(
            n_unit, block_size, n_spk, split_map,
            num_layers=num_layers, dim_model=dim_model,
            use_norm=use_norm, use_attention=use_attention,
            use_pitch_aug=use_pitch_aug, conv_block=conv_block,
        )

    def fast_source_gen(self, f0_frames):
        # f0_frames [B, T, 1] -> combtooth [B, T*block_size]
        n = mx.arange(self.block_size)
        s0 = f0_frames / self.sampling_rate  # [B, T, 1]
        ds0 = s0[:, 1:, :] - s0[:, :-1, :]
        ds0 = mx.pad(ds0, [(0, 0), (0, 1), (0, 0)])  # F.pad(..., (0,0,0,1)) 尾部补 0
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size  # [B, T, block]
        s0 = s0 + ds0 * n / self.block_size
        rad2 = _fmod1(rad[..., -1:] + 0.5) - 0.5  # [B, T, 1]
        rad_acc = _fmod1(mx.cumsum(rad2, axis=1))  # [B, T, 1]
        rad = rad + mx.pad(rad_acc[:, :-1, :], [(0, 0), (1, 0), (0, 0)])
        rad = rad - mx.round(rad)
        combtooth = _sinc(rad / (s0 + 1e-5))
        return combtooth.reshape(f0_frames.shape[0], -1)

    def __call__(self, units_frames, f0_frames, volume_frames, spk_id=None,
                 spk_mix_dict=None, aug_shift=None, initial_phase=None,
                 infer=True, **kwargs):
        # combtooth 激励 + 噪声激励
        combtooth = self.fast_source_gen(f0_frames)  # [B, T*block]
        B = combtooth.shape[0]
        combtooth_frames = combtooth.reshape(B, f0_frames.shape[1], self.block_size)
        noise = mx.random.normal(combtooth.shape)
        noise_frames = noise.reshape(B, f0_frames.shape[1], self.block_size)

        # 参数预测
        ctrls, hidden = self.unit2ctrl(
            units_frames, combtooth_frames, noise_frames, volume_frames,
            spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift,
        )

        src_filter = mx.exp(ctrls["harmonic_magnitude"] + 1j * math.pi * ctrls["harmonic_phase"])
        src_filter = mx.concatenate([src_filter, src_filter[:, -1:, :]], axis=1)
        noise_filter = mx.exp(ctrls["noise_magnitude"] + 1j * math.pi * ctrls["noise_phase"]) / 128
        noise_filter = mx.concatenate([noise_filter, noise_filter[:, -1:, :]], axis=1)

        pad_mode = "reflect" if combtooth.shape[-1] > self.win_length // 2 else "constant"
        combtooth_stft = _stft_center_complex(
            combtooth, self.win_length, self.block_size, self.window, pad_mode)
        noise_stft = _stft_center_complex(
            noise, self.win_length, self.block_size, self.window, pad_mode)

        signal_stft = combtooth_stft * src_filter.transpose(0, 2, 1) \
            + noise_stft * noise_filter.transpose(0, 2, 1)
        signal = _istft_center_complex(signal_stft, self.win_length, self.block_size, self.window)
        return signal, hidden


# ---------------------------------------------------------------------------
# RectifiedFlow (reflow/reflow.py) —— 推理采样
# ---------------------------------------------------------------------------

class RectifiedFlow(nn.Module):
    def __init__(self, velocity_fn, out_dims: int = 128,
                 spec_min: float = -12, spec_max: float = 2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max

    def sample_euler(self, x, t, dt, cond):
        x = x + self.velocity_fn(x, 1000 * t, cond) * dt
        t = t + dt
        return x, t

    def sample_rk4(self, x, t, dt, cond):
        k1 = self.velocity_fn(x, 1000 * t, cond)
        k2 = self.velocity_fn(x + 0.5 * k1 * dt, 1000 * (t + 0.5 * dt), cond)
        k3 = self.velocity_fn(x + 0.5 * k2 * dt, 1000 * (t + 0.5 * dt), cond)
        k4 = self.velocity_fn(x + k3 * dt, 1000 * (t + dt), cond)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        t = t + dt
        return x, t

    def __call__(self, condition, gt_spec=None, infer: bool = True,
                 infer_step: int = 10, method: str = "euler",
                 t_start: float = 0.0, use_tqdm: bool = True, mask=None):
        if not infer:
            raise NotImplementedError("mlx RectifiedFlow 仅支持 infer=True(训练在 torch 侧)")
        cond = condition.transpose(0, 2, 1)  # [B, T, M] -> [B, M, T]
        b = condition.shape[0]
        if t_start < 0.0:
            t_start = 0.0
        shape = (b, 1, self.out_dims, cond.shape[2])  # [B, 1, M, T]

        if gt_spec is None:
            x = mx.random.normal(shape)
            t = mx.zeros((b,))
            dt = 1.0 / infer_step
        else:
            norm_spec = self.norm_spec(gt_spec).transpose(0, 2, 1)[:, None, :, :]
            x = t_start * norm_spec + (1 - t_start) * mx.random.normal(shape)
            t = mx.full((b,), t_start)
            dt = (1.0 - t_start) / infer_step

        for _ in range(infer_step):
            if method == "euler":
                x, t = self.sample_euler(x, t, dt, cond)
            elif method == "rk4":
                x, t = self.sample_rk4(x, t, dt, cond)
            else:
                raise NotImplementedError(method)

        x = x.squeeze(1).transpose(0, 2, 1)  # [B, 1, M, T] -> [B, T, M]
        x = mx.clip(x, -1.0, 1.0)
        return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# ---------------------------------------------------------------------------
# DDSP6Model —— usvc-facing 组合模型(推理 sample)
# ---------------------------------------------------------------------------

def _default_mel_cfg(out_dims: int, win_length: int):
    return {
        "n_fft": win_length,
        "win_length": win_length,
        "n_mels": out_dims,
        "fmin": 40.0,
        "fmax": 16000.0,
        "center": False,
    }


class DDSP6Model(nn.Module):
    """MLX 版 DDSP6Model(sample 接口,推理)。

    属性名与 torch 版一致(ddsp_model / reflow_model),from_ckpt 直接加载。
    """

    def __init__(self, n_units: int = 768, n_spk=1, win_length: int = 2048,
                 sampling_rate: int = 44100, block_size: int = 512,
                 use_norm: bool = True, use_attention: bool = False,
                 use_pitch_aug: bool = True, out_dims: int = 128,
                 n_aux_layers: int = 6, n_aux_chans: int = 512,
                 n_layers: int = 6, n_chans: int = 1024, t_start: float = 0.0,
                 velocity_net: str = "LYNXNet2", mel_extractor=None):
        super().__init__()
        self.t_start = t_start
        self.out_dims = out_dims
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.win_length = win_length
        if velocity_net == "LYNXNet2":
            velocity_cls = LYNXNet2
            conv_block = "linear"
        elif velocity_net == "LYNXNet":
            velocity_cls = LYNXNet
            conv_block = "conv1d"
        else:
            raise ValueError(f"unknown velocity_net {velocity_net!r} (expected 'LYNXNet' or 'LYNXNet2')")

        self.ddsp_model = CombSubSuperFast(
            sampling_rate, block_size, win_length, n_units, n_spk,
            n_aux_layers if n_aux_layers is not None else 3,
            n_aux_chans if n_aux_chans is not None else 256,
            use_norm, use_attention, use_pitch_aug,
            conv_block=conv_block,
        )
        self.reflow_model = RectifiedFlow(
            velocity_cls(in_dims=out_dims, dim_cond=out_dims,
                         n_layers=n_layers, n_chans=n_chans),
            out_dims=out_dims,
        )
        self.mel_extractor = mel_extractor

    # -- mel 提取 ----------------------------------------------------------

    def _ensure_mel_extractor(self):
        if self.mel_extractor is None:
            from ..mel import MelExtractor
            from .._common import FeatureConfigLike

            class _FC(FeatureConfigLike):
                kind = "mel"

                def __init__(self, params):
                    self.params = params

            self.mel_extractor = MelExtractor(
                _FC(_default_mel_cfg(self.out_dims, self.win_length)),
                sr=self.sampling_rate,
                hop=self.block_size,
            )
        return self.mel_extractor

    # -- 推理 ---------------------------------------------------------------

    def sample(self, units, f0, volume, spk_id=None, aug_shift=None,
               infer_step: int = 10, method: str = "euler",
               t_start: float | None = None, use_tqdm: bool = False):
        """Inference:DDSP 合成 -> reflow 精修。

        units [B, T, n_unit];f0 / volume [B, T] 或 [B, T, 1]。
        返回 denormalized mel [B, T, out_dims]。
        """
        if f0.ndim == 2:
            f0 = f0[:, :, None]
        if volume.ndim == 2:
            volume = volume[:, :, None]
        if t_start is None:
            t_start = self.t_start

        ddsp_wav, _hidden = self.ddsp_model(
            units, f0, volume, spk_id=spk_id, aug_shift=aug_shift, infer=True,
        )
        mel_extractor = self._ensure_mel_extractor()
        # mlx MelExtractor.extract 吃 1D,逐行提取
        ddsp_mel = mx.stack(
            [mel_extractor.extract(ddsp_wav[i]) for i in range(ddsp_wav.shape[0])]
        )  # [B, T, M]
        mel = self.reflow_model(
            ddsp_mel, gt_spec=ddsp_mel, infer=True,
            infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm,
        )
        return mel

    # -- ckpt 加载 ----------------------------------------------------------

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, config: dict | None = None,
                  dtype=None, strict: bool = False) -> "DDSP6Model":
        """从 torch ckpt 加载(ddsp_model.* + reflow_model.* 前缀)。

        config 提供构造参数;None 时从权重 shape 反推(见 _infer_config)。
        """
        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype)
        cfg = config if config is not None else cls._infer_config(sd)
        obj = cls(**cfg)
        # 剥 lightning model. 前缀(训练 ckpt 保存为 model.ddsp_model.*)
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        """从 mlx state_dict shape 反推 DDSP6Model 构造参数。"""
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731
        # ---- ddsp_model ----
        stack0 = shape("ddsp_model.unit2ctrl.stack.0.weight")  # mlx (O,K,C)
        if stack0 is None:
            raise ValueError("cannot infer ddsp6 config: missing stack.0.weight")
        n_unit = stack0[2]
        stack2_0 = shape("ddsp_model.unit2ctrl.stack2.0.weight")
        block_size = stack2_0[2] // 2 if stack2_0 else 512
        dense_out = shape("ddsp_model.unit2ctrl.dense_out.weight")  # (n_out, dim)
        if dense_out is None:
            raise ValueError("cannot infer ddsp6 config: missing dense_out.weight")
        n_out, dim_model = dense_out[0], dense_out[1]
        win_length = 2 * (n_out // 4 - 1)
        n_aux_layers = sum(
            1 for k in sd if k.startswith("ddsp_model.unit2ctrl.decoder.encoder_layers.")
        )
        # ---- reflow_model ----
        in_proj = shape("reflow_model.velocity_fn.input_projection.weight")
        if in_proj is None:
            raise ValueError("cannot infer ddsp6 config: missing velocity_fn.input_projection.weight")
        if in_proj.ndim == 2:  # LYNXNet2 Linear (n_chans, in_dims)
            n_chans, out_dims = in_proj[0], in_proj[1]
            velocity_net = "LYNXNet2"
        else:  # LYNXNet Conv1d (n_chans, in_dims, 1)
            n_chans, out_dims = in_proj[0], in_proj[1]
            velocity_net = "LYNXNet"
        n_layers = sum(
            1 for k in sd if k.startswith("reflow_model.velocity_fn.residual_layers.")
        )
        n_spk = 2 if "ddsp_model.unit2ctrl.spk_embed.weight" in sd else 1
        use_pitch_aug = "ddsp_model.unit2ctrl.aug_shift_embed.weight" in sd
        return dict(
            n_units=n_unit, n_spk=n_spk, win_length=win_length,
            sampling_rate=44100, block_size=block_size,
            use_norm=True, use_attention=False, use_pitch_aug=use_pitch_aug,
            out_dims=out_dims, n_aux_layers=n_aux_layers, n_aux_chans=dim_model,
            n_layers=n_layers, n_chans=n_chans, velocity_net=velocity_net,
        )
