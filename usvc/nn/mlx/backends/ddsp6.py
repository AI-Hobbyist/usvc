"""MLX 版 ddsp6 backend(mel-domain Rectified Flow + DDSP),镜像 backends/ddsp6/model.py。

与 torch 版差异(MLX 映射手册):输入 NLC(conv 权重加载时由 _convert 转置,
forward 不需要 Transpose 层 —— 保留 _Noop 占位对齐 ckpt 键位);nn.ModuleList→list;
torch.fmod(x,1)→x-mx.trunc(x)(% 是 floor-mod,负数语义不同);torch.sinc→
mx.where(x==0,1,sin(pi x)/(pi x));center=True stft/istft 手写(_ops.*,帧网格+
window envelop 归一化逐位对齐 torch);weight_norm(Unit2Control)在 from_ckpt 合并。
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .._convert import load_torch_ckpt_to_mlx
from .._ops import (
    hann_window,
    norm_spec as _norm_spec,
    denorm_spec as _denorm_spec,
    ode_euler as _ode_euler,
    ode_rk4 as _ode_rk4,
    reflow_loss as _reflow_loss,
    stft_center_complex as _stft_center_complex,
    istft_center_complex as _istft_center_complex,
)

__all__ = ["SwiGLU", "SinusoidalPosEmb", "LYNXNet2", "LYNXNet", "ConformerNaiveEncoder",
           "Unit2Control", "CombSubSuperFast", "RectifiedFlow", "DDSP6Model"]


class SwiGLU(nn.Module):
    def __call__(self, x):
        out, gate = mx.split(x, 2, axis=-1)
        return out * nn.silu(gate)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x):  # [B] -> [B, dim],fp32 与 torch 一致
        half_dim = self.dim // 2
        freqs = mx.exp(mx.arange(half_dim) * -(math.log(10000) / (half_dim - 1)))
        emb = x[:, None].astype(mx.float32) * freqs[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class _Seq(list):
    """普通 list 替代 nn.Sequential:后者把子模块注册到 .layers 下,state_dict
    键 net.layers.0 与 torch ckpt 的 net.0 错位;list 按索引注册保持键位一致。"""

    def __init__(self, *items):
        super().__init__(items)

    def __call__(self, x):
        for m in self: x = m(x)
        return x


class _Noop(nn.Module):
    """占位 torch Transpose((1,2)):NLC 布局无需转置,但保留索引以对齐 ckpt 键位。"""

    def __call__(self, x):
        return x


def _glu_net(dim, inner, k, drop, conv=False, act=None, in_norm=True):
    """depthwise-gated 块:conv=False=6.3 双 SwiGLU 版,True=6.1/6.2 PReLU 版;
    _Noop 即 torch Transpose 占位,位置与 torch Sequential 逐位一致。"""
    nop = _Noop
    if conv:
        net = [nn.LayerNorm(dim) if in_norm else nn.Identity(), nop(),
               nn.Conv1d(dim, inner * 2, 1), SwiGLU(),
               nn.Conv1d(inner, inner, k, padding=k // 2, groups=inner),
               act, nn.Conv1d(inner, dim, 1), nop(), nn.Dropout(drop)]
    else:
        net = [nn.LayerNorm(dim) if in_norm else nn.Identity(), nop(),
               nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim), nop(),
               nn.Linear(dim, inner * 2), SwiGLU(),
               nn.Linear(inner, inner * 2), SwiGLU(),
               nn.Linear(inner, dim), nn.Dropout(drop)]
    return _Seq(*net)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim: int, expansion_factor: float, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        self.net = _glu_net(dim, int(dim * expansion_factor), kernel_size, dropout, in_norm=True)

    def __call__(self, x): return x + self.net(x)


class LYNXNet2(nn.Module):
    """lynxnet2.py 6.3 velocity network: spec [B,1,M,T]/[B,M,T], step [B], cond [B,M,T]。"""

    def __init__(self, in_dims: int, dim_cond: int, n_layers: int = 6,
                 n_chans: int = 512, n_dilates: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)
        self.diffusion_embedding = _Seq(SinusoidalPosEmb(n_chans), nn.Linear(n_chans, n_chans * 4),
                              nn.GELU(), nn.Linear(n_chans * 4, n_chans))
        self.residual_layers = [
            LYNXNet2Block(n_chans, n_dilates, kernel_size=31, dropout=dropout)
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
        self.output_projection.weight = mx.zeros_like(self.output_projection.weight)  # torch nn.init.zeros_

    def __call__(self, spec, diffusion_step, cond):
        t4 = spec.ndim == 4
        x = spec[:, 0] if t4 else spec  # [B,1,M,T] -> [B,M,T]
        x = self.input_projection(x.transpose(0, 2, 1)) + \
            self.conditioner_projection(cond.transpose(0, 2, 1))
        if diffusion_step.ndim == 2:
            diffusion_step = diffusion_step.reshape(-1)
        x = x + self.diffusion_embedding(diffusion_step)[:, None, :]
        for layer in self.residual_layers:
            x = layer(x)
        x = self.output_projection(self.norm(x)).transpose(0, 2, 1)  # [B,in_dims,T]
        return x[:, None] if t4 else x


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
        self.net = _glu_net(dim, inner_dim, kernel_size, dropout, conv=True, act=act)

    def __call__(self, x): return self.net(x)


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

    def __call__(self, x, conditioner, diffusion_step):  # x/cond NLC;step [B,1,dim]
        return self.convmodule(x + self.diffusion_projection(diffusion_step)
                               + self.conditioner_projection(conditioner)) + x


class LYNXNet(nn.Module):
    """lynxnet.py 6.1/6.2 velocity network。"""

    def __init__(self, in_dims: int, dim_cond: int, n_feats: int = 1, n_layers: int = 6,
                 n_chans: int = 512, n_dilates: int = 2, in_norm: bool = False,
                 activation: str = "PReLU", dropout: float = 0.0):
        super().__init__()
        self.input_projection = nn.Conv1d(in_dims * n_feats, n_chans, 1)
        self.diffusion_embedding = _Seq(SinusoidalPosEmb(n_chans), nn.Linear(n_chans, n_chans * 4),
                              nn.GELU(), nn.Linear(n_chans * 4, n_chans))
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
        t4 = spec.ndim == 4
        x = spec[:, 0] if t4 else spec
        x = nn.gelu(self.input_projection(x.transpose(0, 2, 1)))  # NCL -> NLC
        if diffusion_step.ndim == 2:
            diffusion_step = diffusion_step.reshape(-1)
        step = self.diffusion_embedding(diffusion_step)[:, None, :]  # [B,1,n_chans]
        cond_t = cond.transpose(0, 2, 1)
        for layer in self.residual_layers:
            x = layer(x, cond_t, step)
        x = self.output_projection(self.norm(x)).transpose(0, 2, 1)
        return x[:, None] if t4 else x


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
        if conv_model_type != "mode1":
            raise ValueError(f"{conv_model_type} is not a valid conv_model_type")
        if conv_block == "linear":
            self.net = _glu_net(dim, inner_dim, kernel_size, dropout, in_norm=use_norm)
        else:
            act = nn.PReLU(inner_dim)
            self.net = _glu_net(dim, inner_dim, kernel_size, dropout,
                                conv=True, act=act, in_norm=use_norm)

    def __call__(self, x):
        return self.net(x)


class _TransformerEncoderLayer(nn.Module):
    """torch nn.TransformerEncoderLayer(post-LN,norm_first=False);用
    mx.fast.scaled_dot_product_attention 替代 fused MHA。推理(conv_only=True)
    不会构造,保留给 use_attention=True 配置。"""

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward=None,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = _MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward or d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward or d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.self_attn(x, mask)))
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(nn.gelu(self.linear1(x))))))


class _MultiheadAttention(nn.Module):
    """MHA,in_proj_weight 布局与 torch 一致([3d, d] q/k/v 堆叠)。"""

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
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D))


class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, num_heads: int = 8, use_norm: bool = False,
                 conv_only: bool = False, conv_dropout: float = 0.0,
                 atten_dropout: float = 0.1, conv_block: str = "linear"):
        super().__init__()
        self.conformer = ConformerConvModule(
            dim_model, use_norm=use_norm, dropout=conv_dropout, conv_block=conv_block,
        )
        self.attn = None
        if not conv_only:
            self.attn = _TransformerEncoderLayer(
                d_model=dim_model, nhead=num_heads,
                dim_feedforward=dim_model * 4, dropout=atten_dropout,
            )
            self.norm = nn.LayerNorm(dim_model)
        else:
            self.norm = None

    def __call__(self, x, mask=None):
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask)
        return x + self.conformer(x)


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
        for layer in self.encoder_layers: x = layer(x, mask)
        return x


def _split_to_dict(tensor, tensor_splits):
    """Split last dim by sizes → dict(torch split 语义)。"""
    sizes = list(tensor_splits.values())
    idx = []
    acc = 0
    for s in sizes[:-1]:
        acc += s
        idx.append(acc)
    return dict(zip(tensor_splits.keys(), mx.split(tensor, idx, axis=-1)))


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
        self.spk_embed = nn.Embedding(n_spk, dim_model) if n_spk is not None and n_spk > 1 else None
        self.aug_shift_embed = nn.Linear(1, dim_model, bias=False) if use_pitch_aug else None
        self.stack = _Seq(nn.Conv1d(input_channel, 512, 3, 1, 1), nn.PReLU(512),
                          nn.Conv1d(512, dim_model, 3, 1, 1))
        self.stack2 = _Seq(nn.Conv1d(2 * block_size, 512, 3, 1, 1), nn.PReLU(512),
                           nn.Conv1d(512, dim_model, 3, 1, 1))
        self.decoder = ConformerNaiveEncoder(
            num_layers=num_layers, num_heads=8, dim_model=dim_model,
            use_norm=use_norm, conv_only=not use_attention,
            conv_dropout=0, atten_dropout=0.1, conv_block=conv_block,
        )
        self.norm = nn.LayerNorm(dim_model)
        self.dense_out = nn.Linear(dim_model, sum(output_splits.values()))

    def __call__(self, units, source, noise, volume, spk_id=None,
                 spk_mix_dict=None, aug_shift=None):  # 全 NLC [B,T,C]
        exciter = mx.concatenate([source, noise], axis=-1)  # [B,T,2*block]
        x = self.stack(units) + self.stack2(exciter) + self.volume_embed(volume)
        if self.spk_embed is not None:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    x = x + v * self.spk_embed(mx.array([[k]]))[:, 0:1, :]
            elif spk_id is not None:
                x = x + self.spk_embed(spk_id)[:, None, :]
        if self.aug_shift_embed is not None and aug_shift is not None:
            # aug_shift [B]/[B,1]/[B,1,1] 统一 [B,1](torch 语义)
            x = x + self.aug_shift_embed(aug_shift.reshape(-1, 1) / 5)[:, None, :]
        e = self.dense_out(self.norm(self.decoder(x)))
        return _split_to_dict(e, self.output_splits), x


def _sinc(x):  # torch.sinc:sinc(0)=1
    return mx.where(x == 0, mx.ones_like(x), mx.sin(mx.pi * x) / (mx.pi * x))


def _fmod1(x):  # torch.fmod(x,1.0);% 是 floor-mod,负数语义不同
    return x - mx.trunc(x)


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
        split_map = {k: win_length // 2 + 1 for k in
                     ("harmonic_magnitude", "harmonic_phase",
                      "noise_magnitude", "noise_phase")}
        self.unit2ctrl = Unit2Control(
            n_unit, block_size, n_spk, split_map,
            num_layers=num_layers, dim_model=dim_model,
            use_norm=use_norm, use_attention=use_attention,
            use_pitch_aug=use_pitch_aug, conv_block=conv_block,
        )

    def fast_source_gen(self, f0_frames):  # [B,T,1] -> combtooth [B, T*block]
        n = mx.arange(self.block_size)
        s0 = f0_frames / self.sampling_rate
        ds0 = mx.pad(s0[:, 1:, :] - s0[:, :-1, :], [(0, 0), (0, 1), (0, 0)])  # 尾部补 0
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = _fmod1(rad[..., -1:] + 0.5) - 0.5
        rad_acc = _fmod1(mx.cumsum(rad2, axis=1))
        rad = rad + mx.pad(rad_acc[:, :-1, :], [(0, 0), (1, 0), (0, 0)])
        rad = rad - mx.round(rad)
        combtooth = _sinc(rad / (s0 + 1e-5))
        return combtooth.reshape(f0_frames.shape[0], -1)

    def __call__(self, units_frames, f0_frames, volume_frames, spk_id=None,
                 spk_mix_dict=None, aug_shift=None, initial_phase=None,
                 infer=True, **kwargs):
        combtooth = self.fast_source_gen(f0_frames)  # [B, T*block]
        B = combtooth.shape[0]
        combtooth_frames = combtooth.reshape(B, f0_frames.shape[1], self.block_size)
        noise = mx.random.normal(combtooth.shape)
        noise_frames = noise.reshape(B, f0_frames.shape[1], self.block_size)

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


class RectifiedFlow(nn.Module):
    def __init__(self, velocity_fn, out_dims: int = 128,
                 spec_min: float = -12, spec_max: float = 2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max

    def sample_euler(self, x, t, dt, cond):
        return _ode_euler(self.velocity_fn, x, t, dt, cond)

    def sample_rk4(self, x, t, dt, cond):
        return _ode_rk4(self.velocity_fn, x, t, dt, cond)

    def reflow_loss(self, x_1, t, cond, loss_type="l2_lognorm", mask=None):
        return _reflow_loss(self.velocity_fn, x_1, t, cond, loss_type, mask,
                            layout="b1mt")  # [B,1,M,T] 布局见 _ops.reflow_loss

    def __call__(self, condition, gt_spec=None, infer: bool = True,
                 infer_step: int = 10, method: str = "euler",
                 t_start: float = 0.0, use_tqdm: bool = True, mask=None):
        cond = condition.transpose(0, 2, 1)  # [B,T,M] -> [B,M,T]
        b = condition.shape[0]
        t_start = max(t_start, 0.0)

        if not infer:
            x_1 = self.norm_spec(gt_spec).transpose(0, 2, 1)[:, None]  # [B,1,M,T]
            t = mx.clip(t_start + (1.0 - t_start) * mx.random.uniform(shape=(b,)), 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, t, cond=cond, mask=mask)

        shape = (b, 1, self.out_dims, cond.shape[2])  # [B,1,M,T]
        if gt_spec is None:
            x = mx.random.normal(shape)
            t = mx.zeros((b,))
            dt = 1.0 / infer_step
        else:
            norm_spec = self.norm_spec(gt_spec).transpose(0, 2, 1)[:, None]
            x = t_start * norm_spec + (1 - t_start) * mx.random.normal(shape)
            t = mx.full((b,), t_start)
            dt = (1.0 - t_start) / infer_step

        step = {"euler": self.sample_euler, "rk4": self.sample_rk4}.get(method)
        if step is None:
            raise NotImplementedError(method)
        for _ in range(infer_step):
            x, t = step(x, t, dt, cond)

        x = mx.clip(x.squeeze(1).transpose(0, 2, 1), -1.0, 1.0)  # [B,1,M,T]->[B,T,M]
        return self.denorm_spec(x)

    def norm_spec(self, x):
        return _norm_spec(x, self.spec_min, self.spec_max, cast=False)

    def denorm_spec(self, x):
        return _denorm_spec(x, self.spec_min, self.spec_max)


def _default_mel_cfg(out_dims: int, win_length: int):
    return {"n_fft": win_length, "win_length": win_length, "n_mels": out_dims,
            "fmin": 40.0, "fmax": 16000.0, "center": False}


class DDSP6Model(nn.Module):
    """属性名与 torch 版一致(ddsp_model/reflow_model),from_ckpt 直接加载。"""

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
            velocity_cls, conv_block = LYNXNet2, "linear"
        elif velocity_net == "LYNXNet":
            velocity_cls, conv_block = LYNXNet, "conv1d"
        else:
            raise ValueError(f"unknown velocity_net {velocity_net!r} (expected 'LYNXNet' or 'LYNXNet2')")

        self.ddsp_model = CombSubSuperFast(
            sampling_rate, block_size, win_length, n_units, n_spk,
            n_aux_layers if n_aux_layers is not None else 3,
            n_aux_chans if n_aux_chans is not None else 256,
            use_norm, use_attention, use_pitch_aug, conv_block=conv_block,
        )
        self.reflow_model = RectifiedFlow(
            velocity_cls(in_dims=out_dims, dim_cond=out_dims,
                         n_layers=n_layers, n_chans=n_chans),
            out_dims=out_dims,
        )
        self.mel_extractor = mel_extractor

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
                sr=self.sampling_rate, hop=self.block_size,
            )
        return self.mel_extractor

    def _fit(self, f0, volume, t_start):
        if f0.ndim == 2:
            f0 = f0[:, :, None]
        if volume.ndim == 2:
            volume = volume[:, :, None]
        return f0, volume, self.t_start if t_start is None else t_start

    def _ddsp_mel(self, units, f0, volume, spk_id, aug_shift):
        wav, _ = self.ddsp_model(units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
                                 infer=True)
        ext = self._ensure_mel_extractor()
        return mx.stack([ext.extract(wav[i]) for i in range(wav.shape[0])])  # [B,T,M]

    def forward(self, units, f0, volume, spk_id, gt_spec, aug_shift=None,
                frame_len=None, lambda_ddsp=1.0, t_start=None):
        """训练 forward:DDSP 合成 mel + reflow loss → (loss, {ddsp/reflow float})。"""
        f0, volume, t_start = self._fit(f0, volume, t_start)
        ddsp_mel = self._ddsp_mel(units, f0, volume, spk_id, aug_shift)
        ddsp_loss = mx.mean(mx.square(ddsp_mel - gt_spec))

        if t_start < 1.0:
            from ....train.mlx.objective import lens_to_mask
            mask = lens_to_mask(frame_len, length=gt_spec.shape[1]) if frame_len is not None else None
            reflow_loss = self.reflow_model(
                ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False, mask=mask,
            )
        else:
            reflow_loss = 0.0
        total = lambda_ddsp * ddsp_loss + reflow_loss
        return total, {"ddsp_loss": float(ddsp_loss.item()),
                       "reflow_loss": float(reflow_loss.item())}

    def sample(self, units, f0, volume, spk_id=None, aug_shift=None,
               infer_step: int = 10, method: str = "euler",
               t_start: float | None = None, use_tqdm: bool = False):
        """推理:DDSP 合成 → reflow 精修 → [B,T,out_dims] mel。"""
        f0, volume, t_start = self._fit(f0, volume, t_start)
        ddsp_mel = self._ddsp_mel(units, f0, volume, spk_id, aug_shift)
        return self.reflow_model(
            ddsp_mel, gt_spec=ddsp_mel, infer=True,
            infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm,
        )

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, config: dict | None = None,
                  dtype=None, strict: bool = False) -> "DDSP6Model":
        """从 torch ckpt 加载(ddsp_model.* + reflow_model.* 前缀);config None 时
        从权重 shape 反推(见 _infer_config)。"""
        from .._convert import load_torch_ckpt_to_mlx, strip_prefix

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype)
        if any("ddsp_model.unit2ctrl.residual_layers." in k for k in sd):
            raise NotImplementedError("DDSP-SVC 6.5 checkpoints currently require the PyTorch backend")
        obj = cls(**config) if config is not None else cls(**cls._infer_config(sd))
        if any(k.startswith("model.") for k in sd):
            sd = strip_prefix(sd, "model.")  # 训练 ckpt 的 lightning 前缀
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731
        stack0 = shape("ddsp_model.unit2ctrl.stack.0.weight")  # mlx (O,K,C)
        if stack0 is None:
            raise ValueError("cannot infer ddsp6 config: missing stack.0.weight")
        n_unit = stack0[2]
        stack2_0 = shape("ddsp_model.unit2ctrl.stack2.0.weight")
        block_size = stack2_0[2] // 2 if stack2_0 else 512
        dense_out = shape("ddsp_model.unit2ctrl.dense_out.weight")
        if dense_out is None:
            raise ValueError("cannot infer ddsp6 config: missing dense_out.weight")
        n_out, dim_model = dense_out[0], dense_out[1]
        win_length = 2 * (n_out // 4 - 1)
        n_aux_layers = len({
            int(k.split("encoder_layers.")[1].split(".")[0])
            for k in sd
            if k.startswith("ddsp_model.unit2ctrl.decoder.encoder_layers.")
        })
        in_proj = shape("reflow_model.velocity_fn.input_projection.weight")
        if in_proj is None:
            raise ValueError("cannot infer ddsp6 config: missing velocity_fn.input_projection.weight")
        if len(in_proj) == 2:  # LYNXNet2 Linear (n_chans, in_dims)
            n_chans, out_dims = in_proj[0], in_proj[1]
            velocity_net = "LYNXNet2"
        else:  # LYNXNet Conv1d mlx (n_chans, K, in_dims)
            n_chans, out_dims = in_proj[0], in_proj[2]
            velocity_net = "LYNXNet"
        n_layers = len({
            int(k.split("residual_layers.")[1].split(".")[0])
            for k in sd
            if k.startswith("reflow_model.velocity_fn.residual_layers.")
        })
        n_spk = 2 if "ddsp_model.unit2ctrl.spk_embed.weight" in sd else 1
        use_pitch_aug = "ddsp_model.unit2ctrl.aug_shift_embed.weight" in sd
        return dict(
            n_units=n_unit, n_spk=n_spk, win_length=win_length,
            sampling_rate=44100, block_size=block_size,
            use_norm=True, use_attention=False, use_pitch_aug=use_pitch_aug,
            out_dims=out_dims, n_aux_layers=n_aux_layers, n_aux_chans=dim_model,
            n_layers=n_layers, n_chans=n_chans, velocity_net=velocity_net,
        )
