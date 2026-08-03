"""Reflow-VAE-SVC backend models (Unit2Wav_VAE + Bi_RectifiedFlow).

Verbatim port from /root/ReFlow-VAE-SVC:
  - reflow/reflow.py          -> Bi_RectifiedFlow
  - reflow/naive_v2_diff.py   -> NaiveV2Diff, NaiveV2DiffLayer, DiffusionEmbedding
  - reflow/model_conformer_naive.py -> ConformerConvModule, Transpose, calc_same_padding
  - reflow/wavenet.py         -> WaveNet, ResidualBlock, SinusoidalPosEmb, Conv1d, Mish
  - reflow/vocoder.py         -> Unit2Wav_VAE

Class names / attribute names / constructor signatures are preserved so that
original-project checkpoints load directly (state_dict key compatibility).

Deviations (required by usvc constraints):
  - torch_musa import branches removed (CUDA/torch path only).
  - transformers RoFormerEncoder import is lazy (usvc forbids transformers;
    only needed when transformer_use=True, which the shipped configs don't use).
"""
from __future__ import annotations

import math
import random
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# --- Lazy transformers import (default path never needs it) ---
try:
    from transformers.models.roformer.modeling_roformer import (
        RoFormerConfig,
        RoFormerEncoder,
    )

    _HAS_ROFORMER = True
except ImportError:  # pragma: no cover - transformers absent
    RoFormerConfig = None
    RoFormerEncoder = None
    _HAS_ROFORMER = False

__all__ = [
    "Bi_RectifiedFlow",
    "NaiveV2Diff",
    "NaiveV2DiffLayer",
    "DiffusionEmbedding",
    "ConformerConvModule",
    "Transpose",
    "calc_same_padding",
    "WaveNet",
    "ResidualBlock",
    "SinusoidalPosEmb",
    "Conv1d",
    "Unit2Wav_VAE",
    "ReflowVAEModel",
]


# =====================================================================
# reflow/reflow.py (verbatim, torch_musa removed)
# =====================================================================
class Bi_RectifiedFlow(nn.Module):
    def __init__(self,
                 velocity_fn,
                 spec_min=-12,
                 spec_max=2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.spec_min = spec_min
        self.spec_max = spec_max

    def reflow_loss(self, x_1, x_0, t, cond=None, loss_type='l2_lognorm'):
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond=cond)

        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(x_1 - x_0, v_pred)
        elif loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
            loss = torch.mean(weights[:, None, None, None] * F.mse_loss(x_1 - x_0, v_pred, reduction='none'))
        else:
            raise NotImplementedError()

        return loss

    def sample_euler(self, x, t, dt, cond=None):
        x += self.velocity_fn(x, 1000 * t, cond=cond) * dt
        t += dt
        return x, t

    def sample_rk4(self, x, t, dt, cond=None):
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, 1000 * (t + 0.5 * dt), cond=cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, 1000 * (t + 0.5 * dt), cond=cond)
        k_4 = self.velocity_fn(x + k_3 * dt, 1000 * (t + dt), cond=cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    def sample_heun(self, x, t, dt, cond=None):
        # Predict
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        x_pred = x + k_1 * dt
        t_pred = t + dt
        # Correct
        k_2 = self.velocity_fn(x_pred, 1000 * t_pred, cond=cond)
        x += (k_1 + k_2) / 2 * dt
        t += dt
        return x, t

    def sample_PECECE(self, x, t, dt, cond=None):
        # Predict1
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        x_pred1 = x + k_1 * dt
        t_pred1 = t + dt
        # Correct1
        k_2 = self.velocity_fn(x_pred1, 1000 * t_pred1, cond=cond)
        x_corr1 = x + (k_1 + k_2) / 2 * dt
        # Predict2
        k_3 = self.velocity_fn(x_corr1, 1000 * (t + dt), cond=cond)
        x_pred2 = x_corr1 + k_3 * dt
        # Correct2
        k_4 = self.velocity_fn(x_pred2, 1000 * (t + 2 * dt), cond=cond)
        x += (k_3 + k_4) / 2 * dt
        t += dt
        return x, t

    def forward(self,
                infer=True,
                x_start=None,
                x_end=None,
                cond=None,
                t_start=0.0,
                t_end=1.0,
                infer_step=10,
                method='euler',
                use_tqdm=True):
        if cond is not None:
            cond = cond.transpose(1, 2)  # [B, H, T]
        if not infer:
            x_0 = x_start.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
            x_1 = self.norm_spec(x_end).transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
            t = torch.rand(x_0.shape[0], device=x_0.device)
            t = torch.clip(t, 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, x_0, t, cond=cond)
        else:
            # initial condition and step size of the ODE
            if t_start < 0.0:
                t_start = 0.0
            elif t_start > 1.0:
                t_start = 1.0
            if t_end < 0.0:
                t_end = 0.0
            elif t_end > 1.0:
                t_end = 1.0
            assert t_start < t_end

            if x_start is not None and x_end is None:
                x = x_start.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
                t = torch.full((x_start.shape[0],), t_start, device=x_start.device,
                               dtype=x_start.dtype)
                dt = (t_end - t_start) / infer_step
            elif x_start is None and x_end is not None:
                x = self.norm_spec(x_end).transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
                t = torch.full((x_end.shape[0],), t_end, device=x_end.device,
                               dtype=x_end.dtype)
                dt = -(t_end - t_start) / infer_step

            # sampling
            if method == 'euler':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_euler(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_euler(x, t, dt, cond=cond)

            elif method == 'rk4':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond=cond)

            elif method == 'heun':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_heun(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_heun(x, t, dt, cond=cond)

            elif method == 'PECECE':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_PECECE(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_PECECE(x, t, dt, cond=cond)

            else:
                raise NotImplementedError(method)

            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]

            if dt > 0:
                return self.denorm_spec(x)
            else:
                return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# =====================================================================
# reflow/model_conformer_naive.py (verbatim, torch_musa removed)
# =====================================================================
class ConformerNaiveEncoder(nn.Module):
    """
    Conformer Naive Encoder

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 dim_model: int,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留

        self.encoder_layers = nn.ModuleList(
            [
                CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """

        for (i, layer) in enumerate(self.encoder_layers):
            x = layer(x, mask)
        return x  # (#batch, length, dim_model)


class CFNEncoderLayer(nn.Module):
    """
    Conformer Naive Encoder Layer

    Args:
        dim_model (int): Dimension of model
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.1
        atten_dropout (float): Dropout rate of attention module, default 0.1
    """

    def __init__(self,
                 dim_model: int,
                 num_heads: int = 8,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.1
                 ):
        super().__init__()

        self.conformer = ConformerConvModule(dim_model, use_norm=use_norm, dropout=conv_dropout)

        self.norm = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留

        # selfatt -> fastatt: performer!
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                dropout=atten_dropout,
                activation='gelu'
            )
        else:
            self.attn = None

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        if self.attn is not None:
            x = x + (self.attn(self.norm(x), mask=mask))

        x = x + (self.conformer(x))

        return x  # (#batch, length, dim_model)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            dim,
            expansion_factor=2,
            kernel_size=31,
            dropout=0.,
            use_norm=False,
            conv_model_type='mode1'
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size)

        if conv_model_type == 'mode1':
            self.net = nn.Sequential(
                nn.LayerNorm(dim) if use_norm else nn.Identity(),
                Transpose((1, 2)),
                nn.Conv1d(dim, inner_dim * 2, 1),
                nn.GLU(dim=1),
                nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
                nn.SiLU(),
                nn.Conv1d(inner_dim, dim, 1),
                Transpose((1, 2)),
                nn.Dropout(dropout)
            )
        elif conv_model_type == 'mode2':
            raise NotImplementedError('mode2 not implemented yet')
        else:
            raise ValueError(f'{conv_model_type} is not a valid conv_model_type')

    def forward(self, x):
        return self.net(x)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


# =====================================================================
# reflow/naive_v2_diff.py (verbatim, torch_musa removed)
# =====================================================================
class DiffusionEmbedding(nn.Module):
    """Diffusion Step Embedding"""

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # keep the frequency grid in x.dtype so half/bf16 inference stays
        # dtype-consistent end to end (fp32 grid would promote x -> fp32 and
        # break the following half Linear layers)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NaiveV2DiffLayer(nn.Module):
    def __init__(self,
                 dim_model: int,
                 dim_cond: int,
                 num_heads: int = 4,
                 use_norm: bool = False,
                 conv_only: bool = True,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.1,
                 use_mlp=True,
                 expansion_factor=2,
                 kernel_size=31,
                 wavenet_like=False,
                 conv_model_type='mode1',
                 ):
        super().__init__()

        self.conformer = ConformerConvModule(
            dim_model,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            dropout=conv_dropout,
            use_norm=use_norm,
            conv_model_type=conv_model_type,
        )
        # self.norm = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留
        if wavenet_like:
            self.wavenet_like_proj = nn.Conv1d(dim_model, 2 * dim_model, 1)
        else:
            self.wavenet_like_proj = None

        self.diffusion_step_projection = nn.Conv1d(dim_model, dim_model, 1)
        if dim_cond > 0:
            self.condition_projection = nn.Conv1d(dim_cond, dim_model, 1)
        else:
            self.condition_projection = nn.Identity()

        # selfatt -> fastatt: performer!
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                dropout=atten_dropout,
                activation='gelu'
            )
            self.norm = nn.LayerNorm(dim_model)
        else:
            self.attn = None

    def forward(self, x, condition=None, diffusion_step=None) -> torch.Tensor:
        res_x = x.transpose(1, 2)
        x = x + self.diffusion_step_projection(diffusion_step) + self.condition_projection(condition)
        x = x.transpose(1, 2)

        if self.attn is not None:
            x = (self.attn(self.norm(x)))

        x = self.conformer(x)  # (#batch, dim_model, length)

        if self.wavenet_like_proj is not None:
            x = self.wavenet_like_proj(x.transpose(1, 2)).transpose(1, 2)
            x = F.glu(x, dim=-1)
            return ((x + res_x) / math.sqrt(2.0)).transpose(1, 2), res_x.transpose(1, 2)
        else:
            x = x + res_x
            x = x.transpose(1, 2)
            return x  # (#batch, length, dim_model)


class NaiveV2Diff(nn.Module):
    def __init__(
            self,
            mel_channels=128,
            dim=512,
            use_mlp=True,
            mlp_factor=4,
            condition_dim=256,
            num_layers=20,
            expansion_factor=2,
            kernel_size=31,
            conv_only=True,
            wavenet_like=False,
            use_norm=False,
            conv_model_type='mode1',
            conv_dropout=0.0,
            atten_dropout=0.1,
    ):
        super(NaiveV2Diff, self).__init__()
        self.wavenet_like = wavenet_like
        self.mask_cond_ratio = None

        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = nn.Sequential(
            DiffusionEmbedding(dim),
            nn.Linear(dim, dim * mlp_factor),
            nn.GELU(),
            nn.Linear(dim * mlp_factor, dim),
        )

        if use_mlp and condition_dim > 0:
            self.conditioner_projection = nn.Sequential(
                nn.Conv1d(condition_dim, dim * mlp_factor, 1),
                nn.GELU(),
                nn.Conv1d(dim * mlp_factor, dim, 1),
            )
        else:
            self.conditioner_projection = nn.Identity()

        self.residual_layers = nn.ModuleList(
            [
                NaiveV2DiffLayer(
                    dim_model=dim,
                    dim_cond=dim if use_mlp else condition_dim,
                    num_heads=8,
                    use_norm=use_norm,
                    conv_only=conv_only,
                    conv_dropout=conv_dropout,
                    atten_dropout=atten_dropout,
                    use_mlp=use_mlp,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    wavenet_like=wavenet_like,
                    conv_model_type=conv_model_type,
                )
                for i in range(num_layers)
            ]
        )

        if use_mlp:
            _ = nn.Conv1d(dim * mlp_factor, mel_channels, kernel_size=1)
            nn.init.zeros_(_.weight)
            self.output_projection = nn.Sequential(
                nn.Conv1d(dim, dim * mlp_factor, kernel_size=1),
                nn.GELU(),
                _,
            )
        else:
            self.output_projection = nn.Conv1d(dim, mel_channels, kernel_size=1)
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond=None):
        x = spec
        conditioner = cond
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :return:
        """

        # To keep compatibility with DiffSVC, [B, 1, M, T]
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.gelu(x)

        # Cast long time-step to x.dtype so the parameter-free embedding +
        # following Linear layers stay in half/bf16 inference dtype.
        diffusion_step = self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(-1)
        condition = self.conditioner_projection(conditioner)

        if self.wavenet_like:
            _sk = []
            for layer in self.residual_layers:
                # conditional mask
                if self.mask_cond_ratio is not None:
                    _mask_cond_ratio = random.choice([True, True, False])
                    if _mask_cond_ratio:
                        # 随机从0到mask_cond_ratio中选择一个数
                        _mask_cond_ratio = random.uniform(0, self.mask_cond_ratio)
                    _conditioner = F.dropout(conditioner, _mask_cond_ratio)
                else:
                    _conditioner = conditioner
                # forward
                x, sk = layer(x, _conditioner, diffusion_step)
                _sk.append(sk)
            x = torch.sum(torch.stack(_sk), dim=0) / math.sqrt(len(self.residual_layers))

        else:
            for layer in self.residual_layers:
                # conditional mask
                if self.mask_cond_ratio is not None:
                    _mask_cond_ratio = random.choice([True, True, False])
                    if _mask_cond_ratio:
                        # 随机从0到mask_cond_ratio中选择一个数
                        _mask_cond_ratio = random.uniform(0, self.mask_cond_ratio)
                    _conditioner = F.dropout(conditioner, _mask_cond_ratio)
                else:
                    _conditioner = conditioner
                # forward
                x = layer(x, condition, diffusion_step)

        # MLP and GLU
        x = self.output_projection(x)  # [B, 128, T]

        return x[:, None] if use_4_dim else x


# =====================================================================
# reflow/wavenet.py (verbatim, torch_musa + transformers handled lazily)
# =====================================================================
class Mish(nn.Module):
    """torch.nn.Mish equivalent (explicit class kept for signature parity)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # Compute in the input's dtype when it is floating point (half/bf16
        # inference); long time-steps (training) still use float32.
        dtype = x.dtype if x.is_floating_point else torch.float32
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None].to(dtype) * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, kernel_size=3):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=kernel_size,
            padding=dilation if (kernel_size == 3) else int((kernel_size - 1) * dilation / 2),
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256, dilation=1, kernel_size=3,
                 transformer_use=False, transformer_roformer_use=False, transformer_n_layers=2, transformer_n_head=4):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=(2 ** (i % dilation)) if (dilation != 1) else 1,
                kernel_size=kernel_size
            )
            for i in range(n_layers)
        ])
        self.transformer_roformer_use = transformer_roformer_use if (transformer_roformer_use is not None) else False
        if transformer_use:
            if transformer_roformer_use:
                if not _HAS_ROFORMER:
                    raise ImportError(
                        "WaveNet transformer_roformer_use=True requires the 'transformers' package, "
                        "which usvc deliberately does not depend on."
                    )
                self.transformer = RoFormerEncoder(
                    RoFormerConfig(
                        hidden_size=n_chans,
                        max_position_embeddings=4096,
                        num_attention_heads=transformer_n_head,
                        num_hidden_layers=transformer_n_layers,
                        add_cross_attention=False
                    )
                )
            else:
                transformer_layer = nn.TransformerEncoderLayer(
                    d_model=n_chans,
                    nhead=transformer_n_head,
                    dim_feedforward=n_chans * 4,
                    dropout=0.1,
                    activation='gelu'
                )
                self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_n_layers)
        else:
            self.transformer = None

        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        # Cast long time-step to x.dtype for half/bf16 inference (the
        # parameter-free SinusoidalPosEmb computes in float32 otherwise).
        diffusion_step = self.diffusion_embedding(diffusion_step.to(x.dtype))
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        if self.transformer is not None:
            if self.transformer_roformer_use:
                x = self.transformer(x.transpose(1, 2))[0].transpose(1, 2)
            else:
                x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]


# =====================================================================
# reflow/vocoder.py -> Unit2Wav_VAE (verbatim)
# =====================================================================
class Unit2Wav_VAE(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=6,
            n_chans=512,
            n_hidden=256,
            back_bone='lynxnet',
            use_attention=False):
        super().__init__()
        self.f0_embed = nn.Linear(1, n_hidden)
        self.use_attention = use_attention
        if use_attention:
            self.unit_embed = nn.Linear(n_unit, n_hidden)
            self.volume_embed = nn.Linear(1, n_hidden)
            self.attention = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=n_hidden,
                    nhead=8,
                    dim_feedforward=n_hidden * 4,
                    dropout=0.1,
                    activation='gelu',
                ),
                nn.Linear(n_hidden, out_dims),
            )
        else:
            self.unit_embed = nn.Linear(n_unit, out_dims)
            self.volume_embed = nn.Linear(1, out_dims)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        if back_bone is None or back_bone == 'lynxnet':
            self.reflow_model = Bi_RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=n_hidden, use_mlp=False))
        elif back_bone == 'wavenet':
            self.reflow_model = Bi_RectifiedFlow(WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden))
        else:
            raise ValueError(f" [x] Unknown Backbone: {back_bone}")

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, use_tqdm=True):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """
        # condition
        cond = self.f0_embed((1 + f0 / 700).log())
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    cond = cond + v * self.spk_embed(spk_id_torch - 1)
            else:
                cond = cond + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            cond = cond + self.aug_shift_embed(aug_shift / 5)

        # vae mean
        x = self.unit_embed(units) + self.volume_embed(volume)
        if self.use_attention:
            x = self.attention(x)

        # vae noise
        x += torch.randn_like(x)

        x = self.reflow_model(infer=infer, x_start=x, x_end=gt_spec, cond=cond, infer_step=infer_step, method='euler', use_tqdm=True)

        if return_wav and infer:
            return vocoder.infer(x, f0)
        else:
            return x


# =====================================================================
# usvc wrapper (keeps Unit2Wav_VAE state_dict keys)
# =====================================================================
class ReflowVAEModel(Unit2Wav_VAE):
    """usvc wrapper: inherits Unit2Wav_VAE for checkpoint compat, adds the
    usvc training interface (forward returns (loss, sub), sample returns mel).
    """

    def __init__(self, n_units=768, n_spk=1, win_length=2048, sampling_rate=44100,
                 block_size=512, use_pitch_aug=False, out_dims=128, n_layers=6,
                 n_chans=512, n_hidden=256, back_bone="lynxnet", use_attention=True,
                 t_start=0.0):
        super().__init__(
            sampling_rate, block_size, win_length, n_units, n_spk,
            use_pitch_aug=use_pitch_aug, out_dims=out_dims, n_layers=n_layers,
            n_chans=n_chans, n_hidden=n_hidden, back_bone=back_bone,
            use_attention=use_attention,
        )
        self.t_start = t_start

    def _fit_dims(self, f0, volume, spk_id):
        """Normalize usvc SVCBatch shapes to the original project's convention
        (f0/volume [B, T, 1], spk_id [B, 1], 1-based indexing as in the original
        data loader)."""
        if f0 is not None and f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume is not None and volume.dim() == 2:
            volume = volume.unsqueeze(-1)
        if spk_id is not None and spk_id.dim() == 1:
            spk_id = spk_id.unsqueeze(-1)
        return f0, volume, spk_id

    def forward(self, units, f0, volume, spk_id, gt_spec, aug_shift=None,
                frame_len=None, infer_step=10, method="euler", t_start=None):
        """Training forward.

        Args:
            units:   [B, T, n_unit] content features.
            f0:      [B, T] or [B, T, 1] fundamental frequency (Hz).
            volume:  [B, T] or [B, T, 1] volume (rms).
            spk_id:  [B] or [B, 1] speaker ids (original project convention:
                     spk_embed indexes spk_id - 1).
            gt_spec: [B, T, out_dims] target mel (raw, un-normalized).
            aug_shift: [B, 1] or None pitch augmentation shift (semitones).
            frame_len: ignored (kept for usvc SVCBatch interface parity).
            infer_step / method / t_start: unused in training forward.

        Returns:
            (reflow_loss, {"reflow_loss": float})
        """
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        loss = super().forward(
            units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
            gt_spec=gt_spec, infer=False, return_wav=False, use_tqdm=False,
        )
        return loss, {"reflow_loss": loss.item()}

    def sample(self, units, f0, volume, spk_id, aug_shift=None, frame_len=None,
               infer_step=10, method="euler", t_start=None, use_tqdm=False):
        """Inference forward: VAE latent -> ODE sampling -> mel [B, T, out_dims]."""
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        if t_start is None:
            t_start = self.t_start
        with torch.no_grad():
            mel_gen = super().forward(
                units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
                gt_spec=None, infer=True, infer_step=infer_step, method=method,
                t_start=t_start, use_tqdm=use_tqdm,
            )
        return mel_gen
