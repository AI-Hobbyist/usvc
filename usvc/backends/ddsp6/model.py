"""ddsp6 backend model: CombSubSuperFast + velocity net + RectifiedFlow.

Extracted from the DDSP-SVC training stack (6.1/6.2 and 6.3):
  - reflow/vocoder.py      -> Unit2Wav (combines DDSP + reflow)
  - reflow/reflow.py       -> RectifiedFlow (reflow_loss + euler/rk4 ODE)
  - reflow/lynxnet.py      -> LYNXNet (velocity network, 6.1/6.2)
  - reflow/lynxnet2.py     -> LYNXNet2 (velocity network, 6.3)
  - ddsp/vocoder.py        -> CombSubSuperFast (DDSP synthesizer, STFT-domain filtering)
  - ddsp/unit2control.py   -> Unit2Control (units -> DDSP filter params)
  - ddsp/model_conformer_naive.py -> ConformerNaiveEncoder (Unit2Control backbone;
      conv1d blocks for 6.1/6.2, linear blocks for 6.3)
  - nsf_hifigan/nvSTFT.py  -> MelExtractor (replicated with torch.stft + librosa)

Version selection: ``velocity_net="LYNXNet"`` reproduces DDSP-SVC-6.1/6.2
(including the conv1d ConformerConvModule backbone); ``velocity_net="LYNXNet2"``
(default) reproduces 6.3. Class / attribute names are kept identical to the
original so that original checkpoints load directly (``ddsp_model.*``,
``reflow_model.*``, ``reflow_model.velocity_fn.*`` key prefixes).

torchaudio removed:
  - ``torchaudio.transforms.Resample`` (Vocoder.extract): not needed — usvc
    pre-stages audio at the target sample rate; DDSP synthesizes at model sr.
  - ``torchaudio.transforms.Spectrogram`` (ddsp/loss.py RSSLoss/SSSLoss):
    not used by Unit2Wav training (only by an optional multi-scale spectral
    loss that DDSP6.3's train loop does not apply to the combined loss).
  - nsf_hifigan nvSTFT: reimplemented in pure torch (see MelExtractor).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

__all__ = [
    "LYNXNet",
    "LYNXNet2",
    "ConformerNaiveEncoder",
    "Unit2Control",
    "CombSubSuperFast",
    "RectifiedFlow",
    "MelExtractor",
    "Unit2Wav",
    "DDSP6Model",
]


# ---------------------------------------------------------------------------
# LYNXNet2 (reflow/lynxnet2.py, verbatim)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            out_min, out_max = torch.aminmax(out.detach())
            gate_min, gate_max = torch.aminmax(gate.detach())
            max_abs_out = torch.max(-out_min, out_max).float()
            max_abs_gate = torch.max(-gate_min, gate_max).float()
            max_abs_value = max_abs_out * max_abs_gate
            if max_abs_value > 1000:
                ratio = (1000 / max_abs_value).half()
                gate *= ratio
                return (out * gate).clamp(-1000 * ratio, 1000 * ratio) / ratio
        return out * gate


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


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0.):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            _dropout
        )

    def forward(self, x):
        return x + self.net(x)


class LYNXNet2(nn.Module):
    def __init__(self, in_dims, dim_cond, n_layers=6, n_chans=512, n_dilates=1, dropout=0.):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=n_chans,
                    expansion_factor=n_dilates,
                    kernel_size=31,
                    dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """

        # To keep compatibility with DiffSVC, [B, 1, M, T]
        x = spec
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x.transpose(1, 2))
        x = x + self.conditioner_projection(cond.transpose(1, 2))
        # Cast long time-step to x.dtype for half/bf16 inference.
        x = x + self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(1)

        for layer in self.residual_layers:
            x = layer(x)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        return x[:, None] if use_4_dim else x


# ---------------------------------------------------------------------------
# LYNXNet (reflow/lynxnet.py, DDSP-SVC-6.1/6.2 velocity network, verbatim)
# ---------------------------------------------------------------------------
# 6.1/6.2 use LYNXNet; 6.3 uses LYNXNet2 above. Key structural differences:
#   - input_projection is Conv1d followed by F.gelu (vs Linear, no gelu)
#   - residual layer: x + diffusion_projection(t) + conditioner_projection(cond)
#     before the conv module (post-add), residual added after convmodule
#   - post-norm LayerNorm applied on channel-last before output_projection
# The LYNXNet SwiGLU has no fp16 clamp (6.1 trains in fp32).

class LYNXSwiGLU(nn.Module):
    # 6.1/6.2 SwiGLU (no fp16 overflow protection).
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return out * F.silu(gate)


class LYNXConvModule(nn.Module):
    @staticmethod
    def calc_same_padding(kernel_size):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)

    def __init__(self, dim, expansion_factor, kernel_size=31, in_norm=False, activation='PReLU', dropout=0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        activation_classes = {
            'SiLU': nn.SiLU,
            'ReLU': nn.ReLU,
            'PReLU': lambda: nn.PReLU(inner_dim)
        }
        activation = activation if activation is not None else 'PReLU'
        if activation not in activation_classes:
            raise ValueError(f'{activation} is not a valid activation')
        _activation = activation_classes[activation]()
        padding = self.calc_same_padding(kernel_size)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            LYNXSwiGLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
            _activation,
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            _dropout
        )

    def forward(self, x):
        return self.net(x)


class LYNXNetResidualLayer(nn.Module):
    def __init__(self, dim_cond, dim, expansion_factor, kernel_size=31, in_norm=False, activation='PReLU', dropout=0.):
        super().__init__()
        self.diffusion_projection = nn.Conv1d(dim, dim, 1)
        self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
        self.convmodule = LYNXConvModule(dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size, in_norm=in_norm, activation=activation, dropout=dropout)

    def forward(self, x, conditioner, diffusion_step):
        res_x = x.transpose(1, 2)
        x = x + self.diffusion_projection(diffusion_step) + self.conditioner_projection(conditioner)
        x = x.transpose(1, 2)
        x = self.convmodule(x)  # (#batch, dim, length)
        x = x + res_x
        x = x.transpose(1, 2)

        return x  # (#batch, length, dim)


class LYNXNet(nn.Module):
    def __init__(self, in_dims, dim_cond, n_feats=1, n_layers=6, n_chans=512, n_dilates=2, in_norm=False, activation='PReLU', dropout=0.):
        """
        LYNXNet(Linear Gated Depthwise Separable Convolution Network)

        Used by DDSP-SVC-6.1/6.2 (LYNXNet2 replaced it in 6.3).
        """
        super().__init__()
        self.input_projection = nn.Conv1d(in_dims * n_feats, n_chans, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNetResidualLayer(
                    dim_cond=dim_cond,
                    dim=n_chans,
                    expansion_factor=n_dilates,
                    kernel_size=31,
                    in_norm=in_norm,
                    activation=activation,
                    dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Conv1d(n_chans, in_dims * n_feats, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """

        # To keep compatibility with DiffSVC, [B, 1, M, T]
        x = spec
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.gelu(x)

        # Cast long time-step to x.dtype for half/bf16 inference.
        diffusion_step = self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(-1)

        for layer in self.residual_layers:
            x = layer(x, cond, diffusion_step)

        # post-norm
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # MLP and GLU
        x = self.output_projection(x)  # [B, 128, T]

        return x[:, None] if use_4_dim else x


# ---------------------------------------------------------------------------
# ConformerNaiveEncoder (ddsp/model_conformer_naive.py, verbatim)
# ---------------------------------------------------------------------------

class _ConformerSwiGLU(nn.Module):
    """Original SwiGLU from model_conformer_naive.py (kept separate from the
    LYNXNet2 SwiGLU above — the fp16 clamp path differs slightly)."""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            out_min, out_max = torch.aminmax(out.detach())
            gate_min, gate_max = torch.aminmax(gate.detach())
            if torch.max(-out_min, out_max) * torch.max(-gate_min, gate_max) > 1000:
                return (out.float() * gate.float()).clamp(-1000, 1000).half()
        return out * gate


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            dim,
            expansion_factor=None,
            kernel_size=31,
            dropout=0.,
            use_norm=False,
            conv_model_type='mode1',
            conv_block='linear'
    ):
        super().__init__()

        if conv_block not in ('linear', 'conv1d'):
            raise ValueError(f'{conv_block} is not a valid conv_block')
        if expansion_factor is None:
            # 6.1/6.2 conv1d blocks default to expansion_factor=2; 6.3 linear
            # blocks default to expansion_factor=1 (original defaults).
            expansion_factor = 2 if conv_block == 'conv1d' else 1
        inner_dim = dim * expansion_factor

        if conv_model_type == 'mode1':
            if conv_block == 'conv1d':
                # DDSP-SVC-6.1/6.2 ConformerConvModule (verbatim): depthwise
                # Conv1d + PReLU block, SwiGLU over the channel dim.
                padding = kernel_size // 2
                self.net = nn.Sequential(
                    nn.LayerNorm(dim) if use_norm else nn.Identity(),
                    Transpose((1, 2)),
                    nn.Conv1d(dim, inner_dim * 2, 1),
                    _ConformerSwiGLU61(dim=1),
                    nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding, groups=inner_dim),
                    nn.PReLU(num_parameters=inner_dim),
                    nn.Conv1d(inner_dim, dim, 1),
                    Transpose((1, 2)),
                    nn.Dropout(dropout)
                )
            else:
                self.net = nn.Sequential(
                    nn.LayerNorm(dim) if use_norm else nn.Identity(),
                    Transpose((1, 2)),
                    nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
                    Transpose((1, 2)),
                    nn.Linear(dim, inner_dim * 2),
                    _ConformerSwiGLU(),
                    nn.Linear(inner_dim, inner_dim * 2),
                    _ConformerSwiGLU(),
                    nn.Linear(inner_dim, dim),
                    nn.Dropout(dropout)
                )
        elif conv_model_type == 'mode2':
            raise NotImplementedError('mode2 not implemented yet')
        else:
            raise ValueError(f'{conv_model_type} is not a valid conv_model_type')

    def forward(self, x):
        return self.net(x)


class _ConformerSwiGLU61(nn.Module):
    """6.1/6.2 conformer SwiGLU (chunk-based, no fp16 clamp)."""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * F.silu(gate)


class CFNEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model: int,
                 num_heads: int = 8,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.1,
                 conv_block: str = 'linear'
                 ):
        super().__init__()

        self.conformer = ConformerConvModule(dim_model, use_norm=use_norm, dropout=conv_dropout,
                                             conv_block=conv_block)

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

    def forward(self, x, mask=None) -> torch.Tensor:
        if self.attn is not None:
            x = x + (self.attn(self.norm(x), mask))

        x = x + self.conformer(x)

        return x  # (#batch, length, dim_model)


class ConformerNaiveEncoder(nn.Module):
    """Conformer Naive Encoder (from Diffusion-SVC by CNChTu, MIT).

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16
        conv_only (bool): Whether to use only conv module without attention
        conv_dropout (float): Dropout rate of conv module
        atten_dropout (float): Dropout rate of attention module
        conv_block (str): 'linear' (DDSP-SVC-6.3) or 'conv1d' (6.1/6.2) block style
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 dim_model: int,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.,
                 conv_block: str = 'linear'
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
                CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout,
                                conv_block=conv_block)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None) -> torch.Tensor:
        for (i, layer) in enumerate(self.encoder_layers):
            x = layer(x, mask)
        return x  # (#batch, length, dim_model)


# ---------------------------------------------------------------------------
# Unit2Control (ddsp/unit2control.py, verbatim; dropped unused `gin` import)
# ---------------------------------------------------------------------------

def _split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Unit2Control(nn.Module):
    def __init__(
            self,
            input_channel,
            block_size,
            n_spk,
            output_splits,
            num_layers=3,
            dim_model=256,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False,
            conv_block='linear'):
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

        self.stack = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(input_channel, 512, 3, 1, 1)),
            nn.PReLU(num_parameters=512),
            nn.utils.weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.stack2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(2 * block_size, 512, 3, 1, 1)),
            nn.PReLU(num_parameters=512),
            nn.utils.weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.decoder = ConformerNaiveEncoder(
            num_layers=num_layers,
            num_heads=8,
            dim_model=dim_model,
            use_norm=use_norm,
            conv_only=not use_attention,
            conv_dropout=0,
            atten_dropout=0.1,
            conv_block=conv_block)
        self.norm = nn.LayerNorm(dim_model)
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = nn.utils.weight_norm(nn.Linear(dim_model, self.n_out))

    def forward(self, units, source, noise, volume, spk_id=None, spk_mix_dict=None, aug_shift=None):
        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''
        exciter = torch.cat((source, noise), dim=-1).transpose(1, 2)
        x = self.stack(units.transpose(1, 2)) + self.stack2(exciter)
        x = x.transpose(1, 2) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = _split_to_dict(e, self.output_splits)

        return controls, x


# ---------------------------------------------------------------------------
# CombSubSuperFast (ddsp/vocoder.py — only the synthesizer class, verbatim)
# ---------------------------------------------------------------------------

class CombSubSuperFast(torch.nn.Module):
    def __init__(self,
                 sampling_rate,
                 block_size,
                 win_length,
                 n_unit=256,
                 n_spk=1,
                 num_layers=3,
                 dim_model=256,
                 use_norm=False,
                 use_attention=False,
                 use_pitch_aug=False,
                 conv_block='linear'):
        super().__init__()

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        # Unit2Control
        split_map = {
            'harmonic_magnitude': win_length // 2 + 1,
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': win_length // 2 + 1,
            'noise_phase': win_length // 2 + 1
        }
        self.unit2ctrl = Unit2Control(
            n_unit,
            block_size,
            n_spk,
            split_map,
            num_layers=num_layers,
            dim_model=dim_model,
            use_norm=use_norm,
            use_attention=use_attention,
            use_pitch_aug=use_pitch_aug,
            conv_block=conv_block)

    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None,
                aug_shift=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1
            spk_id: B x 1
        '''
        # combtooth exciter signal
        combtooth = self.fast_source_gen(f0_frames)
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)

        # noise exciter signal
        noise = torch.randn_like(combtooth)
        noise_frames = noise.unfold(1, self.block_size, self.block_size)

        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, combtooth_frames, noise_frames, volume_frames,
                                       spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift)

        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)
        noise_filter = torch.exp(ctrls['noise_magnitude'] + 1.j * np.pi * ctrls['noise_phase']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)

        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        combtooth_stft = torch.stft(
            combtooth,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.block_size,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode=pad_mode)

        # noise part filter
        noise_stft = torch.stft(
            noise,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.block_size,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode=pad_mode)

        # apply the filters
        signal_stft = combtooth_stft * src_filter.permute(0, 2, 1) + noise_stft * noise_filter.permute(0, 2, 1)

        # take the istft to resynthesize audio.
        signal = torch.istft(
            signal_stft,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.block_size,
            window=self.window,
            center=True)

        return signal, hidden


# ---------------------------------------------------------------------------
# RectifiedFlow (reflow/reflow.py, verbatim)
#
# NOTE on FlowMatchingObjective reuse (usvc/train/objective.py):
#   DDSP's reflow loss is mathematically the same rectified-flow velocity loss
#   that FlowMatchingObjective implements, but it is NOT reused here because
#   exact numerical parity with DDSP-SVC-6.3 requires:
#     1. t sampled in [t_start, 1) (t_start >= 0 offset) and clipped to
#        [1e-7, 1-1e-7] — FlowMatchingObjective samples t in [0, 1) with no
#        t_start offset and no clip.
#     2. the SAME t used for xt construction, the velocity network input
#        (scaled x1000) and the l2_lognorm weights. FlowMatchingObjective
#        re-samples t for the lognorm weights (independent sample), which is
#        unbiased in expectation but differs per-step from the original.
#     3. 4-D layout [B, 1, M, T] with T last, and velocity network consuming
#        ``1000 * t`` (LYNXNet2's sinusoidal embedding expects large steps).
#   Keeping the verbatim implementation makes usvc's loss bit-comparable with
#   the original project for the loss-overlap validation.
# ---------------------------------------------------------------------------

class RectifiedFlow(nn.Module):
    def __init__(self,
                 velocity_fn,
                 out_dims=128,
                 spec_min=-12,
                 spec_max=2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max

    def reflow_loss(self, x_1, t, cond, loss_type='l2_lognorm'):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond)

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

    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, 1000 * t, cond) * dt
        t += dt
        return x, t

    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, 1000 * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, 1000 * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, 1000 * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, 1000 * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    def forward(self,
                condition,
                gt_spec=None,
                infer=True,
                infer_step=10,
                method='euler',
                t_start=0.0,
                use_tqdm=True):
        cond = condition.transpose(1, 2)  # [B, H, T]
        b, device = condition.shape[0], condition.device
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec)
            x_1 = x_1.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            t = t_start + (1.0 - t_start) * torch.rand(b, device=device)
            t = torch.clip(t, 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, t, cond=cond)
        else:
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])  # [B, 1, M, T]

            # initial condition and step size of the ODE
            if gt_spec is None:
                x = torch.randn(shape, device=device, dtype=cond.dtype)
                t = torch.full((b,), 0, device=device)
                dt = 1.0 / infer_step
            else:
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device, dtype=cond.dtype)
                t = torch.full((b,), t_start, device=device)
                dt = (1.0 - t_start) / infer_step

            if method == 'euler':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)

            elif method == 'rk4':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond)

            else:
                raise NotImplementedError(method)

            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]

            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# ---------------------------------------------------------------------------
# MelExtractor (nsf_hifigan/nvSTFT.get_mel replica; no torchaudio)
# ---------------------------------------------------------------------------

class MelExtractor:
    """nvSTFT.get_mel replica with keyshift=0, speed=1 (pure torch + librosa).

    This is the mel front-end the original project uses to convert the DDSP
    synthesized waveform into the ``ddsp_mel`` used for the DDSP reconstruction
    loss (MSE against gt_spec). Parameters match the DDSP-SVC-6.3 44.1 kHz
    nsf-hifigan config (128 mels, n_fft=2048, hop=512, fmin=40, fmax=16000).

    Returns log-mel [B, n_frames, n_mels] (dynamic-range compressed, clip 1e-5).
    """

    def __init__(self, sr=44100, n_mels=128, n_fft=2048, win_size=2048,
                 hop_length=512, fmin=40.0, fmax=16000.0, clip_val=1e-5):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self._cache = {}

    def _ensure(self, device):
        key = str(device)
        if key not in self._cache:
            import librosa

            mel_basis = torch.from_numpy(
                librosa.filters.mel(
                    sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                    fmin=self.fmin, fmax=self.fmax,
                )
            ).float().to(device)
            hann = torch.hann_window(self.win_size).to(device)
            self._cache[key] = (mel_basis, hann)
        return self._cache[key]

    def extract(self, y: torch.Tensor) -> torch.Tensor:
        """y: [B, L] waveform at self.sr. Returns [B, n_frames, n_mels] log-mel.

        Deliberately NOT inference_mode/no_grad: the original project runs
        ``vocoder.extract`` inside the training forward with autograd enabled,
        so ddsp_loss/reflow_loss gradients propagate back through the mel
        extraction into the DDSP model. An inference-mode guard here both
        breaks LYNXNet (Conv1d on an inference tensor cannot save for
        backward) and silently detaches the DDSP model from training.
        Callers that don't need gradients (e.g. sampling) wrap in no_grad.
        """
        mel_basis, hann = self._ensure(y.device)

        # Same padding as nvSTFT.get_mel (center=False path).
        pad_left = (self.win_size - self.hop_length) // 2
        pad_right = max(
            (self.win_size - self.hop_length + 1) // 2,
            self.win_size - y.size(-1) - pad_left,
        )
        mode = 'reflect' if pad_right < y.size(-1) else 'constant'
        y = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=hann,
            center=False,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.abs()
        mel = torch.matmul(mel_basis, spec)  # [B, n_mels, T]
        mel = torch.log(torch.clamp(mel, min=self.clip_val))
        return mel.transpose(1, 2)  # [B, T, n_mels]


# ---------------------------------------------------------------------------
# Unit2Wav (reflow/vocoder.py — training/inference combiner, verbatim except
# torchaudio-free mel extraction and a return_wav stub)
# ---------------------------------------------------------------------------

class Unit2Wav(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False,
            out_dims=128,
            n_aux_layers=3,
            n_aux_chans=256,
            n_layers=6,
            n_chans=512,
            mel_extractor=None,
            velocity_net="LYNXNet2"):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        if velocity_net == "LYNXNet":
            velocity_cls = LYNXNet
            conv_block = "conv1d"  # 6.1/6.2 ConformerConvModule (Conv1d+PReLU)
        elif velocity_net == "LYNXNet2":
            velocity_cls = LYNXNet2
            conv_block = "linear"  # 6.3 ConformerConvModule (Linear blocks)
        else:
            raise ValueError(f"unknown velocity_net {velocity_net!r} (expected 'LYNXNet' or 'LYNXNet2')")
        self.velocity_net = velocity_net
        self.ddsp_model = CombSubSuperFast(
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            n_aux_layers if n_aux_layers is not None else 3,
            n_aux_chans if n_aux_chans is not None else 256,
            use_norm,
            use_attention,
            use_pitch_aug,
            conv_block=conv_block)
        self.reflow_model = RectifiedFlow(
            velocity_cls(in_dims=out_dims, dim_cond=out_dims, n_layers=n_layers, n_chans=n_chans),
            out_dims=out_dims)

        # Pure-torch nvSTFT replacement (no torchaudio). Defaults match the
        # 44.1k nsf-hifigan config used by DDSP-SVC-6.3.
        self.mel_extractor = mel_extractor if mel_extractor is not None else MelExtractor(
            sr=sampling_rate, n_mels=out_dims, n_fft=win_length,
            win_size=win_length, hop_length=block_size,
        )

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0,
                silence_front=0, use_tqdm=True):

        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''
        ddsp_wav, hidden = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                           aug_shift=aug_shift, infer=infer)
        start_frame = int(silence_front * self.sampling_rate / self.block_size)
        extractor = vocoder if vocoder is not None else self.mel_extractor
        if extractor is not None:
            ddsp_mel = extractor.extract(ddsp_wav[:, start_frame * self.block_size:])
            # Mel extraction is computed in float32 internally; align to the
            # model dtype so half/bf16 inference stays consistent.
            ddsp_mel = ddsp_mel.to(ddsp_wav.dtype)
        else:
            ddsp_mel = None

        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            if t_start < 1.0:
                reflow_loss = self.reflow_model(ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False)
            else:
                reflow_loss = torch.tensor(0)
            return ddsp_loss, reflow_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_step=infer_step,
                                        method=method, t_start=t_start, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                # Original calls the frozen NSF-HiFiGAN generator here
                # (vocoder.infer). usvc does not ship the HiFiGAN generator
                # weights, so wav-domain synthesis is not available.
                raise NotImplementedError(
                    "return_wav=True requires the frozen NSF-HiFiGAN generator "
                    "(pretrain/nsf_hifigan), which usvc does not bundle."
                )
            else:
                return mel


# ---------------------------------------------------------------------------
# DDSP6Model — usvc-facing wrapper.
#
# Subclasses Unit2Wav so that state_dict keys are identical to the original
# project (``ddsp_model.*`` / ``reflow_model.*``) and original checkpoints load
# directly. ``forward`` is overridden to expose the (total_loss, sub_losses)
# interface used by usvc BaseSVCModule; inference is via ``sample``.
# ---------------------------------------------------------------------------

class DDSP6Model(Unit2Wav):
    """DDSP + reflow model with usvc training interface.

    velocity_net selects the reflow velocity network:
      - "LYNXNet"  (DDSP-SVC-6.1/6.2)
      - "LYNXNet2" (DDSP-SVC-6.3, default)

    Defaults match DDSP-SVC-6.3 configs/reflow.yaml:
      sampling_rate=44100, block_size=512, win_length=2048,
      encoder_out_channels=768 (contentvec768l12tta2x),
      use_norm=True, use_attention=False, use_pitch_aug=True,
      n_aux_layers=6, n_aux_chans=512, n_layers=6, n_chans=1024, out_dims=128.
    """

    def __init__(self, n_units=768, n_spk=1, win_length=2048,
                 sampling_rate=44100, block_size=512,
                 use_norm=True, use_attention=False, use_pitch_aug=True,
                 out_dims=128, n_aux_layers=6, n_aux_chans=512,
                 n_layers=6, n_chans=1024, t_start=0.0, mel_extractor=None,
                 velocity_net="LYNXNet2"):
        super().__init__(
            sampling_rate, block_size, win_length, n_units, n_spk,
            use_norm=use_norm, use_attention=use_attention, use_pitch_aug=use_pitch_aug,
            out_dims=out_dims, n_aux_layers=n_aux_layers, n_aux_chans=n_aux_chans,
            n_layers=n_layers, n_chans=n_chans, mel_extractor=mel_extractor,
            velocity_net=velocity_net,
        )
        self.t_start = t_start

    def forward(self, units, f0, volume, spk_id, gt_spec, aug_shift=None,
                frame_len=None, lambda_ddsp=1.0, t_start=None):
        """Training forward.

        Args:
            units:   [B, T, n_unit] content features.
            f0:      [B, T] or [B, T, 1] fundamental frequency (Hz).
            volume:  [B, T] or [B, T, 1] volume.
            spk_id:  [B] 1-based speaker ids (original DDSP convention; unused
                     when n_spk == 1).
            gt_spec: [B, T, out_dims] target mel (raw, un-normalized).
            aug_shift: [B, 1] or None pitch augmentation shift (semitones).
            frame_len: ignored (kept for usvc SVCBatch interface parity).
            lambda_ddsp: weight of the DDSP reconstruction loss.
            t_start: reflow ODE start (defaults to self.t_start).

        Returns:
            (total_loss, {"ddsp_loss": float, "reflow_loss": float})
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume.dim() == 2:
            volume = volume.unsqueeze(-1)
        if t_start is None:
            t_start = self.t_start

        ddsp_loss, reflow_loss = super().forward(
            units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
            gt_spec=gt_spec, infer=False, t_start=t_start,
        )
        reflow_loss = reflow_loss.to(ddsp_loss.device)
        total = lambda_ddsp * ddsp_loss + reflow_loss
        return total, {
            "ddsp_loss": float(ddsp_loss.item()),
            "reflow_loss": float(reflow_loss.item()),
        }

    def sample(self, units, f0, volume, spk_id, aug_shift=None,
               infer_step=10, method='euler', t_start=None, use_tqdm=False):
        """Inference: DDSP synthesis -> reflow refinement.

        Returns denormalized mel [B, T, out_dims].
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume.dim() == 2:
            volume = volume.unsqueeze(-1)
        if t_start is None:
            t_start = self.t_start

        return super().forward(
            units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
            gt_spec=None, infer=True, infer_step=infer_step,
            method=method, t_start=t_start, use_tqdm=use_tqdm,
        )
