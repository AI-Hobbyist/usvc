"""rift_dit backend: DiT + Rectified Flow model (extracted from RIFT-SVC).

Source files (read-only reference):
  - RIFT-SVC/rift_svc/dit.py     (CondEmbedding, InputEmbedding, DiT)
  - RIFT-SVC/rift_svc/modules.py (LoRALinear, AdaLayerNormZero, Attention,
                                  DiTBlock, TimestepEmbedding)
  - RIFT-SVC/rift_svc/rf.py      (Rectified Flow training/sampling logic)

Adaptations:
  - jaxtyping annotations removed (type-only), replaced with torch.Tensor
  - torchdiffeq.odeint replaced with an equivalent fixed-grid Euler loop
    (identical numerics: torchdiffeq 'euler' on a uniform grid is exactly
    x_{i+1} = x_i + (t_{i+1} - t_i) * f(t_i, x_i))
  - Training loss computed via usvc.train.objective.FlowMatchingObjective
    (shared with ddsp6 backend; math identical to RIFT rf.py forward)
  - Mel normalization via usvc.nn.normalization.MelNormalizer (-12/2)
  - x_transformers kept (RotaryEmbedding, apply_rotary_pos_emb)
"""
from __future__ import annotations

import math
from typing import List, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

from ...nn.normalization import MelNormalizer
from ...train.objective import FlowMatchingObjective

__all__ = [
    "RIFTDiTModel",
    "DiT",
    "CondEmbedding",
    "InputEmbedding",
    "TimestepEmbedding",
]

# ---------------------------------------------------------------------------
# helpers (from RIFT-SVC rift_svc/utils.py)
# ---------------------------------------------------------------------------


def exists(v) -> bool:
    return v is not None


def lens_to_mask(t: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """[b] lengths -> [b, n] boolean mask (True = valid frame)."""
    if not exists(length):
        length = t.amax()
    seq = torch.arange(length, device=t.device)
    return seq < t[..., None]


# ---------------------------------------------------------------------------
# LoRA / attention / DiT blocks (from RIFT-SVC rift_svc/modules.py)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / math.sqrt(rank)
        in_features = linear.in_features
        out_features = linear.out_features
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        # Initialize LoRA parameters
        nn.init.normal_(self.A, mean=0, std=math.sqrt(self.rank) / self.linear.in_features)
        nn.init.zeros_(self.B)
        # Freeze original linear layer parameters
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        original_out = self.linear(x)
        lora_out = (x @ self.A) @ self.B.T
        return original_out + lora_out * self.scale


# AdaLayerNormZero: returns modulated x for attn input + params for mlp modulation
class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.proj(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer (no mlp modulation)
class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.proj(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ReLU^2
class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x, inplace=True).square()


# FeedForward
class ConvMLP(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4, dropout: float = 0.0, kernel_size: int = 7):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.activation = ReLU2()
        self.dropout = nn.Dropout(dropout)
        self.mlp_proj = nn.Linear(dim, inner_dim)
        self.mlp_out = nn.Linear(inner_dim, dim_out)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.mlp_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.mlp_out(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.dim = dim
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = int(dim // head_dim)
        self.inner_dim = dim
        self.dropout = dropout
        self.scale = 1 / dim

        self.q_proj = nn.Linear(dim, self.inner_dim)
        self.k_proj = nn.Linear(dim, self.inner_dim)
        self.v_proj = nn.Linear(dim, self.inner_dim)

        self.norm_q = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)
        self.norm_k = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)

        self.attn_out = nn.Linear(self.inner_dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # [b n d]
        mask: torch.Tensor | None = None,   # [b n]
        rope=None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # projections
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if xpos_scale is not None else (1., 1.)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.num_heads
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # mask
        if mask is not None:
            attn_mask = mask
            attn_mask = rearrange(attn_mask, 'b n -> b 1 1 n')
            attn_mask = attn_mask.expand(batch_size, self.num_heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scale)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        x = x.to(query.dtype)

        # linear proj and dropout
        x = self.attn_out(x)
        x = self.attn_dropout(x)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            x = x.masked_fill(~mask, 0.)

        return x


# DiT Block
class DiTBlock(nn.Module):
    def __init__(
        self, dim: int, head_dim: int, ff_mult: float = 4,
        dropout: float = 0.0, kernel_size: int = 31,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim=dim,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.mlp_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = ConvMLP(dim=dim, mult=ff_mult, dropout=dropout, kernel_size=kernel_size)

    def forward(
        self,
        x: torch.Tensor,                    # [b n d]
        t: torch.Tensor,                    # [b d]
        mask: torch.Tensor | None = None,   # [b n]
        rope=None,
    ) -> torch.Tensor:
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.mlp_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        mlp_output = self.mlp(norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


# sinusoidal position embedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# time step conditioning embedding
class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time2emb = SinusPositionEmbedding(freq_embed_dim)
        self.time_emb = nn.Linear(freq_embed_dim, dim)
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim, dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time = self.time2emb(timestep)
        time = self.time_emb(time.to(self.time_emb.weight.dtype))
        time = self.act(time)
        time = self.proj(time)
        return time


# ---------------------------------------------------------------------------
# Conditioning / input embeddings (from RIFT-SVC rift_svc/dit.py)
# ---------------------------------------------------------------------------


class CondEmbedding(nn.Module):
    """Conditional embedding for f0, rms, cvec."""

    def __init__(self, cvec_dim: int, cond_dim: int):
        super().__init__()
        self.cvec_dim = cvec_dim
        self.cond_dim = cond_dim

        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)
        self.out = nn.Linear(cond_dim, cond_dim)

        self.ln_cvec = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6)
        self.ln = nn.LayerNorm(cond_dim, elementwise_affine=True, eps=1e-6)

    def forward(
        self,
        f0: torch.Tensor,   # [b n]
        rms: torch.Tensor,  # [b n]
        cvec: torch.Tensor, # [b n d]
    ):
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)

        f0_embed = self.f0_embed(f0 / 1200)
        rms_embed = self.rms_embed(rms)
        cvec_embed = self.ln_cvec(self.cvec_embed(cvec))

        cond = f0_embed + rms_embed + cvec_embed
        cond = self.ln(self.out(cond))
        return cond


class InputEmbedding(nn.Module):
    """Noised input mel and context mixing embedding."""

    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor):
        x = self.mel_embed(x)
        x = torch.cat((x, cond_embed), dim=-1)
        x = self.proj(x)
        x = self.ln(x)
        return x


# ---------------------------------------------------------------------------
# DiT backbone (from RIFT-SVC rift_svc/dit.py)
# ---------------------------------------------------------------------------


class DiT(nn.Module):
    def __init__(
        self,
        dim: int, depth: int, head_dim: int = 64, dropout: float = 0.0, ff_mult: int = 4,
        n_mel_channels: int = 128, num_speaker: int = 1, cvec_dim: int = 768,
        kernel_size: int = 31, zero_null_spk: bool = False,
        init_std: float = 1,
    ):
        super().__init__()

        self.num_speaker = num_speaker
        self.spk_embed = nn.Embedding(num_speaker, dim)
        self.null_spk_embed = nn.Embedding(1, dim)
        self.tembed = TimestepEmbedding(dim)
        self.cond_embed = CondEmbedding(cvec_dim, dim)
        self.input_embed = InputEmbedding(n_mel_channels, dim)

        self.rotary_embed = RotaryEmbedding(head_dim)

        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    head_dim=head_dim,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    kernel_size=kernel_size,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.output = nn.Linear(dim, n_mel_channels)

        self.init_std = init_std
        self.apply(self._init_weights)
        for block in self.transformer_blocks:
            torch.nn.init.constant_(block.attn_norm.proj.weight, 0)
            torch.nn.init.constant_(block.attn_norm.proj.bias, 0)

        torch.nn.init.constant_(self.norm_out.proj.weight, 0)
        torch.nn.init.constant_(self.norm_out.proj.bias, 0)
        torch.nn.init.constant_(self.output.weight, 0)
        torch.nn.init.constant_(self.output.bias, 0)

        if zero_null_spk:
            self.null_spk_embed.weight.data.zero_()
            self.null_spk_embed.requires_grad = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            # Spectral parameterization from the paper (arXiv:2310.17813).
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            fan_out = module.weight.shape[0]
            fan_in = module.weight.shape[1] * module.weight.shape[2]
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std / math.sqrt(self.dim))

    def forward(
        self,
        x: torch.Tensor,                    # [b n d1] noised input mel
        spk: torch.Tensor,                  # [b] speaker
        f0: torch.Tensor,                   # [b n]
        rms: torch.Tensor,                  # [b n]
        cvec: torch.Tensor,                 # [b n d2]
        time: torch.Tensor,                 # [b] time step
        drop_speaker: Union[bool, torch.Tensor] = False,   # [b] or scalar
        mask: torch.Tensor | None = None,   # [b n]
        skip_layers: Union[int, List[int], None] = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b=batch)

        if isinstance(drop_speaker, bool):
            drop_speaker = torch.full((batch,), drop_speaker, dtype=torch.bool, device=x.device)

        spk_embeds = self.spk_embed(spk)
        null_spk_embeds = self.null_spk_embed(torch.zeros_like(spk, dtype=torch.long))
        spk_embeds = torch.where(drop_speaker.unsqueeze(-1), null_spk_embeds, spk_embeds)

        t = self.tembed(time)
        t = t + spk_embeds

        cond_embed = self.cond_embed(f0, rms, cvec)
        x = self.input_embed(x, cond_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if skip_layers is not None:
            if isinstance(skip_layers, int):
                skip_layers = [skip_layers]

        for i, block in enumerate(self.transformer_blocks):
            if skip_layers is not None and i in skip_layers:
                continue
            x = block(x, t, mask=mask, rope=rope)

        x = self.norm_out(x, t)
        output = self.output(x)

        return output

    def apply_lora(self, rank, alpha):
        for n, p in self.named_parameters():
            p.requires_grad = False
        self.spk_embed.weight.requires_grad = True
        # Apply LoRA to k_proj and v_proj in each attention block
        for block in self.transformer_blocks:
            block.attn.k_proj = LoRALinear(block.attn.k_proj, rank, alpha)
            block.attn.v_proj = LoRALinear(block.attn.v_proj, rank, alpha)

    def merge_lora(self):
        for block in self.transformer_blocks:
            if isinstance(block.attn.k_proj, LoRALinear):
                with torch.no_grad():
                    delta = block.attn.k_proj.B @ block.attn.k_proj.A.T
                    block.attn.k_proj.linear.weight.add_(delta)
                block.attn.k_proj = block.attn.k_proj.linear

            if isinstance(block.attn.v_proj, LoRALinear):
                with torch.no_grad():
                    delta = block.attn.v_proj.B @ block.attn.v_proj.A.T
                    block.attn.v_proj.linear.weight.add_(delta)
                block.attn.v_proj = block.attn.v_proj.linear

    def freeze_adaln_and_tembed(self):
        for p in self.tembed.parameters():
            p.requires_grad = False
        for p in self.norm_out.parameters():
            p.requires_grad = False
        for block in self.transformer_blocks:
            for p in block.attn_norm.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# RIFTDiTModel: training loss + ODE sampling wrapper
# ---------------------------------------------------------------------------


class RIFTDiTModel(nn.Module):
    """DiT + Rectified Flow model (RIFT-SVC adaptation).

    - Training: ``forward()`` computes flow-matching velocity MSE via
      ``usvc.train.objective.FlowMatchingObjective`` (math identical to
      RIFT-SVC rf.py:192-233).
    - Sampling: ``sample()`` runs the same ODE as RIFT-SVC rf.py:41-190
      (euler on uniform grid, CFG support), with torchdiffeq replaced by an
      equivalent fixed-grid Euler loop.
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        ff_mult: int = 4,
        n_mel_channels: int = 128,
        num_speaker: int = 1,
        cvec_dim: int = 768,
        kernel_size: int = 31,
        zero_null_spk: bool = False,
        init_std: float = 1,
        time_schedule: str = "lognorm",
        loss_type: str = "l2",
    ):
        super().__init__()
        self.transformer = DiT(
            dim=dim,
            depth=depth,
            head_dim=head_dim,
            dropout=dropout,
            ff_mult=ff_mult,
            n_mel_channels=n_mel_channels,
            num_speaker=num_speaker,
            cvec_dim=cvec_dim,
            kernel_size=kernel_size,
            zero_null_spk=zero_null_spk,
            init_std=init_std,
        )
        self.normalizer = MelNormalizer(spec_min=-12, spec_max=2)
        self.objective = FlowMatchingObjective(
            self.normalizer,
            time_schedule=time_schedule,
            loss_type=loss_type,
        )
        self.n_mel_channels = n_mel_channels

    @property
    def device(self):
        return next(self.parameters()).device

    # -- velocity_fn for FlowMatchingObjective ------------------------------

    def compute_velocity(self, xt: torch.Tensor, t: torch.Tensor, cond: dict) -> torch.Tensor:
        """velocity_fn(xt, t, cond) -> pred. cond: dict with spk_id/f0/rms/cvec/mask/drop_speaker."""
        return self.transformer(
            x=xt,
            spk=cond["spk_id"],
            f0=cond["f0"],
            rms=cond["rms"],
            cvec=cond["cvec"],
            time=t,
            drop_speaker=cond.get("drop_speaker", False),
            mask=cond.get("mask", None),
        )

    # -- training -----------------------------------------------------------

    def forward(
        self,
        mel: torch.Tensor,            # [b n d] target mel (raw)
        spk_id: torch.Tensor,         # [b]
        f0: torch.Tensor,             # [b n]
        rms: torch.Tensor,            # [b n]
        cvec: torch.Tensor,           # [b n d]
        frame_len: torch.Tensor | None = None,   # [b]
        drop_speaker: Union[bool, torch.Tensor] = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow-matching training loss. Returns (loss, pred)."""
        batch, seq_len = mel.shape[0], mel.shape[1]
        if not exists(frame_len):
            frame_len = torch.full((batch,), seq_len, device=mel.device)

        mask = lens_to_mask(frame_len, length=seq_len)

        cond = {
            "spk_id": spk_id,
            "f0": f0,
            "rms": rms,
            "cvec": cvec,
            "drop_speaker": drop_speaker,
            "mask": mask,
        }

        last_pred = {}

        def vfn(xt, t, c):
            pred = self.compute_velocity(xt, t, c)
            last_pred["pred"] = pred
            return pred

        loss = self.objective.compute_loss(mel, vfn, cond, mask=mask)
        return loss, last_pred["pred"]

    # -- sampling -----------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        src_mel: torch.Tensor,            # [b n d]
        spk_id: torch.Tensor,             # [b]
        f0: torch.Tensor,                 # [b n]
        rms: torch.Tensor,                # [b n]
        cvec: torch.Tensor,               # [b n d]
        frame_len: torch.Tensor | None = None,   # [b]
        steps: int = 32,
        bad_cvec: torch.Tensor | None = None,
        ds_cfg_strength: float = 0.0,
        spk_cfg_strength: float = 0.0,
        skip_cfg_strength: float = 0.0,
        cfg_skip_layers: Union[int, List[int], None] = None,
        cfg_rescale: float = 0.7,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ODE sample from noise to mel (RIFT-SVC rf.py sample()).

        Returns (out_mel [b n d], trajectory [steps, b, n, d]).
        Equivalent to RIFT's torchdiffeq odeint(method='euler') on the same
        uniform grid — a fixed-grid Euler loop with identical step updates.
        """
        self.eval()

        batch, mel_seq_len, num_mel_channels = src_mel.shape
        device = src_mel.device

        if not exists(frame_len):
            frame_len = torch.full((batch,), mel_seq_len, device=device)

        mask = lens_to_mask(frame_len, length=mel_seq_len)

        # Define the ODE function
        def fn(t, x):
            use_ds_cfg = ds_cfg_strength > 1e-5
            use_spk_cfg = spk_cfg_strength > 1e-5
            use_skip_cfg = skip_cfg_strength > 1e-5
            cfg_flag = use_ds_cfg or use_skip_cfg or use_spk_cfg

            if use_ds_cfg:
                assert exists(bad_cvec), "bad_cvec is required when cfg_strength is greater than 0"

            num_cond = 1 + int(use_ds_cfg) + int(use_spk_cfg)
            need_batched = num_cond > 1

            if not need_batched:
                pred = self.transformer(
                    x=x, spk=spk_id, f0=f0, rms=rms, cvec=cvec, time=t, mask=mask
                )
                std_pred = pred.std() if cfg_rescale > 1e-5 and cfg_flag else None
            else:
                orig_batch = x.shape[0]

                x_batched = x.repeat_interleave(num_cond, dim=0)
                spk_batched = spk_id.repeat_interleave(num_cond, dim=0)
                f0_batched = f0.repeat_interleave(num_cond, dim=0)
                rms_batched = rms.repeat_interleave(num_cond, dim=0)
                t_batched = t.repeat_interleave(num_cond, dim=0) if isinstance(t, torch.Tensor) and t.ndim > 0 else t
                mask_batched = mask.repeat_interleave(num_cond, dim=0) if exists(mask) else None

                if use_ds_cfg and use_spk_cfg:
                    cvec_batched = torch.stack([cvec, bad_cvec, cvec], dim=1).reshape(-1, *cvec.shape[1:])
                elif use_ds_cfg:
                    cvec_batched = torch.stack([cvec, bad_cvec], dim=1).reshape(-1, *cvec.shape[1:])
                else:  # use_spk_cfg only
                    cvec_batched = cvec.repeat_interleave(num_cond, dim=0)

                drop_speaker_batched = torch.zeros(orig_batch * num_cond, dtype=torch.bool, device=x.device)
                if use_spk_cfg:
                    drop_idx = num_cond - 1
                    drop_speaker_batched[drop_idx::num_cond] = True

                preds_batched = self.transformer(
                    x=x_batched, spk=spk_batched, f0=f0_batched, rms=rms_batched,
                    cvec=cvec_batched, time=t_batched, mask=mask_batched,
                    drop_speaker=drop_speaker_batched,
                )

                std_pred = preds_batched[0::num_cond].std() if cfg_rescale > 1e-5 and cfg_flag else None

                preds_reshaped = preds_batched.reshape(orig_batch, num_cond, *preds_batched.shape[1:])
                pred = preds_reshaped[:, 0]

                cond_idx = 1
                if use_ds_cfg:
                    pred = pred + (pred - preds_reshaped[:, cond_idx]) * ds_cfg_strength
                    cond_idx += 1

                if use_spk_cfg:
                    pred = pred + (pred - preds_reshaped[:, cond_idx]) * spk_cfg_strength

            if use_skip_cfg:
                skip_pred = self.transformer(
                    x=x, spk=spk_id, f0=f0, rms=rms, cvec=cvec, time=t,
                    mask=mask, skip_layers=cfg_skip_layers,
                )
                pred = pred + (pred - skip_pred) * skip_cfg_strength

            if cfg_rescale > 1e-5 and cfg_flag:
                std_cfg = pred.std()
                pred_rescaled = pred * (std_pred / std_cfg)
                pred = cfg_rescale * pred_rescaled + (1 - cfg_rescale) * pred

            return pred

        # Noise input
        y0 = torch.randn(batch, mel_seq_len, num_mel_channels, device=self.device,
                         dtype=src_mel.dtype)
        y0 = y0.masked_fill(~mask.unsqueeze(-1), 0)

        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=src_mel.dtype)

        # Fixed-grid Euler loop — numerically identical to torchdiffeq
        # odeint(method='euler') on the same uniform grid:
        #   x_{i+1} = x_i + (t_{i+1} - t_i) * fn(t_i, x_i)
        trajectory_list = [y0]
        x = y0
        for i in range(steps - 1):
            x = x + (t[i + 1] - t[i]) * fn(t[i], x)
            trajectory_list.append(x)
        trajectory = torch.stack(trajectory_list)

        sampled = trajectory[-1]
        out = self.normalizer.denorm(sampled)
        out = torch.where(mask.unsqueeze(-1), out, src_mel)

        return out, trajectory
