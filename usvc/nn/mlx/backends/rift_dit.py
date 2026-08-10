"""MLX 版 rift_dit backend(DiT + Rectified Flow,镜像 ``usvc/backends/rift_dit/model.py``)。

推理最小集:
  - SinusPositionEmbedding / TimestepEmbedding(正弦 step 嵌入 + MLP)
  - CondEmbedding(f0/1200 + rms + cvec)→ dim、InputEmbedding(noised mel + cond)
  - AdaLayerNormZero / _Final(adaLN:siLU + Linear 双输出 modulate)
  - Attention(SDPA + norm_q/norm_k + 可选 rotary embedding,scale=1/dim)
  - ConvMLP(dwconv + ReLU^2 + MLP)、DiTBlock、DiT(depth 层 + final adaLN)
  - RotaryEmbedding(x_transformers 2.x 兼容:interleaved RoPE,use_xpos=False)
  - RIFTDiTModel:组合 DiT + Rectified Flow Euler ODE sample(CFG 支持)

关键差异(torch → mlx):
  - 布局全程 NLC [B,T,C];Conv1d 原生 NLC,权重 mlx (O,K,C)。
  - ModuleList/Sequential → 普通 list(mlx load_weights 键名 ``.0.`` 一致)。
  - 无 bias 的 LayerNorm:mlx ``nn.LayerNorm(dim, affine=False)``(0 参数)。
  - rotary 用 ``mx.fast.scaled_dot_product_attention``,scale **必须显式** =
    ``1/dim``(torch Attention.scale = 1/dim,注意不是 head_dim^-0.5)。
  - LoRA:k_proj/v_proj 可换 ``LoRALinear``(键结构 ``.linear.weight`` + A/B
    与 torch 一致);ckpt 不含 alpha,默认 alpha=rank(见 from_ckpt 说明)。
  - 采样全程 fp32(ODE 迭代数值敏感)。
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "RIFTDiTModel",
    "DiT",
    "DiTBlock",
    "Attention",
    "CondEmbedding",
    "InputEmbedding",
    "TimestepEmbedding",
    "SinusPositionEmbedding",
    "AdaLayerNormZero",
    "AdaLayerNormZero_Final",
    "ConvMLP",
    "LoRALinear",
    "RotaryEmbedding",
]


# ---------------------------------------------------------------------------
# 正弦 / 时间嵌入
# ---------------------------------------------------------------------------


class SinusPositionEmbedding(nn.Module):
    """正弦位置嵌入(镜像 model.py::SinusPositionEmbedding,默认 scale=1000)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.half_dim = half_dim
        emb = math.log(10000.0) / (half_dim - 1)
        self.freqs = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)

    def __call__(self, x, scale: float = 1000):
        """x: [B] → [B, dim] 正弦嵌入。"""
        x = x.astype(mx.float32)
        emb = scale * x[:, None] * self.freqs[None, :]      # [B, half_dim]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedding(nn.Module):
    """时间步条件嵌入(镜像 model.py::TimestepEmbedding)。"""

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time2emb = SinusPositionEmbedding(freq_embed_dim)
        self.time_emb = nn.Linear(freq_embed_dim, dim)
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim, dim)

    def __call__(self, timestep):
        """timestep: [B] → [B, dim]。"""
        time = self.time2emb(timestep)
        time = self.time_emb(time)
        time = nn.silu(time)
        time = self.proj(time)
        return time


# ---------------------------------------------------------------------------
# adaLN modulate(torch modules.py::AdaLayerNormZero / _Final)
# ---------------------------------------------------------------------------


class AdaLayerNormZero(nn.Module):
    """adaLN:输出 6 路 modulate(attn shift/scale/gate + mlp shift/scale/gate)。

    norm 无 affine(0 参数),proj Linear(dim → dim*6)。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x, emb=None):
        """x [B, n, dim], emb [B, dim] → (modulated_x [B,n,dim], gate_msa,
        shift_mlp, scale_mlp, gate_mlp) 各 [B, dim]。"""
        emb = self.proj(nn.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            emb, 6, axis=-1
        )
        x = self.norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    """final adaLN(仅 attn 调制,2 路输出)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x, emb):
        emb = self.proj(nn.silu(emb))
        scale, shift = mx.split(emb, 2, axis=-1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ---------------------------------------------------------------------------
# MLP / Attention(torch modules.py::ReLU2 / ConvMLP / Attention)
# ---------------------------------------------------------------------------


class ReLU2(nn.Module):
    """ReLU 后平方(ReLU^2)。"""

    def __call__(self, x):
        return mx.square(mx.maximum(x, 0.0))


class ConvMLP(nn.Module):
    """dwconv(31, groups=dim)+ LayerNorm + MLP(ff_mult) + ReLU^2。"""

    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4,
                 dropout: float = 0.0, kernel_size: int = 7):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.activation = ReLU2()
        self.dropout = nn.Dropout(dropout)
        self.mlp_proj = nn.Linear(dim, inner_dim)
        self.mlp_out = nn.Linear(inner_dim, dim_out)

    def __call__(self, x):
        """x [B, n, dim] → [B, n, dim_out](NLC,dwconv 原生 NLC,无需转置)。"""
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.mlp_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.mlp_out(x)
        return x


def _rotate_half(x):
    """interleaved RoPE 的 rotate_half(r=2):[..., 2d] → [..., 2d]。"""
    shape = x.shape
    d2 = shape[-1] // 2
    x = x.reshape(*shape[:-1], d2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = mx.stack([-x2, x1], axis=-1)
    return x.reshape(*shape[:-1], -1)


def _apply_rotary_pos_emb(t, freqs, scale=1.0):
    """镜像 x_transformers apply_rotary_pos_emb(use_xpos=False,scale=1)。

    t [B, h, n, hd], freqs [1, n, hd]。
    """
    seq_len = t.shape[-2]
    freqs = freqs[:, -seq_len:, :]              # [1, n, hd]
    if t.ndim == 4 and freqs.ndim == 3:
        # 与 torch apply_rotary_pos_emb 一致:仅当 t 是 [B,h,n,hd] 时加 head 维
        freqs = freqs[:, None, :, :]            # [1, 1, n, hd]
    cos_f = mx.cos(freqs) * scale
    sin_f = mx.sin(freqs) * scale
    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    out = t_rot * cos_f + _rotate_half(t_rot) * sin_f
    return mx.concatenate([out, t_pass], axis=-1)


class RotaryEmbedding(nn.Module):
    """x_transformers RotaryEmbedding('lang', use_xpos=False)。

    inv_freq = 1 / (base ** (arange(0, dim, 2) / dim));forward 输出
    interleaved freqs [1, n, dim] + scale=1.0。
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def forward_from_seq_len(self, seq_len: int):
        t = mx.arange(seq_len, dtype=mx.float32)          # [n]
        freqs = t[:, None] * self.inv_freq[None, :]        # [n, dim/2]
        freqs = mx.stack([freqs, freqs], axis=-1).reshape(seq_len, -1)  # [n, dim] interleaved
        return freqs[None, :, :], 1.0


class LoRALinear(nn.Module):
    """LoRA 包装(镜像 model.py::LoRALinear)。

    键结构与 torch 一致:``linear.weight/bias`` + ``A``/``B``。scale =
    alpha/sqrt(rank),与 torch 相同(注意不是标准 LoRA 的 alpha/rank)。
    """

    def __init__(self, linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / math.sqrt(rank)
        # mlx Linear 无 in/out_features 属性,从 weight 形状取 (out, in)
        self.A = mx.zeros((linear.weight.shape[1], rank))
        self.B = mx.zeros((linear.weight.shape[0], rank))

    def __call__(self, x):
        return self.linear(x) + (x @ self.A) @ self.B.T * self.scale


class Attention(nn.Module):
    """多头自注意力(SDPA)。scale = 1/dim(torch 一致,非 head_dim^-0.5)。"""

    def __init__(self, dim: int, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = int(dim // head_dim)
        self.inner_dim = dim
        self.dropout = dropout
        self.scale = 1 / dim

        self.q_proj = nn.Linear(dim, self.inner_dim)
        self.k_proj = nn.Linear(dim, self.inner_dim)
        self.v_proj = nn.Linear(dim, self.inner_dim)

        self.norm_q = nn.LayerNorm(self.head_dim, affine=False, eps=1e-6)
        self.norm_k = nn.LayerNorm(self.head_dim, affine=False, eps=1e-6)

        self.attn_out = nn.Linear(self.inner_dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None, rope=None):
        """x [B, n, dim] → [B, n, dim]。mask [B, n] bool(True=有效)。"""
        batch_size, seq_len = x.shape[0], x.shape[1]

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # rotary position embedding(在 reshape 前,与 torch 一致)
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            )
            query = _apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = _apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        h, hd = self.num_heads, self.head_dim
        query = query.reshape(batch_size, seq_len, h, hd).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_len, h, hd).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, seq_len, h, hd).transpose(0, 2, 1, 3)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # bool mask:True 参与计算(与 torch F.scaled_dot_product_attention 一致)
        sdpa_mask = None
        if mask is not None:
            sdpa_mask = mask[:, None, None, :]       # [B, 1, 1, T]

        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=sdpa_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        out = self.attn_out(out)
        out = self.attn_dropout(out)

        if mask is not None:
            out = mx.where(mask[:, :, None], out, mx.zeros_like(out))
        return out


# ---------------------------------------------------------------------------
# DiT block / embeddings / backbone
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """adaLN → Attention → gated residual;MLP → gated residual。"""

    def __init__(self, dim: int, head_dim: int, ff_mult: float = 4,
                 dropout: float = 0.0, kernel_size: int = 31):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, head_dim=head_dim, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.mlp = ConvMLP(dim=dim, mult=ff_mult, dropout=dropout,
                           kernel_size=kernel_size)

    def __call__(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa[:, None, :] * attn_output
        norm = self.mlp_norm(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        mlp_output = self.mlp(norm)
        x = x + gate_mlp[:, None, :] * mlp_output
        return x


class CondEmbedding(nn.Module):
    """f0(/1200)+ rms + cvec 条件嵌入(镜像 model.py::CondEmbedding)。"""

    def __init__(self, cvec_dim: int, cond_dim: int):
        super().__init__()
        self.cvec_dim = cvec_dim
        self.cond_dim = cond_dim
        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)
        self.out = nn.Linear(cond_dim, cond_dim)
        self.ln_cvec = nn.LayerNorm(cond_dim, affine=False, eps=1e-6)
        self.ln = nn.LayerNorm(cond_dim, affine=True, eps=1e-6)

    def __call__(self, f0, rms, cvec):
        """f0/rms [B, n] 或 [B, n, 1],cvec [B, n, d] → [B, n, cond_dim]。"""
        if f0.ndim == 2:
            f0 = f0[:, :, None]
        if rms.ndim == 2:
            rms = rms[:, :, None]
        f0_embed = self.f0_embed(f0.astype(mx.float32) / 1200)
        rms_embed = self.rms_embed(rms.astype(mx.float32))
        cvec_embed = self.ln_cvec(self.cvec_embed(cvec.astype(mx.float32)))
        cond = f0_embed + rms_embed + cvec_embed
        cond = self.ln(self.out(cond))
        return cond


class InputEmbedding(nn.Module):
    """noised mel + cond 拼接嵌入(镜像 model.py::InputEmbedding)。"""

    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim, affine=False, eps=1e-6)

    def __call__(self, x, cond_embed):
        x = self.mel_embed(x)
        x = mx.concatenate([x, cond_embed], axis=-1)
        x = self.proj(x)
        x = self.ln(x)
        return x


class DiT(nn.Module):
    """DiT 主干(镜像 model.py::DiT)。x [B, n, n_mel] → [B, n, n_mel]。"""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        ff_mult: float = 4,
        n_mel_channels: int = 128,
        num_speaker: int = 1,
        cvec_dim: int = 768,
        kernel_size: int = 31,
        zero_null_spk: bool = False,
        init_std: float = 1.0,
    ):
        super().__init__()
        self.num_speaker = num_speaker
        self.dim = dim
        self.depth = depth
        self.spk_embed = nn.Embedding(num_speaker, dim)
        self.null_spk_embed = nn.Embedding(1, dim)
        self.tembed = TimestepEmbedding(dim)
        self.cond_embed = CondEmbedding(cvec_dim, dim)
        self.input_embed = InputEmbedding(n_mel_channels, dim)
        self.rotary_embed = RotaryEmbedding(head_dim)
        # mlx 无 ModuleList:普通 list,键名 transformer_blocks.{i}.* 与 torch 一致
        self.transformer_blocks = [
            DiTBlock(dim=dim, head_dim=head_dim, ff_mult=ff_mult,
                     dropout=dropout, kernel_size=kernel_size)
            for _ in range(depth)
        ]
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.output = nn.Linear(dim, n_mel_channels)
        self.init_std = init_std
        if zero_null_spk:
            self.null_spk_embed.weight = mx.zeros_like(self.null_spk_embed.weight)

    # -- LoRA --------------------------------------------------------------

    def apply_lora(self, rank: int, alpha: float):
        """k_proj/v_proj 换成 LoRALinear(镜像 torch apply_lora)。"""
        for block in self.transformer_blocks:
            block.attn.k_proj = LoRALinear(block.attn.k_proj, rank, alpha)
            block.attn.v_proj = LoRALinear(block.attn.v_proj, rank, alpha)

    def merge_lora(self):
        """折叠 LoRA 回普通 Linear(weight += (A @ B.T) * scale)。"""
        for block in self.transformer_blocks:
            for attr in ("k_proj", "v_proj"):
                proj = getattr(block.attn, attr)
                if isinstance(proj, LoRALinear):
                    delta = (proj.A @ proj.B.T) * proj.scale   # [in, out]
                    w = proj.linear.weight + delta.T            # [out, in]
                    lin = nn.Linear(proj.linear.weight.shape[1],
                                    proj.linear.weight.shape[0])
                    lin.weight = w
                    lin.bias = proj.linear.bias
                    setattr(block.attn, attr, lin)

    # -- forward -----------------------------------------------------------

    def __call__(
        self,
        x,                     # [B, n, n_mel] noised mel
        spk,                   # [B] 或标量 speaker
        f0,                    # [B, n] 或 [B, n, 1]
        rms,                   # [B, n] 或 [B, n, 1]
        cvec,                  # [B, n, cvec_dim]
        time,                  # [B] 或标量 step
        drop_speaker=False,    # bool 或 [B] bool
        mask=None,             # [B, n] bool(True=有效)
        skip_layers=None,      # int 或 list[int]
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = mx.broadcast_to(time.reshape(1), (batch,))

        if isinstance(drop_speaker, bool):
            drop_speaker = mx.full((batch,), drop_speaker, dtype=mx.bool_)
        elif drop_speaker.ndim == 0:
            drop_speaker = mx.broadcast_to(drop_speaker.reshape(1), (batch,))

        if spk.ndim == 0:
            spk = mx.full((batch,), spk, dtype=mx.int32)
        spk = spk.astype(mx.int32)

        spk_embeds = self.spk_embed(spk)                       # [B, dim]
        null_spk_embeds = self.null_spk_embed(mx.zeros_like(spk))
        spk_embeds = mx.where(drop_speaker[:, None], null_spk_embeds, spk_embeds)

        t = self.tembed(time)                                   # [B, dim]
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


# ---------------------------------------------------------------------------
# RIFTDiTModel(组合 + ODE sample + ckpt 加载)
# ---------------------------------------------------------------------------


class RIFTDiTModel(nn.Module):
    """DiT + Rectified Flow(镜像 model.py::RIFTDiTModel 的采样部分)。

    sample() 从噪声 Euler ODE 到 mel(CFG:ds/spk/skip strength 可选)。
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        ff_mult: float = 4,
        n_mel_channels: int = 128,
        num_speaker: int = 1,
        cvec_dim: int = 768,
        kernel_size: int = 31,
        zero_null_spk: bool = False,
        init_std: float = 1.0,
        spec_min: float = -12.0,
        spec_max: float = 2.0,
    ):
        super().__init__()
        self.transformer = DiT(
            dim=dim, depth=depth, head_dim=head_dim, dropout=dropout,
            ff_mult=ff_mult, n_mel_channels=n_mel_channels,
            num_speaker=num_speaker, cvec_dim=cvec_dim,
            kernel_size=kernel_size, zero_null_spk=zero_null_spk,
            init_std=init_std,
        )
        self.n_mel_channels = n_mel_channels
        self.spec_min = spec_min
        self.spec_max = spec_max

    # -- 归一化(与 MelNormalizer -12/2 一致)-------------------------------

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    # -- ODE 速度(CFG)------------------------------------------------------

    def _ode_velocity(
        self, t, x, cvec, f0, rms, spk_id, mask, bad_units,
        ds_cfg_strength, spk_cfg_strength, skip_cfg_strength,
        cfg_skip_layers, cfg_rescale,
    ):
        """fn(t, x) → velocity。镜像 model.py::sample 的 fn(含 CFG)。"""
        use_ds_cfg = ds_cfg_strength > 1e-5
        use_spk_cfg = spk_cfg_strength > 1e-5
        use_skip_cfg = skip_cfg_strength > 1e-5
        cfg_flag = use_ds_cfg or use_spk_cfg or use_skip_cfg

        if use_ds_cfg:
            assert bad_units is not None, (
                "bad_units is required when ds_cfg_strength > 0"
            )

        num_cond = 1 + int(use_ds_cfg) + int(use_spk_cfg)
        need_batched = num_cond > 1

        if not need_batched:
            pred = self.transformer(
                x=x, spk=spk_id, f0=f0, rms=rms, cvec=cvec, time=t, mask=mask
            )
            std_pred = pred.std() if (cfg_rescale > 1e-5 and cfg_flag) else None
        else:
            orig_batch = x.shape[0]
            n = num_cond
            x_batched = mx.repeat(x, n, axis=0)
            spk_batched = mx.repeat(spk_id, n, axis=0)
            f0_batched = mx.repeat(f0, n, axis=0)
            rms_batched = mx.repeat(rms, n, axis=0)
            t_batched = mx.broadcast_to(t.reshape(1), (orig_batch * n,))
            mask_batched = mx.repeat(mask, n, axis=0) if mask is not None else None

            if use_ds_cfg and use_spk_cfg:
                cvec_batched = mx.stack([cvec, bad_units, cvec], axis=1).reshape(
                    -1, *cvec.shape[1:]
                )
            elif use_ds_cfg:
                cvec_batched = mx.stack([cvec, bad_units], axis=1).reshape(
                    -1, *cvec.shape[1:]
                )
            else:  # use_spk_cfg only
                cvec_batched = mx.repeat(cvec, n, axis=0)

            drop_speaker_batched = mx.zeros(orig_batch * n, dtype=mx.bool_)
            if use_spk_cfg:
                drop_speaker_batched = (
                    mx.arange(orig_batch * n) % n == (n - 1)
                )

            preds_batched = self.transformer(
                x=x_batched, spk=spk_batched, f0=f0_batched, rms=rms_batched,
                cvec=cvec_batched, time=t_batched, mask=mask_batched,
                drop_speaker=drop_speaker_batched,
            )
            std_pred = (
                preds_batched[0::n].std() if (cfg_rescale > 1e-5 and cfg_flag) else None
            )

            preds_reshaped = preds_batched.reshape(
                orig_batch, n, *preds_batched.shape[1:]
            )
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

    # -- ODE sample --------------------------------------------------------

    def sample(
        self,
        units,                        # cvec [B, n, cvec_dim](ContentVec 输出)
        f0,                           # [B, n] 或 [B, n, 1]
        rms,                          # [B, n] 或 [B, n, 1](volume)
        spk_id=None,                  # [B] 或标量
        aug_shift=None,               # rift_dit 无 pitch aug(接口占位,忽略)
        gt_spec=None,                 # [B, n, M] 可选,pad 帧回填源 mel
        t_start: float = 0.0,
        steps: int = 32,
        bad_units=None,               # 坏 cvec(ds CFG 条件)
        ds_cfg_strength: float = 0.0,
        spk_cfg_strength: float = 0.0,
        skip_cfg_strength: float = 0.0,
        cfg_skip_layers=None,
        cfg_rescale: float = 0.7,
        frame_len=None,               # [B] 有效帧数
        mask=None,                    # [B, n] bool(优先于 frame_len)
    ):
        """从噪声 Euler ODE 采样到 mel。返回 denormalized mel [B, n, M](fp32)。"""
        self.eval()
        cvec = units.astype(mx.float32)
        f0 = f0.astype(mx.float32)
        rms = rms.astype(mx.float32)
        batch, mel_seq_len = cvec.shape[0], cvec.shape[1]
        M = self.n_mel_channels

        if mask is None:
            mask = mx.full((batch, mel_seq_len), True, dtype=mx.bool_)
            if frame_len is not None:
                mask = mx.arange(mel_seq_len)[None, :] < frame_len[:, None]

        if spk_id is None:
            spk_id = mx.zeros((batch,), dtype=mx.int32)
        elif spk_id.ndim == 0:
            spk_id = mx.full((batch,), spk_id, dtype=mx.int32)

        # noise 输入(mask 外位置置 0,与 torch 一致)
        y0 = mx.random.normal((batch, mel_seq_len, M), dtype=mx.float32)
        y0 = mx.where(mask[:, :, None], y0, mx.zeros_like(y0))

        t_start = max(0.0, t_start)
        t = mx.linspace(t_start, 1.0, steps, dtype=mx.float32)

        x = y0
        for i in range(steps - 1):
            dt = t[i + 1] - t[i]
            vel = self._ode_velocity(
                t[i], x, cvec, f0, rms, spk_id, mask, bad_units,
                ds_cfg_strength, spk_cfg_strength, skip_cfg_strength,
                cfg_skip_layers, cfg_rescale,
            )
            x = x + dt * vel
            mx.eval(x)  # 强制计算,释放上步中间结果(控峰值)

        out = self.denorm_spec(x)
        if gt_spec is not None:
            out = mx.where(mask[:, :, None], out, gt_spec.astype(mx.float32))
        return out

    # -- ckpt 加载 ---------------------------------------------------------

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str | Path,
        *,
        config: dict | None = None,
        strict: bool = False,
        dtype=None,
        lora_alpha: float | None = None,
    ):
        """从 torch rift_dit ckpt 加载(推理)。

        自动剥 ``model.`` / ``module.`` 前缀;裸 DiT sd(无 ``transformer.``
        前缀)自动包 ``transformer.``。LoRA ckpt(k_proj/v_proj 为
        ``linear.weight`` + A/B)自动套 LoRALinear —— ckpt 不存 alpha,
        默认 ``alpha=rank``(RIFT-SVC 惯例),可用 ``lora_alpha`` 覆盖,
        或之后调用 ``merge_lora()`` 折叠成普通权重。
        """
        from .._convert import load_torch_ckpt_to_mlx

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype or mx.float32)
        # 剥顶层 lightning wrapper
        for pre in ("model.", "module.", "transformer_model."):
            if any(k.startswith(pre) for k in sd):
                sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
                break
        # 裸 DiT sd → 包 transformer. 前缀
        if any(k.startswith("spk_embed.") for k in sd):
            sd = {"transformer." + k: v for k, v in sd.items()}

        if config is None:
            config = cls._infer_config(sd)

        # LoRA 检测(键形如 transformer_blocks.0.attn.k_proj.linear.weight)
        lora_keys = [k for k in sd if re.search(r"\.attn\.(?:k_proj|v_proj)\.linear\.", k)]
        obj = cls(**config)
        if lora_keys:
            rank = int(sd[lora_keys[0]].shape[1])
            obj.transformer.apply_lora(rank, lora_alpha if lora_alpha is not None else rank)
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        """从权重 shape 反推 RIFTDiTModel 构造参数(键已含 transformer. 前缀)。"""
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731

        cw = shape("transformer.cond_embed.cvec_embed.weight")   # (dim, cvec_dim)
        if cw is None:
            raise ValueError(
                "rift_dit ckpt missing transformer.cond_embed.cvec_embed.weight"
            )
        dim, cvec_dim = int(cw[0]), int(cw[1])

        sw = shape("transformer.spk_embed.weight")
        num_speaker = int(sw[0]) if sw is not None else 1

        mw = shape("transformer.input_embed.mel_embed.weight")   # (dim, n_mel)
        n_mel_channels = int(mw[1]) if mw is not None else 128

        rf = shape("transformer.rotary_embed.inv_freq")          # [head_dim/2]
        head_dim = int(2 * rf[0]) if rf is not None else 64

        depth = len({
            int(k.split("transformer_blocks.")[1].split(".")[0])
            for k in sd if "transformer.transformer_blocks." in k
        })

        mp = shape("transformer.transformer_blocks.0.mlp.mlp_proj.weight")
        ff_mult = int(mp[0] // mp[1]) if mp is not None else 4

        dw = shape("transformer.transformer_blocks.0.mlp.dwconv.weight")
        kernel_size = int(dw[1]) if dw is not None else 31

        return dict(
            dim=dim, depth=depth, head_dim=head_dim,
            n_mel_channels=n_mel_channels, num_speaker=num_speaker,
            cvec_dim=cvec_dim, ff_mult=ff_mult, kernel_size=kernel_size,
        )
