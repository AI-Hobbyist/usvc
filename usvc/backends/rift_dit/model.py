"""rift_dit backend: DiT + Rectified Flow (extracted from RIFT-SVC).

Source (read-only): RIFT-SVC/rift_svc/{dit.py,modules.py,rf.py}. Adapted:
jaxtyping removed; torchdiffeq.odeint -> fixed-grid Euler loop (identical
numerics: 'euler' on a uniform grid is exactly x_{i+1}=x_i+(t_{i+1}-t_i)f(t_i,x_i));
loss via usvc.train.objective.FlowMatchingObjective; mel norm via
MelNormalizer (-12/2); x_transformers kept (RotaryEmbedding).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

from ...nn.normalization import MelNormalizer
from ...train.objective import FlowMatchingObjective

__all__ = ["RIFTDiTModel", "DiT", "CondEmbedding", "InputEmbedding", "TimestepEmbedding"]


def exists(v):
    return v is not None


def lens_to_mask(t, length=None):  # [b] lengths -> [b,n] bool mask (True = valid)
    if not exists(length):
        length = t.amax()
    return torch.arange(length, device=t.device) < t[..., None]


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / math.sqrt(rank)
        self.A = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.normal_(self.A, mean=0, std=math.sqrt(rank) / linear.in_features)
        nn.init.zeros_(self.B)
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

    def forward(self, x):
        lora = (x @ self.A) @ self.B.T
        return self.linear(x) + lora * self.scale


class AdaLayerNormZero(nn.Module):  # returns modulated x + mlp-modulation params
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


class AdaLayerNormZero_Final(nn.Module):  # final layer only (no mlp modulation)
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.proj(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x, inplace=True).square()


class ConvMLP(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, kernel_size=7):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.activation = ReLU2()
        self.dropout = nn.Dropout(dropout)
        self.mlp_proj = nn.Linear(dim, inner_dim)
        self.mlp_out = nn.Linear(inner_dim, dim_out)

    def forward(self, x):  # B,T,C -> B,C,T dwconv -> B,T,C
        return self.mlp_out(self.dropout(self.activation(self.mlp_proj(self.norm(self.dwconv(x.transpose(1, 2)).transpose(1, 2))))))


class Attention(nn.Module):
    def __init__(self, dim, head_dim=64, dropout=0.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = int(dim // head_dim)
        self.inner_dim = dim
        self.dropout = dropout
        self.scale = 1 / dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
        self.norm_k = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
        self.attn_out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rope=None):  # x [b n d]
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if xpos_scale is not None else (1., 1.)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        query, key, value = (i.view(x.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2) for i in (query, key, value))
        query = self.norm_q(query)
        key = self.norm_k(key)
        attn_mask = None
        if mask is not None:
            attn_mask = rearrange(mask, 'b n -> b 1 1 n').expand(x.shape[0], self.num_heads, query.shape[-2], key.shape[-2])
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scale)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.num_heads * self.head_dim).to(query.dtype)
        x = self.attn_dropout(self.attn_out(x))
        if mask is not None:
            x = x.masked_fill(~rearrange(mask, 'b n -> b n 1'), 0.)
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim, head_dim, ff_mult=4, dropout=0.0, kernel_size=31):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim, head_dim=head_dim, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = ConvMLP(dim=dim, mult=ff_mult, dropout=dropout, kernel_size=kernel_size)

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm, mask=mask, rope=rope)
        norm = self.mlp_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp.unsqueeze(1) * self.mlp(norm)


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * (-math.log(10000) / (half_dim - 1)))
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time2emb = SinusPositionEmbedding(freq_embed_dim)
        self.time_emb = nn.Linear(freq_embed_dim, dim)
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim, dim)

    def forward(self, timestep):
        time = self.time2emb(timestep)
        time = self.time_emb(time.to(self.time_emb.weight.dtype))
        return self.proj(self.act(time))


class CondEmbedding(nn.Module):
    def __init__(self, cvec_dim, cond_dim):
        super().__init__()
        self.cvec_dim = cvec_dim
        self.cond_dim = cond_dim
        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)
        self.out = nn.Linear(cond_dim, cond_dim)
        self.ln_cvec = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6)
        self.ln = nn.LayerNorm(cond_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, f0, rms, cvec):  # [b n] -> [b n 1]
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)
        cond = self.f0_embed(f0 / 1200) + self.rms_embed(rms) + self.ln_cvec(self.cvec_embed(cvec))
        return self.ln(self.out(cond))


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond_embed):
        return self.ln(self.proj(torch.cat((self.mel_embed(x), cond_embed), dim=-1)))


class DiT(nn.Module):
    def __init__(self, dim, depth, head_dim=64, dropout=0.0, ff_mult=4, n_mel_channels=128,
                 num_speaker=1, cvec_dim=768, kernel_size=31, zero_null_spk=False, init_std=1):
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
        self.transformer_blocks = nn.ModuleList(DiTBlock(dim, head_dim, ff_mult, dropout, kernel_size) for _ in range(depth))
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.output = nn.Linear(dim, n_mel_channels)
        self.init_std = init_std
        self.apply(self._init_weights)
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.proj.weight, 0)
            nn.init.constant_(block.attn_norm.proj.bias, 0)
        for p in (self.norm_out.proj, self.output):
            nn.init.constant_(p.weight, 0)
            nn.init.constant_(p.bias, 0)
        if zero_null_spk:
            self.null_spk_embed.weight.data.zero_()
            self.null_spk_embed.requires_grad = False

    def _init_weights(self, module):  # spectral parameterization (arXiv:2310.17813)
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            fan_out = module.weight.shape[0]
            fan_in = module.weight.shape[1] * module.weight.shape[2]
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std / math.sqrt(self.dim))

    def forward(self, x, spk, f0, rms, cvec, time, drop_speaker=False, mask=None, skip_layers=None):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b=batch)
        if isinstance(drop_speaker, bool):
            drop_speaker = torch.full((batch,), drop_speaker, dtype=torch.bool, device=x.device)
        spk_embeds = self.spk_embed(spk)
        null_spk_embeds = self.null_spk_embed(torch.zeros_like(spk, dtype=torch.long))
        spk_embeds = torch.where(drop_speaker.unsqueeze(-1), null_spk_embeds, spk_embeds)
        t = self.tembed(time) + spk_embeds
        x = self.input_embed(x, self.cond_embed(f0, rms, cvec))
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        if isinstance(skip_layers, int):
            skip_layers = [skip_layers]
        for i, block in enumerate(self.transformer_blocks):
            if skip_layers is not None and i in skip_layers:
                continue
            x = block(x, t, mask=mask, rope=rope)
        return self.output(self.norm_out(x, t))

    def apply_lora(self, rank, alpha):
        for n, p in self.named_parameters():
            p.requires_grad = False
        self.spk_embed.weight.requires_grad = True
        for block in self.transformer_blocks:  # LoRA on k_proj/v_proj only
            block.attn.k_proj = LoRALinear(block.attn.k_proj, rank, alpha)
            block.attn.v_proj = LoRALinear(block.attn.v_proj, rank, alpha)

    def merge_lora(self):
        for block in self.transformer_blocks:
            for name in ("k_proj", "v_proj"):
                proj = getattr(block.attn, name)
                if isinstance(proj, LoRALinear):
                    with torch.no_grad():
                        proj.linear.weight.add_(proj.B @ proj.A.T)
                    setattr(block.attn, name, proj.linear)

    def freeze_adaln_and_tembed(self):
        for p in list(self.tembed.parameters()) + list(self.norm_out.parameters()):
            p.requires_grad = False
        for block in self.transformer_blocks:
            for p in block.attn_norm.parameters():
                p.requires_grad = False


class RIFTDiTModel(nn.Module):
    """DiT + Rectified Flow (RIFT-SVC rf.py adaptation; torchdiffeq -> Euler)."""

    def __init__(self, dim=512, depth=8, head_dim=64, dropout=0.0, ff_mult=4,
                 n_mel_channels=128, num_speaker=1, cvec_dim=768, kernel_size=31,
                 zero_null_spk=False, init_std=1, time_schedule="lognorm", loss_type="l2"):
        super().__init__()
        self.transformer = DiT(dim=dim, depth=depth, head_dim=head_dim, dropout=dropout,
                               ff_mult=ff_mult, n_mel_channels=n_mel_channels, num_speaker=num_speaker,
                               cvec_dim=cvec_dim, kernel_size=kernel_size, zero_null_spk=zero_null_spk,
                               init_std=init_std)
        self.normalizer = MelNormalizer(spec_min=-12, spec_max=2)
        self.objective = FlowMatchingObjective(self.normalizer, time_schedule=time_schedule, loss_type=loss_type)
        self.n_mel_channels = n_mel_channels

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_velocity(self, xt, t, cond):
        return self.transformer(x=xt, spk=cond["spk_id"], f0=cond["f0"], rms=cond["rms"],
                                cvec=cond["cvec"], time=t, drop_speaker=cond.get("drop_speaker", False),
                                mask=cond.get("mask", None))

    def forward(self, mel, spk_id, f0, rms, cvec, frame_len=None, drop_speaker=False):
        """Flow-matching training loss. Returns (loss, pred)."""
        batch, seq_len = mel.shape[0], mel.shape[1]
        if not exists(frame_len):
            frame_len = torch.full((batch,), seq_len, device=mel.device)
        mask = lens_to_mask(frame_len, length=seq_len)
        cond = {"spk_id": spk_id, "f0": f0, "rms": rms, "cvec": cvec,
                "drop_speaker": drop_speaker, "mask": mask}
        last_pred = {}

        def vfn(xt, t, c):
            last_pred["pred"] = self.compute_velocity(xt, t, c)
            return last_pred["pred"]

        loss = self.objective.compute_loss(mel, vfn, cond, mask=mask)
        return loss, last_pred["pred"]

    @torch.no_grad()
    def sample(self, src_mel, spk_id, f0, rms, cvec, frame_len=None, steps=32,
               bad_cvec=None, ds_cfg_strength=0.0, spk_cfg_strength=0.0,
               skip_cfg_strength=0.0, cfg_skip_layers=None, cfg_rescale=0.7):
        """ODE sample noise -> mel (RIFT-SVC rf.py). Returns (out [b n d], trajectory)."""
        self.eval()
        batch, mel_seq_len, num_mel_channels = src_mel.shape
        device = src_mel.device
        if not exists(frame_len):
            frame_len = torch.full((batch,), mel_seq_len, device=device)
        mask = lens_to_mask(frame_len, length=mel_seq_len)

        def fn(t, x):
            use_ds_cfg = ds_cfg_strength > 1e-5
            use_spk_cfg = spk_cfg_strength > 1e-5
            use_skip_cfg = skip_cfg_strength > 1e-5
            cfg_flag = use_ds_cfg or use_skip_cfg or use_spk_cfg
            if use_ds_cfg:
                assert exists(bad_cvec), "bad_cvec is required when cfg_strength is greater than 0"
            num_cond = 1 + int(use_ds_cfg) + int(use_spk_cfg)
            std_pred = None
            if num_cond == 1:
                pred = self.transformer(x=x, spk=spk_id, f0=f0, rms=rms, cvec=cvec, time=t, mask=mask)
                if cfg_rescale > 1e-5 and cfg_flag:
                    std_pred = pred.std()
            else:
                orig_batch = x.shape[0]
                x_batched = x.repeat_interleave(num_cond, dim=0)
                spk_batched = spk_id.repeat_interleave(num_cond, dim=0)
                f0_batched = f0.repeat_interleave(num_cond, dim=0)
                rms_batched = rms.repeat_interleave(num_cond, dim=0)
                t_batched = t.repeat_interleave(num_cond, dim=0) if isinstance(t, torch.Tensor) and t.ndim > 0 else t
                mask_batched = mask.repeat_interleave(num_cond, dim=0) if exists(mask) else None
                # cvec per cond: [cvec, bad_cvec][ds] then [., cvec][spk]
                if use_ds_cfg and use_spk_cfg:
                    cvec_batched = torch.stack([cvec, bad_cvec, cvec], dim=1).reshape(-1, *cvec.shape[1:])
                elif use_ds_cfg:
                    cvec_batched = torch.stack([cvec, bad_cvec], dim=1).reshape(-1, *cvec.shape[1:])
                else:
                    cvec_batched = cvec.repeat_interleave(num_cond, dim=0)
                drop_speaker_batched = torch.zeros(orig_batch * num_cond, dtype=torch.bool, device=x.device)
                if use_spk_cfg:
                    drop_speaker_batched[num_cond - 1::num_cond] = True
                preds_batched = self.transformer(x=x_batched, spk=spk_batched, f0=f0_batched, rms=rms_batched,
                                                 cvec=cvec_batched, time=t_batched, mask=mask_batched,
                                                 drop_speaker=drop_speaker_batched)
                if cfg_rescale > 1e-5 and cfg_flag:
                    std_pred = preds_batched[0::num_cond].std()
                preds_reshaped = preds_batched.reshape(orig_batch, num_cond, *preds_batched.shape[1:])
                pred = preds_reshaped[:, 0]
                cond_idx = 1
                if use_ds_cfg:
                    pred = pred + (pred - preds_reshaped[:, cond_idx]) * ds_cfg_strength
                    cond_idx += 1
                if use_spk_cfg:
                    pred = pred + (pred - preds_reshaped[:, cond_idx]) * spk_cfg_strength
            if use_skip_cfg:
                skip_pred = self.transformer(x=x, spk=spk_id, f0=f0, rms=rms, cvec=cvec, time=t,
                                             mask=mask, skip_layers=cfg_skip_layers)
                pred = pred + (pred - skip_pred) * skip_cfg_strength
            if cfg_rescale > 1e-5 and cfg_flag:
                std_cfg = pred.std()
                pred = cfg_rescale * (pred * (std_pred / std_cfg)) + (1 - cfg_rescale) * pred
            return pred

        y0 = torch.randn(batch, mel_seq_len, num_mel_channels, device=self.device, dtype=src_mel.dtype)
        y0 = y0.masked_fill(~mask.unsqueeze(-1), 0)
        t = torch.linspace(0, 1, steps, device=self.device, dtype=src_mel.dtype)
        # fixed-grid Euler == torchdiffeq odeint(method='euler') on same grid
        trajectory = [y0]
        x = y0
        for i in range(steps - 1):
            x = x + (t[i + 1] - t[i]) * fn(t[i], x)
            trajectory.append(x)
        trajectory = torch.stack(trajectory)
        out = self.normalizer.denorm(trajectory[-1])
        out = torch.where(mask.unsqueeze(-1), out, src_mel)
        return out, trajectory