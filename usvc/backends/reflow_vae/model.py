"""Reflow-VAE-SVC backend models (Unit2Wav_VAE + Bi_RectifiedFlow).

Verbatim port from /root/ReFlow-VAE-SVC: reflow/reflow.py -> Bi_RectifiedFlow,
reflow/naive_v2_diff.py -> NaiveV2Diff(+Layer), DiffusionEmbedding,
reflow/model_conformer_naive.py -> ConformerConvModule/Transpose/calc_same_padding,
reflow/wavenet.py -> WaveNet, ResidualBlock, SinusoidalPosEmb, Conv1d, Mish,
reflow/vocoder.py -> Unit2Wav_VAE. Class/attribute names and constructor
signatures are preserved so original-project checkpoints load directly.
Deviations: torch_musa branches removed; transformers RoFormerEncoder import
is lazy (usvc forbids transformers; only needed when transformer_use=True).
"""
import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ...nn._commons import calc_same_padding

# Lazy transformers import (default path never needs it).
try:
    from transformers.models.roformer.modeling_roformer import RoFormerConfig, RoFormerEncoder
    _HAS_ROFORMER = True
except ImportError:  # pragma: no cover - transformers absent
    RoFormerConfig, RoFormerEncoder, _HAS_ROFORMER = None, None, False


class Bi_RectifiedFlow(nn.Module):
    def __init__(self, velocity_fn, spec_min=-12, spec_max=2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.spec_min, self.spec_max = spec_min, spec_max

    def reflow_loss(self, x_1, x_0, t, cond=None, loss_type='l2_lognorm', mask=None):
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v = self.velocity_fn(x_t, 1000 * t, cond=cond)
        if mask is not None:
            m = mask.to(x_1.dtype)[:, None, None, :]  # mask [B,T] over T-last [B,1,M,T]
            elem = F.mse_loss(x_1 - x_0, v, reduction='none')
            if loss_type == 'l2_lognorm':
                w = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
                elem = elem * w[:, None, None, None]
            return (elem * m).sum() / (m.sum() * elem.shape[1] * elem.shape[2] + 1e-8)
        if loss_type == 'l1':
            return (x_1 - x_0 - v).abs().mean()
        if loss_type == 'l2':
            return F.mse_loss(x_1 - x_0, v)
        if loss_type != 'l2_lognorm':
            raise NotImplementedError()
        w = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
        return torch.mean(w[:, None, None, None] * F.mse_loss(x_1 - x_0, v, reduction='none'))

    def sample_euler(self, x, t, dt, cond=None):
        x += self.velocity_fn(x, 1000 * t, cond=cond) * dt
        t += dt
        return x, t

    def sample_rk4(self, x, t, dt, cond=None):
        k1 = self.velocity_fn(x, 1000 * t, cond=cond)
        k2 = self.velocity_fn(x + .5 * k1 * dt, 1000 * (t + .5 * dt), cond=cond)
        k3 = self.velocity_fn(x + .5 * k2 * dt, 1000 * (t + .5 * dt), cond=cond)
        k4 = self.velocity_fn(x + k3 * dt, 1000 * (t + dt), cond=cond)
        x += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        t += dt
        return x, t

    def sample_heun(self, x, t, dt, cond=None):
        k1 = self.velocity_fn(x, 1000 * t, cond=cond)
        k2 = self.velocity_fn(x + k1 * dt, 1000 * (t + dt), cond=cond)
        x += (k1 + k2) / 2 * dt
        t += dt
        return x, t

    def sample_PECECE(self, x, t, dt, cond=None):
        k1 = self.velocity_fn(x, 1000 * t, cond=cond)
        k2 = self.velocity_fn(x + k1 * dt, 1000 * (t + dt), cond=cond)
        xc = x + (k1 + k2) / 2 * dt
        k3 = self.velocity_fn(xc, 1000 * (t + dt), cond=cond)
        k4 = self.velocity_fn(xc + k3 * dt, 1000 * (t + 2 * dt), cond=cond)
        x += (k3 + k4) / 2 * dt
        t += dt
        return x, t

    def forward(self, infer=True, x_start=None, x_end=None, cond=None, t_start=0.0, t_end=1.0,
                infer_step=10, method='euler', use_tqdm=True, mask=None):
        if cond is not None:
            cond = cond.transpose(1, 2)  # [B, H, T]
        if not infer:
            x_0 = x_start.transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
            x_1 = self.norm_spec(x_end).transpose(1, 2).unsqueeze(1)
            t = torch.clip(torch.rand(x_0.shape[0], device=x_0.device), 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, x_0, t, cond=cond, mask=mask)
        t_start = 0.0 if t_start < 0.0 else 1.0 if t_start > 1.0 else t_start
        t_end = 0.0 if t_end < 0.0 else 1.0 if t_end > 1.0 else t_end
        assert t_start < t_end
        if x_start is not None and x_end is None:
            x = x_start.transpose(1, 2).unsqueeze(1)
            t = torch.full((x_start.shape[0],), t_start, device=x_start.device, dtype=x_start.dtype)
            dt = (t_end - t_start) / infer_step
        elif x_start is None and x_end is not None:
            x = self.norm_spec(x_end).transpose(1, 2).unsqueeze(1)
            t = torch.full((x_end.shape[0],), t_end, device=x_end.device, dtype=x_end.dtype)
            dt = -(t_end - t_start) / infer_step
        step = {'euler': self.sample_euler, 'rk4': self.sample_rk4,
                'heun': self.sample_heun, 'PECECE': self.sample_PECECE}.get(method)
        if step is None:
            raise NotImplementedError(method)
        it = tqdm(range(infer_step), desc='sample time step', total=infer_step) if use_tqdm else range(infer_step)
        for _ in it:
            x, t = step(x, t, dt, cond=cond)
        x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
        return self.denorm_spec(x) if dt > 0 else x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class ConformerNaiveEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, use_norm=False, conv_only=False,
                 conv_dropout=0., atten_dropout=0.):
        super().__init__()
        self.num_layers, self.num_heads, self.dim_model, self.use_norm = num_layers, num_heads, dim_model, use_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.encoder_layers = nn.ModuleList([
            CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout)
            for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads=8, use_norm=False, conv_only=False, conv_dropout=0., atten_dropout=0.1):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model, use_norm=use_norm, dropout=conv_dropout)
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(dim_model, num_heads, dim_model * 4, atten_dropout, activation='gelu')
        else:
            self.attn = None

    def forward(self, x, mask=None):
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask=mask)
        return x + self.conformer(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0., use_norm=False, conv_model_type='mode1'):
        super().__init__()
        if conv_model_type == 'mode2':
            raise NotImplementedError('mode2 not implemented yet')
        if conv_model_type != 'mode1':
            raise ValueError(f'{conv_model_type} is not a valid conv_model_type')
        inner = dim * expansion_factor
        pad = calc_same_padding(kernel_size)
        self.net = nn.ModuleList((
            nn.LayerNorm(dim) if use_norm else nn.Identity(), Transpose((1, 2)),
            nn.Conv1d(dim, inner * 2, 1), nn.GLU(dim=1),
            nn.Conv1d(inner, inner, kernel_size, padding=pad[0], groups=inner), nn.SiLU(),
            nn.Conv1d(inner, dim, 1), Transpose((1, 2)), nn.Dropout(dropout)))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__(); self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, x):
        device = x.device; half = self.dim // 2
        # grid in input dtype so half/bf16 inference stays dtype-consistent
        # (fp32 grid would promote x and break the following half Linears).
        dtype = x.dtype if x.is_floating_point else torch.float32
        emb = torch.exp(torch.arange(half, device=device, dtype=dtype) * (-math.log(10000) / (half - 1)))
        emb = x[:, None].to(dtype) * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


DiffusionEmbedding = SinusoidalPosEmb  # original class name; identical math


class NaiveV2DiffLayer(nn.Module):
    def __init__(self, dim_model, dim_cond, num_heads=4, use_norm=False, conv_only=True, conv_dropout=0.,
                 atten_dropout=0.1, use_mlp=True, expansion_factor=2, kernel_size=31, wavenet_like=False,
                 conv_model_type='mode1'):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model, expansion_factor=expansion_factor, kernel_size=kernel_size,
                                             dropout=conv_dropout, use_norm=use_norm, conv_model_type=conv_model_type)
        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留
        self.wavenet_like_proj = nn.Conv1d(dim_model, 2 * dim_model, 1) if wavenet_like else None
        self.diffusion_step_projection = nn.Conv1d(dim_model, dim_model, 1)
        self.condition_projection = nn.Conv1d(dim_cond, dim_model, 1) if dim_cond > 0 else nn.Identity()
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(dim_model, num_heads, dim_model * 4, atten_dropout, activation='gelu')
            self.norm = nn.LayerNorm(dim_model)
        else:
            self.attn = None

    def forward(self, x, condition=None, diffusion_step=None):
        res_x = x.transpose(1, 2)
        x = (x + self.diffusion_step_projection(diffusion_step) + self.condition_projection(condition)).transpose(1, 2)
        if self.attn is not None:
            x = self.attn(self.norm(x))
        x = self.conformer(x)
        if self.wavenet_like_proj is None:
            return (x + res_x).transpose(1, 2)
        x = F.glu(self.wavenet_like_proj(x.transpose(1, 2)).transpose(1, 2), dim=-1)
        return ((x + res_x) / math.sqrt(2.0)).transpose(1, 2), res_x.transpose(1, 2)


class NaiveV2Diff(nn.Module):
    def __init__(self, mel_channels=128, dim=512, use_mlp=True, mlp_factor=4, condition_dim=256, num_layers=20,
                 expansion_factor=2, kernel_size=31, conv_only=True, wavenet_like=False, use_norm=False,
                 conv_model_type='mode1', conv_dropout=0.0, atten_dropout=0.1):
        super().__init__()
        self.wavenet_like = wavenet_like
        self.mask_cond_ratio = None
        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = nn.Sequential(
            DiffusionEmbedding(dim), nn.Linear(dim, dim * mlp_factor), nn.GELU(), nn.Linear(dim * mlp_factor, dim))
        self.conditioner_projection = nn.Sequential(
            nn.Conv1d(condition_dim, dim * mlp_factor, 1), nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1)) if use_mlp and condition_dim > 0 else nn.Identity()
        self.residual_layers = nn.ModuleList([
            NaiveV2DiffLayer(dim_model=dim, dim_cond=dim if use_mlp else condition_dim, num_heads=8,
                             use_norm=use_norm, conv_only=conv_only, conv_dropout=conv_dropout,
                             atten_dropout=atten_dropout, use_mlp=use_mlp, expansion_factor=expansion_factor,
                             kernel_size=kernel_size, wavenet_like=wavenet_like, conv_model_type=conv_model_type)
            for _ in range(num_layers)])
        if use_mlp:
            out = nn.Conv1d(dim * mlp_factor, mel_channels, 1)
            nn.init.zeros_(out.weight)
            self.output_projection = nn.Sequential(nn.Conv1d(dim, dim * mlp_factor, 1), nn.GELU(), out)
        else:
            self.output_projection = nn.Conv1d(dim, mel_channels, 1)
            nn.init.zeros_(self.output_projection.weight)

    def _cond(self, conditioner):
        if self.mask_cond_ratio is None:
            return conditioner
        if not random.choice([True, True, False]):
            return conditioner
        return F.dropout(conditioner, random.uniform(0, self.mask_cond_ratio))

    def forward(self, spec, diffusion_step, cond=None):
        use_4d = spec.dim() == 4  # DiffSVC compat [B, 1, M, T]
        x = spec[:, 0] if use_4d else spec
        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"
        x = F.gelu(self.input_projection(x))  # [B, residual_channel, T]
        d = self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(-1)
        if self.wavenet_like:
            sk = []
            for layer in self.residual_layers:
                x, s = layer(x, self._cond(cond), d)
                sk.append(s)
            x = torch.sum(torch.stack(sk), dim=0) / math.sqrt(len(self.residual_layers))
        else:
            condition = self.conditioner_projection(cond)
            for layer in self.residual_layers:
                # quirk kept verbatim: mask RNG/dropout runs per layer but the
                # projected (unmasked) condition is what the layer receives.
                if self.mask_cond_ratio is not None and random.choice([True, True, False]):
                    F.dropout(cond, random.uniform(0, self.mask_cond_ratio))
                x = layer(x, condition, d)
        x = self.output_projection(x)  # [B, 128, T]
        return x[:, None] if use_4d else x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, kernel_size=3):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size,
                                      padding=dilation if kernel_size == 3 else int((kernel_size - 1) * dilation / 2),
                                      dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        d = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = self.dilated_conv(x + d) + self.conditioner_projection(conditioner)
        # split not chunk: avoid onnx::Slice.
        gate, filt = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        res, skip = torch.split(self.output_projection(y), [self.residual_channels, self.residual_channels], dim=1)
        return (x + res) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256, dilation=1, kernel_size=3,
                 transformer_use=False, transformer_roformer_use=False, transformer_n_layers=2, transformer_n_head=4):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(nn.Linear(n_chans, n_chans * 4), Mish(), nn.Linear(n_chans * 4, n_chans))
        self.residual_layers = nn.ModuleList([
            ResidualBlock(encoder_hidden=n_hidden, residual_channels=n_chans,
                          dilation=(2 ** (i % dilation)) if dilation != 1 else 1, kernel_size=kernel_size)
            for i in range(n_layers)])
        self.transformer_roformer_use = transformer_roformer_use if transformer_roformer_use is not None else False
        if transformer_use:
            if transformer_roformer_use:
                if not _HAS_ROFORMER:
                    raise ImportError(
                        "WaveNet transformer_roformer_use=True requires the 'transformers' package, "
                        "which usvc deliberately does not depend on.")
                self.transformer = RoFormerEncoder(RoFormerConfig(
                    hidden_size=n_chans, max_position_embeddings=4096, num_attention_heads=transformer_n_head,
                    num_hidden_layers=transformer_n_layers, add_cross_attention=False))
            else:
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(n_chans, transformer_n_head, n_chans * 4, 0.1, activation='gelu'),
                    num_layers=transformer_n_layers)
        else:
            self.transformer = None
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        x = F.relu(self.input_projection(spec.squeeze(1)))  # [B, residual_channel, T]
        d = self.mlp(self.diffusion_embedding(diffusion_step.to(x.dtype)))
        sk = []
        for layer in self.residual_layers:
            x, s = layer(x, cond, d)
            sk.append(s)
        x = F.relu(self.skip_projection(torch.sum(torch.stack(sk), dim=0) / math.sqrt(len(self.residual_layers))))
        if self.transformer is not None:
            if self.transformer_roformer_use:
                x = self.transformer(x.transpose(1, 2))[0].transpose(1, 2)
            else:
                x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(x)[:, None]  # [B, mel_bins, T]


class Unit2Wav_VAE(nn.Module):
    def __init__(self, sampling_rate, block_size, win_length, n_unit, n_spk, use_pitch_aug=False,
                 out_dims=128, n_layers=6, n_chans=512, n_hidden=256, back_bone='lynxnet', use_attention=False):
        super().__init__()
        self.f0_embed = nn.Linear(1, n_hidden)
        self.use_attention = use_attention
        if use_attention:
            self.unit_embed = nn.Linear(n_unit, n_hidden)
            self.volume_embed = nn.Linear(1, n_hidden)
            self.attention = nn.Sequential(
                nn.TransformerEncoderLayer(n_hidden, 8, n_hidden * 4, 0.1, activation='gelu'),
                nn.Linear(n_hidden, out_dims))
        else:
            self.unit_embed = nn.Linear(n_unit, out_dims)
            self.volume_embed = nn.Linear(1, out_dims)
        self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False) if use_pitch_aug else None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        if back_bone is None or back_bone == 'lynxnet':
            self.reflow_model = Bi_RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans,
                                                             num_layers=n_layers, condition_dim=n_hidden, use_mlp=False))
        elif back_bone == 'wavenet':
            self.reflow_model = Bi_RectifiedFlow(WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans,
                                                         n_hidden=n_hidden))
        else:
            raise ValueError(f" [x] Unknown Backbone: {back_bone}")

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, use_tqdm=True):
        # condition: voiced f0 map; method/use_tqdm/t_start ignored below (verbatim original).
        cond = self.f0_embed((1 + f0 / 700).log())
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    cond = cond + v * self.spk_embed(torch.tensor([[k]], dtype=torch.long, device=units.device))
            else:
                cond = cond + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            cond = cond + self.aug_shift_embed(aug_shift / 5)
        x = self.unit_embed(units) + self.volume_embed(volume)  # vae mean
        if self.use_attention:
            x = self.attention(x)
        x += torch.randn_like(x)  # vae noise
        x = self.reflow_model(infer=infer, x_start=x, x_end=gt_spec, cond=cond, infer_step=infer_step,
                              method='euler', use_tqdm=True)
        return vocoder.infer(x, f0) if return_wav and infer else x


# usvc wrapper: inherits Unit2Wav_VAE for checkpoint compat (state_dict keys unchanged).
class ReflowVAEModel(Unit2Wav_VAE):
    def __init__(self, n_units=768, n_spk=1, win_length=2048, sampling_rate=44100, block_size=512,
                 use_pitch_aug=False, out_dims=128, n_layers=6, n_chans=512, n_hidden=256,
                 back_bone="lynxnet", use_attention=True, t_start=0.0):
        super().__init__(sampling_rate, block_size, win_length, n_units, n_spk, use_pitch_aug, out_dims,
                         n_layers, n_chans, n_hidden, back_bone, use_attention)
        self.t_start = t_start

    def _fit_dims(self, f0, volume, spk_id):
        if f0 is not None and f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume is not None and volume.dim() == 2:
            volume = volume.unsqueeze(-1)
        if spk_id is not None and spk_id.dim() == 1:
            spk_id = spk_id.unsqueeze(-1)  # usvc convention: 0-based, spk_embed indexes directly
        return f0, volume, spk_id

    def forward(self, units, f0, volume, spk_id, gt_spec, aug_shift=None, frame_len=None,
                infer_step=10, method="euler", t_start=None):
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        loss = super().forward(units, f0, volume, spk_id=spk_id, aug_shift=aug_shift, gt_spec=gt_spec,
                               infer=False, return_wav=False, use_tqdm=False)
        return loss, {"reflow_loss": loss.item()}

    def sample(self, units, f0, volume, spk_id, aug_shift=None, frame_len=None, infer_step=10,
               method="euler", t_start=None, use_tqdm=False):
        if t_start is None:
            t_start = self.t_start
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        # NOTE: t_start/method/use_tqdm are accepted but ignored by the original
        # Unit2Wav_VAE reflow call (hardcoded 'euler', tqdm always on); kept for
        # interface parity — do not "fix" without revalidating against upstream.
        with torch.no_grad():
            return super().forward(units, f0, volume, spk_id=spk_id, aug_shift=aug_shift, gt_spec=None,
                                   infer=True, infer_step=infer_step, method=method, t_start=t_start,
                                   use_tqdm=use_tqdm)