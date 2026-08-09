"""Diffusion-SVC style DDPM diffusion model extracted for usvc.

Components:
  - DiffusionEmbedding / ConvNeXtBlock / ConvNext: denoiser network
    (from Diffusion-SVC diffusion/convnext.py, fish-diffusion derived)
  - GaussianDiffusion: DDPM sampler (from Diffusion-SVC diffusion/diffusion.py)
    retained for inference only (ddim/plms/ddpm); dpm-solver/unipc removed.
  - DiffusionModel: condition embeddings + ConvNext denoiser + training loss.

Training loss reuses ``usvc.train.objective.DDPMObjective`` — the math is
identical to Diffusion-SVC's GaussianDiffusion.p_losses:
    betas = linspace(1e-4, 0.02, 1000)
    x0 = norm(mel)                 (spec_min=-12, spec_max=2 → [-1, 1])
    t ~ randint(0, 1000)
    noise = randn_like(x0)
    xt = sqrt(acp[t])*x0 + sqrt(1-acp[t])*noise
    loss = MSE(denoise_fn(xt, t, cond), noise)

Only layout differs: p_losses uses [B, 1, M, T]; DDPMObjective uses [B, T, M].
The denoise_fn wrapper transposes, so the loss is numerically identical.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from usvc._triton.dwconv1d import dwconv1d as _triton_dwconv1d
    _HAS_TRITON_DWCONV = True
except Exception:
    _HAS_TRITON_DWCONV = False


def _dwconv_apply(x, weight, bias, dilation):
    """Triton depthwise conv1d with PyTorch fallback (数值等价)."""
    if _HAS_TRITON_DWCONV:
        try:
            return _triton_dwconv1d(x.contiguous(), weight, bias, dilation)
        except Exception:
            pass
    return F.conv1d(x, weight, bias, groups=weight.shape[0],
                    dilation=dilation, padding=dilation * (weight.shape[2] - 1) // 2)

from ...nn.normalization import MelNormalizer
from ...train.objective import DDPMObjective

__all__ = [
    "DiffusionEmbedding",
    "ConvNeXtBlock",
    "ConvNext",
    "GaussianDiffusion",
    "DiffusionModel",
]


# ---------------------------------------------------------------------------
# Denoiser: ConvNext (from Diffusion-SVC diffusion/convnext.py)
# ---------------------------------------------------------------------------


class DiffusionEmbedding(nn.Module):
    """Diffusion Step Embedding."""

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        dtype = x.dtype if x.is_floating_point else torch.float32
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None].to(dtype) * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from ConvNeXt to 1D audio signal."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        layer_scale_init_value: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=dilation,
            padding=int(dilation * (7 - 1) / 2),
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )
        self.diffusion_step_projection = nn.Conv1d(dim, dim, 1)
        self.condition_projection = nn.Conv1d(dim, dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        diffusion_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        x = (
            x
            + self.diffusion_step_projection(diffusion_step)
            + self.condition_projection(condition)
        )

        if getattr(self, "_use_triton_dwconv", False):
            # Triton depthwise conv1d via registered custom op (compile-friendly,
            # bit-level equal, ~2.3x fwd). contiguous() satisfies CUDA-graph
            # capture layout checks.
            x = _dwconv_apply(x, self.dwconv.weight, self.dwconv.bias,
                              self.dwconv.dilation[0])
        else:
            x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNext(nn.Module):
    """ConvNext denoiser. Input/output [B, M, T]; supports [B, 1, M, T]."""

    def __init__(
        self,
        mel_channels=128,
        dim=512,
        mlp_factor=4,
        condition_dim=256,
        num_layers=20,
        dilation_cycle=4,
        gradient_checkpointing=False,
    ):
        super(ConvNext, self).__init__()

        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = nn.Sequential(
            DiffusionEmbedding(dim),
            nn.Linear(dim, dim * mlp_factor),
            nn.GELU(),
            nn.Linear(dim * mlp_factor, dim),
        )
        self.conditioner_projection = nn.Sequential(
            nn.Conv1d(condition_dim, dim * mlp_factor, 1),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        )

        self.residual_layers = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=dim * mlp_factor,
                    dilation=2 ** (i % dilation_cycle),
                )
                for i in range(num_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, mel_channels, kernel_size=1),
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, spec, diffusion_step, cond):
        # To keep compatibility with DiffSVC, [B, 1, M, T]
        use_4_dim = False
        if spec.dim() == 4:
            spec = spec[:, 0]
            use_4_dim = True

        assert spec.dim() == 3, f"mel must be 3 dim tensor, but got {spec.dim()}"

        x = self.input_projection(spec)  # x [B, residual_channel, T]
        x = F.gelu(x)

        # diffusion_step is a long tensor; the embedding computes in float32.
        # Cast it to x.dtype first so the embedding + following Linear layers
        # stay in half/bf16 inference dtype (the embedding uses x.dtype when
        # x is a floating point tensor).
        diffusion_step = self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(-1)
        condition = self.conditioner_projection(cond)

        for layer in self.residual_layers:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, condition, diffusion_step
                )
            else:
                x = layer(x, condition, diffusion_step)

        x = self.output_projection(x)  # [B, 128, T]

        return x[:, None] if use_4_dim else x


# ---------------------------------------------------------------------------
# DDPM sampler (from Diffusion-SVC diffusion/diffusion.py; dpm-solver/unipc removed)
# ---------------------------------------------------------------------------


def _exists(x):
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if callable(d) else d


def _extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    """DDPM core for sampling (training loss lives in DDPMObjective).

    Extracted from Diffusion-SVC diffusion/diffusion.py. Sampling methods
    retained: ddim, plms (pndm), ddpm. dpm-solver/unipc removed.
    """

    def __init__(
        self,
        denoise_fn,
        out_dims=128,
        timesteps=1000,
        k_step=1000,
        max_beta=0.02,
        spec_min=-12,
        spec_max=2,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.out_dims = out_dims

        betas = np.linspace(1e-4, max_beta, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.k_step = k_step
        self.noise_list = deque(maxlen=4)

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )
        self.register_buffer(
            "spec_min", torch.FloatTensor([spec_min])[None, None, :out_dims]
        )
        self.register_buffer(
            "spec_max", torch.FloatTensor([spec_max])[None, None, :out_dims]
        )

    # -- helpers ------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        noise = _default(noise, lambda: torch.randn_like(x_start))
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = _noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        a_t = _extract(self.alphas_cumprod, t, x.shape).to(x.dtype)
        a_prev = _extract(
            self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape
        ).to(x.dtype)
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (
            x / a_t.sqrt()
            + (((1 - a_prev) / a_prev).sqrt() - ((1 - a_t) / a_t).sqrt())
            * noise_pred
        )
        return x_prev

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """PLMS sampling (Pseudo Numerical Methods for Diffusion Models)."""

        def get_x_pred(x, noise_t, t):
            a_t = _extract(self.alphas_cumprod, t, x.shape).to(x.dtype)
            a_prev = _extract(
                self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape
            ).to(x.dtype)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1
                / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
                * noise_t
            )
            return x + x_delta

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(
                x_pred, torch.maximum(t - interval, torch.zeros_like(t)), cond=cond
            )
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (
                55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]
            ) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        condition,
        gt_spec=None,
        infer_speedup=10,
        method="ddim",
        k_step=None,
        use_tqdm=False,
    ):
        """Sample mel from condition. Returns denormalized [B, T, M].

        Args:
            condition: [B, T, C] condition embeddings (before transpose).
            gt_spec: optional [B, T, M] for partial noising (k_step < 1000).
            method: 'ddim' | 'plms' | 'ddpm'.
        """
        cond = condition.transpose(1, 2)  # [B, C, T]
        b, device = condition.shape[0], condition.device
        shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])

        if gt_spec is None or k_step is None:
            t = self.k_step
            x = torch.randn(shape, device=device, dtype=condition.dtype)
        else:
            t = k_step
            norm_spec = self.norm_spec(gt_spec)
            norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
            x = self.q_sample(
                x_start=norm_spec, t=torch.tensor([t - 1], device=device).long()
            )

        bar = None
        if use_tqdm:
            from tqdm import tqdm

            bar = tqdm(desc="sample time step", total=t // infer_speedup)

        if method == "plms":
            self.noise_list = deque(maxlen=4)
            for i in reversed(range(0, t, infer_speedup)):
                x = self.p_sample_plms(
                    x, torch.full((b,), i, device=device, dtype=torch.long),
                    infer_speedup, cond=cond,
                )
                if bar:
                    bar.update(1)
        elif method == "ddim":
            for i in reversed(range(0, t, infer_speedup)):
                x = self.p_sample_ddim(
                    x, torch.full((b,), i, device=device, dtype=torch.long),
                    infer_speedup, cond=cond,
                )
                if bar:
                    bar.update(1)
        elif method == "ddpm":
            for i in reversed(range(0, t)):
                x = self.p_sample(
                    x, torch.full((b,), i, device=device, dtype=torch.long), cond
                )
                if bar:
                    bar.update(1)
        else:
            raise NotImplementedError(method)

        if bar:
            bar.close()

        x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
        return self.denorm_spec(x)

    # -- normalization (kept for sampler parity) ----------------------------

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# ---------------------------------------------------------------------------
# DiffusionModel: condition embeddings + ConvNext + training loss
# (Unit2MelV2 extraction; naive/combo/speaker-encoder removed)
# ---------------------------------------------------------------------------


class DiffusionModel(nn.Module):
    """Diffusion-SVC style DDPM model for usvc training.

    Training forward returns (loss, {"eps_loss": value}) using DDPMObjective.
    Conditioning: units/f0/volume embeddings + optional spk/aug_shift.
    """

    def __init__(
        self,
        n_units=768,
        n_spk=1,
        out_dims=128,
        n_hidden=256,
        use_pitch_aug=False,
        timesteps=1000,
        max_beta=0.02,
        spec_min=-12,
        spec_max=2,
        cn_layers=20,
        cn_chans=384,
        mlp_factor=4,
        gradient_checkpointing=False,
        k_step=1000,
        denoiser="convnext",
        wn_layers=20,
        wn_chans=384,
    ):
        super().__init__()
        self.n_spk = n_spk
        self.out_dims = out_dims
        self.n_hidden = n_hidden
        self.use_pitch_aug = use_pitch_aug
        self.k_step = k_step

        # condition embeddings (Unit2MelV2)
        self.unit_embed = nn.Linear(n_units, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        else:
            self.spk_embed = None

        # denoiser (registered only via decoder; see GaussianDiffusion)
        if denoiser == "wavenet":
            from .wavenet import WaveNet

            denoise_fn = WaveNet(
                in_dims=out_dims,
                n_layers=wn_layers,
                n_chans=wn_chans,
                n_hidden=n_hidden,
            )
        else:  # convnext (existing default)
            denoise_fn = ConvNext(
                mel_channels=out_dims,
                dim=cn_chans,
                mlp_factor=mlp_factor,
                condition_dim=n_hidden,
                num_layers=cn_layers,
                dilation_cycle=4,
                gradient_checkpointing=gradient_checkpointing,
            )

        # sampler (inference only)
        self.decoder = GaussianDiffusion(
            denoise_fn,
            out_dims=out_dims,
            timesteps=timesteps,
            k_step=k_step,
            max_beta=max_beta,
            spec_min=spec_min,
            spec_max=spec_max,
        )

        # training loss: reuse shared DDPMObjective (math identical to p_losses)
        self.objective = DDPMObjective(
            MelNormalizer(spec_min, spec_max),
            timesteps=timesteps,
            max_beta=max_beta,
        )
        self.objective.register_buffers(self)

    def _embed_condition(self, units, f0, volume, spk_id, aug_shift):
        """Build [B, T, n_hidden] condition embeddings."""
        # f0/volume may be [B, T] or [B, T, 1]
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume.dim() == 2:
            volume = volume.unsqueeze(-1)

        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.spk_embed is not None and spk_id is not None:
            # usvc convention: 0-based speaker ids (dataset spk2idx).
            # spk_id is [B] (usvc SVCBatch); unsqueeze to [B, 1] so the
            # embedding broadcasts against x [B, T, n_hidden] correctly
            # regardless of T (embedding([B]) -> [B, H] would only align
            # when T == B by accident).
            spk = spk_id if spk_id.dim() > 1 else spk_id.unsqueeze(1)
            x = x + self.spk_embed(spk)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        return x

    def _denoise_wrapper(self, xt, t, cond):
        """Adapt DDPMObjective [B, T, M] layout to ConvNext [B, 1, M, T].

        cond: [B, n_hidden, T] (already transposed).
        """
        xt4 = xt.transpose(1, 2).unsqueeze(1)  # [B, T, M] -> [B, 1, M, T]
        pred4 = self.decoder.denoise_fn(xt4, t, cond)  # [B, 1, M, T]
        return pred4[:, 0].transpose(1, 2)  # [B, T, M]

    def forward(
        self,
        units,
        f0,
        volume,
        spk_id,
        gt_spec,
        aug_shift=None,
        frame_len=None,
    ):
        """Training forward. Returns (loss, {"eps_loss": scalar}).

        Args:
            units: [B, T, D]
            f0: [B, T] or [B, T, 1]
            volume: [B, T] or [B, T, 1]
            spk_id: [B] 1-based (Diffusion-SVC convention)
            gt_spec: [B, T, M] target mel (raw)
            aug_shift: [B] or [B, 1] optional pitch shift
            frame_len: [B] true frame counts; used to mask padding frames so the
                DDPM loss is averaged only over real frames (padding mel=0 is a
                loud non-silence target that would otherwise pollute gradients).
        """
        x = self._embed_condition(units, f0, volume, spk_id, aug_shift)
        cond = x.transpose(1, 2)  # [B, n_hidden, T]

        from ...train.objective import lens_to_mask
        mask = lens_to_mask(frame_len, length=gt_spec.shape[1]) if frame_len is not None else None
        if mask is not None:
            mask = mask.to(gt_spec.device)  # frame_len may be on CPU
        loss = self.objective.compute_loss(
            gt_spec, self._denoise_wrapper, cond, t_max=self.k_step, mask=mask,
        )
        return loss, {"eps_loss": loss.item()}

    @torch.no_grad()
    def sample(
        self,
        units,
        f0,
        volume,
        spk_id,
        aug_shift=None,
        gt_spec=None,
        infer_speedup=10,
        method="ddim",
        k_step=None,
        use_tqdm=False,
    ):
        """Generate mel [B, T, M]. gt_spec optional for partial noising."""
        x = self._embed_condition(units, f0, volume, spk_id, aug_shift)
        return self.decoder.sample(
            x,
            gt_spec=gt_spec,
            infer_speedup=infer_speedup,
            method=method,
            k_step=k_step,
            use_tqdm=use_tqdm,
        )
