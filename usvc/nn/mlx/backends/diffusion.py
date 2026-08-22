"""MLX diffusion backend(mel-domain DDPM,镜像 backends/diffusion/model.py)。

NLC 全程 [B,T,M](torch denoiser 是 [B,1,M,T]);weight_norm 由加载时合并;
ModuleList/Sequential → 普通 list(键名 .0. 一致);gelu 恒精确;采样全程 fp32
(DDPM/PLMS 迭代数值敏感,engine.py:226)。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .._ops import denorm_spec as _denorm_spec, norm_spec as _norm_spec

__all__ = [
    "DiffusionEmbedding",
    "SinusoidalPosEmb",
    "ConvNeXtBlock",
    "ConvNext",
    "ResidualBlock",
    "WaveNet",
    "GaussianDiffusion",
    "DiffusionModel",
]


class DiffusionEmbedding(nn.Module):
    def __init__(self, d_denoiser: int):
        super().__init__()
        self.dim = d_denoiser
        half_dim = d_denoiser // 2
        self.half_dim = half_dim
        emb = math.log(10000.0) / (half_dim - 1)
        self.freqs = mx.exp(mx.arange(half_dim) * -emb)  # [half_dim]

    def __call__(self, x):
        x = x.astype(mx.float32)
        x = x[:, None] * self.freqs[None, :]
        return mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.half_dim = half_dim
        emb = math.log(10000.0) / (half_dim - 1)
        self.freqs = mx.exp(mx.arange(half_dim) * -emb)

    def __call__(self, x):
        x = x.astype(mx.float32)
        x = x[:, None] * self.freqs[None, :]
        return mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, groups=dim, dilation=dilation,
            padding=int(dilation * (7 - 1) / 2),
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (mx.full((dim,), float(layer_scale_init_value))
                      if layer_scale_init_value is not None and layer_scale_init_value > 0
                      else None)
        self.diffusion_step_projection = nn.Conv1d(dim, dim, 1)
        self.condition_projection = nn.Conv1d(dim, dim, 1)

    def __call__(self, x, condition, diffusion_step):
        residual = x
        x = (x + self.diffusion_step_projection(diffusion_step)
             + self.condition_projection(condition))
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return residual + x


class ConvNext(nn.Module):
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
        super().__init__()
        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = [  # 键名 diffusion_embedding.{0..3} 与 torch Sequential 一致
            DiffusionEmbedding(dim), nn.Linear(dim, dim * mlp_factor),
            nn.GELU(), nn.Linear(dim * mlp_factor, dim),
        ]
        self.conditioner_projection = [  # 键名 0/2(Conv1d, GELU, Conv1d)
            nn.Conv1d(condition_dim, dim * mlp_factor, 1), nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        ]
        self.residual_layers = [
            ConvNeXtBlock(dim=dim, intermediate_dim=dim * mlp_factor,
                          dilation=2 ** (i % dilation_cycle))
            for i in range(num_layers)
        ]
        self.output_projection = [  # 键名 0/2
            nn.Conv1d(dim, dim, kernel_size=1), nn.GELU(),
            nn.Conv1d(dim, mel_channels, kernel_size=1),
        ]
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, spec, diffusion_step, cond):
        x = nn.gelu(self.input_projection(spec))
        step_emb = diffusion_step
        for m in self.diffusion_embedding:
            step_emb = m(step_emb)              # [B] → [B, dim]
        diffusion_step = step_emb[:, None, :]   # [B, 1, dim]
        condition = cond
        for m in self.conditioner_projection:
            condition = m(condition)            # [B, T, dim]
        for layer in self.residual_layers:
            x = layer(x, condition, diffusion_step)
        for m in self.output_projection:
            x = m(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden: int, residual_channels: int,
                 dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels, kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2), dilation=dilation,
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def __call__(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step)[:, None, :]
        y = self.dilated_conv(x + diffusion_step) + self.conditioner_projection(conditioner)
        C = self.residual_channels
        gate, filt = y[..., :C], y[..., C:]
        y = self.output_projection(mx.sigmoid(gate) * mx.tanh(filt))
        residual, skip = y[..., :C], y[..., C:]
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.in_dims = in_dims
        self.n_chans = n_chans
        self.input_projection = nn.Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = [nn.Linear(n_chans, n_chans * 4), nn.Mish(),
                    nn.Linear(n_chans * 4, n_chans)]
        self.residual_layers = [
            ResidualBlock(encoder_hidden=n_hidden, residual_channels=n_chans,
                          dilation=dilation, kernel_size=kernel_size)
            for _ in range(n_layers)
        ]
        self.skip_projection = nn.Conv1d(n_chans, n_chans, 1)
        self.output_projection = nn.Conv1d(n_chans, in_dims, 1)
        # torch wavenet 构造时 output_projection 权重置 0(构造对齐)
        self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond):
        x = nn.relu(self.input_projection(spec))
        step_emb = self.diffusion_embedding(diffusion_step)  # [B, C]
        for m in self.mlp:
            step_emb = m(step_emb)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, step_emb)
            skip.append(skip_connection)
        x = mx.sum(mx.stack(skip), axis=0) / math.sqrt(len(self.residual_layers))
        x = nn.relu(self.skip_projection(x))
        return self.output_projection(x)


def _exists(x):
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if callable(d) else d


def _extract(a, t, x_shape):
    """a: [T] buffer;t: [B] int → 广播 [B,1,1,...](等价 torch gather+reshape)。"""
    b = t.shape[0]
    return mx.take(a, t, axis=0).reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM 采样核心。denoise_fn(x [B,T,M], t [B], cond [B,T,C]);全程 fp32。"""

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

        self.num_timesteps = int(timesteps)
        self.k_step = k_step
        self.noise_list = []

        self.betas = mx.array(betas, dtype=mx.float32)
        self.alphas_cumprod = mx.array(alphas_cumprod, dtype=mx.float32)
        self.alphas_cumprod_prev = mx.array(alphas_cumprod_prev, dtype=mx.float32)
        self.sqrt_alphas_cumprod = mx.array(np.sqrt(alphas_cumprod), dtype=mx.float32)
        self.sqrt_one_minus_alphas_cumprod = mx.array(np.sqrt(1.0 - alphas_cumprod), dtype=mx.float32)
        self.log_one_minus_alphas_cumprod = mx.array(np.log(1.0 - alphas_cumprod), dtype=mx.float32)
        self.sqrt_recip_alphas_cumprod = mx.array(np.sqrt(1.0 / alphas_cumprod), dtype=mx.float32)
        self.sqrt_recipm1_alphas_cumprod = mx.array(np.sqrt(1.0 / alphas_cumprod - 1), dtype=mx.float32)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = mx.array(posterior_variance, dtype=mx.float32)
        self.posterior_log_variance_clipped = mx.array(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=mx.float32)
        self.posterior_mean_coef1 = mx.array(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=mx.float32)
        self.posterior_mean_coef2 = mx.array(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=mx.float32)
        self.spec_min = mx.array(np.zeros((1, 1, out_dims)) + spec_min, dtype=mx.float32)
        self.spec_max = mx.array(np.zeros((1, 1, out_dims)) + spec_max, dtype=mx.float32)

    def q_sample(self, x_start, t, noise=None):
        noise = _default(noise, lambda: mx.random.normal(shape=x_start.shape, dtype=mx.float32))
        return (_extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (_extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def p_mean_variance(self, x, t, cond):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, cond=cond))
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (_extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                          + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = mx.random.normal(shape=x.shape, dtype=mx.float32)
        nonzero_mask = (1.0 - (t == 0).astype(mx.float32)).reshape(
            b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * mx.exp(0.5 * model_log_variance) * noise

    def p_sample_ddim(self, x, t, interval, cond):
        a_t = _extract(self.alphas_cumprod, t, x.shape)
        a_prev = _extract(self.alphas_cumprod, mx.maximum(t - interval, 0), x.shape)
        noise_pred = self.denoise_fn(x, t, cond=cond)
        return mx.sqrt(a_prev) * (
            x / mx.sqrt(a_t)
            + (mx.sqrt((1 - a_prev) / a_prev) - mx.sqrt((1 - a_t) / a_t)) * noise_pred
        )

    def p_sample_plms(self, x, t, interval, cond):
        """PLMS(伪数值方法):noise_list 长度 → 一阶/二阶/三阶外推系数。"""

        def get_x_pred(x, noise_t, t):
            a_t = _extract(self.alphas_cumprod, t, x.shape)
            a_prev = _extract(self.alphas_cumprod, mx.maximum(t - interval, 0), x.shape)
            a_t_sq, a_prev_sq = mx.sqrt(a_t), mx.sqrt(a_prev)
            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1 / (a_t_sq * (mx.sqrt((1 - a_prev) * a_t)
                                + mx.sqrt((1 - a_t) * a_prev))) * noise_t
            )
            return x + x_delta

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, mx.maximum(t - interval, 0), cond=cond)
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
        if len(noise_list) > 4:
            del noise_list[0]
        return x_prev

    def sample(self, condition, gt_spec=None, infer_speedup=10, method="ddim",
               k_step=None, use_tqdm=False):
        """condition [B,T,C] → denormalized mel [B,T,M]。"""
        b, T = condition.shape[0], condition.shape[1]
        cond = condition.astype(mx.float32)
        shape = (b, T, self.out_dims)

        if gt_spec is None or k_step is None:
            t = self.k_step
            x = mx.random.normal(shape, dtype=mx.float32)
        else:
            t = int(k_step)
            x = self.q_sample(x_start=self.norm_spec(gt_spec),
                              t=mx.array([t - 1], dtype=mx.int32))

        if method == "plms":
            self.noise_list = []
            for i in range(t - infer_speedup, -1, -infer_speedup):
                x = self.p_sample_plms(x, mx.full((b,), i, dtype=mx.int32), infer_speedup, cond=cond)
                mx.eval(x)
        elif method == "ddim":
            for i in range(t - infer_speedup, -1, -infer_speedup):
                x = self.p_sample_ddim(x, mx.full((b,), i, dtype=mx.int32), infer_speedup, cond=cond)
                mx.eval(x)  # 强制计算,释放上步中间结果(降峰值)
        elif method == "ddpm":
            for i in range(t - 1, -1, -1):
                x = self.p_sample(x, mx.full((b,), i, dtype=mx.int32), cond)
                mx.eval(x)
        else:
            raise NotImplementedError(method)

        return self.denorm_spec(x)

    def norm_spec(self, x):
        return _norm_spec(x, self.spec_min, self.spec_max, cast=True)

    def denorm_spec(self, x):
        return _denorm_spec(x, self.spec_min, self.spec_max)


class DiffusionModel(nn.Module):
    """Diffusion-SVC 风格 DDPM 模型(condition 嵌入 + denoiser + DDPM 采样)。"""

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

        self.unit_embed = nn.Linear(n_units, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False) if use_pitch_aug else None
        self.spk_embed = nn.Embedding(n_spk, n_hidden) if (n_spk is not None and n_spk > 1) else None

        if denoiser == "wavenet":
            denoise_fn = WaveNet(in_dims=out_dims, n_layers=wn_layers,
                                 n_chans=wn_chans, n_hidden=n_hidden)
        else:  # convnext(默认)
            denoise_fn = ConvNext(mel_channels=out_dims, dim=cn_chans,
                                  mlp_factor=mlp_factor, condition_dim=n_hidden,
                                  num_layers=cn_layers, dilation_cycle=4,
                                  gradient_checkpointing=gradient_checkpointing)
        self.decoder = GaussianDiffusion(
            denoise_fn, out_dims=out_dims, timesteps=timesteps, k_step=k_step,
            max_beta=max_beta, spec_min=spec_min, spec_max=spec_max)

    def _embed_condition(self, units, f0, volume, spk_id, aug_shift=None):
        units = units.astype(mx.float32)
        if f0.ndim == 2:
            f0 = f0[:, :, None]
        if volume.ndim == 2:
            volume = volume[:, :, None]
        x = (self.unit_embed(units) + self.f0_embed(mx.log(1 + f0.astype(mx.float32) / 700))
             + self.volume_embed(volume.astype(mx.float32)))
        if self.spk_embed is not None and spk_id is not None:
            spk = spk_id if spk_id.ndim > 1 else spk_id[:, None]  # usvc: 0-based id
            x = x + self.spk_embed(spk.astype(mx.int32))
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift.astype(mx.float32) / 5)
        return x

    def _denoise_wrapper(self, xt, t, cond):
        """mlx denoiser 已是 NLC [B,T,M](torch 需 transpose 到 [B,1,M,T])。"""
        return self.decoder.denoise_fn(xt, t, cond)

    def _objective(self):
        if getattr(self, "_ddpm_obj", None) is None:
            from ....train.mlx.objective import DDPMObjective
            self._ddpm_obj = DDPMObjective(
                spec_min=-12.0, spec_max=2.0,
                timesteps=self.decoder.num_timesteps, max_beta=0.02)
        return self._ddpm_obj

    def forward(self, units, f0, volume, spk_id, gt_spec, aug_shift=None, frame_len=None):
        """DDPM epsilon-prediction loss → (loss, {"eps_loss": loss})。"""
        from ....train.mlx.objective import lens_to_mask
        cond = self._embed_condition(units, f0, volume, spk_id, aug_shift)
        mask = (lens_to_mask(frame_len, length=gt_spec.shape[1])
                if frame_len is not None else None)
        loss = self._objective().compute_loss(gt_spec, self._denoise_wrapper, cond,
                                              t_max=self.k_step, mask=mask)
        return loss, {"eps_loss": loss}

    def sample(self, units, f0, volume, spk_id, aug_shift=None, gt_spec=None,
               infer_speedup=10, method="ddim", k_step=None, use_tqdm=False):
        """→ denormalized mel [B, T, M](gt_spec 时浅扩散 k_step<1000)。"""
        x = self._embed_condition(units, f0, volume, spk_id, aug_shift)
        return self.decoder.sample(x, gt_spec=gt_spec, infer_speedup=infer_speedup,
                                   method=method, k_step=k_step, use_tqdm=use_tqdm)

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, key_prefix: str | None = "model",
                  strict: bool = False, dtype=None, k_step: int | None = None):
        """键形如 model.unit_embed.*;支持 'module.' 或无前缀(回退顺序保留)。"""
        from .._convert import load_torch_ckpt_to_mlx, strip_prefix

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype or mx.float32)
        if key_prefix:
            stripped = strip_prefix(sd, key_prefix + ".")
            sd = stripped if stripped else sd  # 前缀不匹配 → 裸 sd 回退
        cfg = cls._infer_config(sd)
        if k_step is not None:
            cfg["k_step"] = int(k_step)
        obj = cls(**cfg)
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731

        uw = shape("unit_embed.weight")
        if uw is None:
            raise ValueError("diffusion ckpt missing unit_embed.weight")
        n_units, n_hidden = uw[1], uw[0]
        sw = shape("spk_embed.weight")
        n_spk = int(sw[0]) if sw is not None else 1
        use_pitch_aug = shape("aug_shift_embed.weight") is not None

        # denoiser 判定:residual_layers.0 有 dilated_conv → wavenet;dwconv → convnext
        if shape("decoder.denoise_fn.residual_layers.0.dilated_conv.weight") is not None:
            in_w = shape("decoder.denoise_fn.input_projection.weight")
            n = 0
            while shape(f"decoder.denoise_fn.residual_layers.{n}.dilated_conv.weight") is not None:
                n += 1
            return dict(n_units=n_units, n_spk=n_spk, out_dims=int(in_w[2]),
                        n_hidden=n_hidden, use_pitch_aug=use_pitch_aug,
                        k_step=1000, denoiser="wavenet",
                        wn_layers=n, wn_chans=int(in_w[0]))
        op = shape("decoder.denoise_fn.output_projection.2.weight")
        if op is None:
            raise ValueError("diffusion ckpt has neither wavenet nor convnext denoiser keys")
        p1 = shape("decoder.denoise_fn.residual_layers.0.pwconv1.weight")
        n = 0
        while shape(f"decoder.denoise_fn.residual_layers.{n}.dwconv.weight") is not None:
            n += 1
        return dict(n_units=n_units, n_spk=n_spk, out_dims=int(op[0]),
                    n_hidden=n_hidden, use_pitch_aug=use_pitch_aug,
                    k_step=1000, denoiser="convnext", cn_layers=n,
                    cn_chans=int(op[2]),
                    mlp_factor=int(p1[0] // p1[1]) if p1 is not None else 4)