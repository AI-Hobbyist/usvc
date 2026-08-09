"""MLX 版 diffusion backend(mel-domain DDPM,镜像 ``usvc/backends/diffusion/model.py``)。

推理最小集:
  - WaveNet denoiser(weight_norm 由加载时合并)+ ConvNext denoiser
  - DiffusionEmbedding / SinusoidalPosEmb(正弦 step 嵌入 + MLP)
  - GaussianDiffusion DDPM sampler(ddim / plms / ddpm)
  - DiffusionModel(condition 嵌入 + denoiser + sampler 组合)

关键差异(torch → mlx,见 MLX 映射手册):
  - 输入布局:torch denoiser 是 [B,1,M,T](NCL),mlx 全程 NLC [B,T,M]。
    spec/cond 不转置,Conv1d 原生 NLC。
  - weight_norm 权重由 ``_convert.load_torch_ckpt_to_mlx`` 加载时合并
    (torch wavenet/convnext 实际无 weight_norm,但合并逻辑幂等)。
  - ModuleList/Sequential → 普通 list(mlx load_weights 键名 ``name.0.weight``
    与 torch 一致)。
  - gelu() 无 approx 参数(mlx 恒为精确,torch 推理也用精确,数值一致)。
  - 扩散采样全程 fp32(DDPM/PLMS 迭代数值敏感,engine.py:226 注)。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

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


# ---------------------------------------------------------------------------
# 正弦 step 嵌入(两种命名,torch 两个文件各一份,数学相同)
# ---------------------------------------------------------------------------

class DiffusionEmbedding(nn.Module):
    """Diffusion step 正弦嵌入(镜像 model.py::DiffusionEmbedding)。"""

    def __init__(self, d_denoiser: int):
        super().__init__()
        self.dim = d_denoiser
        half_dim = d_denoiser // 2
        self.half_dim = half_dim
        # 预计算频率:exp(arange(half_dim) * -log(10000)/(half_dim-1))
        emb = math.log(10000.0) / (half_dim - 1)
        self.freqs = mx.exp(mx.arange(half_dim) * -emb)  # [half_dim] fp32

    def __call__(self, x):
        """x: [B] step(int 或 float)→ [B, dim] 正弦嵌入。"""
        x = x.astype(mx.float32)
        x = x[:, None] * self.freqs[None, :]          # [B, half_dim]
        return mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)


class SinusoidalPosEmb(nn.Module):
    """正弦位置嵌入(镜像 wavenet.py::SinusoidalPosEmb)。"""

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


# ---------------------------------------------------------------------------
# ConvNext denoiser(镜像 model.py::ConvNeXtBlock / ConvNext)
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """ConvNeXt 1D block(NLC 版,镜像 model.py::ConvNeXtBlock)。

    dwconv 走原生 mlx Conv1d(groups=dim),不用 triton(MLX 无 CUDA 内核)。
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        # depthwise conv1d:kernel 7,groups=dim,padding=3*dilation(same)
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, groups=dim, dilation=dilation,
            padding=int(dilation * (7 - 1) / 2),
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        if layer_scale_init_value is not None and layer_scale_init_value > 0:
            self.gamma = mx.full((dim,), float(layer_scale_init_value))
        else:
            self.gamma = None
        self.diffusion_step_projection = nn.Conv1d(dim, dim, 1)
        self.condition_projection = nn.Conv1d(dim, dim, 1)

    def __call__(self, x, condition, diffusion_step):
        """x/condition: [B, T, dim];diffusion_step: [B, 1, dim]。"""
        residual = x
        x = (
            x
            + self.diffusion_step_projection(diffusion_step)
            + self.condition_projection(condition)
        )
        x = self.dwconv(x)          # [B, T, dim](NLC,dwconv 输出同布局)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return residual + x


class ConvNext(nn.Module):
    """ConvNext denoiser。输入/输出 [B, T, M](NLC)。

    镜像 model.py::ConvNext;布局从 NCL 换成 NLC,数学逐位一致。
    """

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
        # [DiffusionEmbedding, Linear(dim, dim*4), GELU, Linear(dim*4, dim)]
        # 键名 diffusion_embedding.{0,1,2,3} 与 torch Sequential 一致
        self.diffusion_embedding = [
            DiffusionEmbedding(dim),
            nn.Linear(dim, dim * mlp_factor),
            nn.GELU(),
            nn.Linear(dim * mlp_factor, dim),
        ]
        # conditioner_projection:{Conv1d, GELU, Conv1d}(键名 0/2)
        self.conditioner_projection = [
            nn.Conv1d(condition_dim, dim * mlp_factor, 1),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        ]
        self.residual_layers = [
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=dim * mlp_factor,
                dilation=2 ** (i % dilation_cycle),
            )
            for i in range(num_layers)
        ]
        # output_projection:{Conv1d, GELU, Conv1d}(键名 0/2)
        self.output_projection = [
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, mel_channels, kernel_size=1),
        ]
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, spec, diffusion_step, cond):
        """spec: [B, T, M];diffusion_step: [B] step;cond: [B, T, condition_dim]。

        Returns [B, T, M]。
        """
        x = self.input_projection(spec)      # [B, T, dim]
        x = nn.gelu(x)
        step_emb = self.diffusion_embedding[0](diffusion_step)   # [B, dim]
        step_emb = self.diffusion_embedding[1](step_emb)
        step_emb = self.diffusion_embedding[2](step_emb)
        step_emb = self.diffusion_embedding[3](step_emb)
        diffusion_step = step_emb[:, None, :]                    # [B, 1, dim]
        condition = self.conditioner_projection[0](cond)
        condition = self.conditioner_projection[1](condition)
        condition = self.conditioner_projection[2](condition)    # [B, T, dim]

        for layer in self.residual_layers:
            x = layer(x, condition, diffusion_step)

        x = self.output_projection[0](x)
        x = self.output_projection[1](x)
        x = self.output_projection[2](x)     # [B, T, M]
        return x


# ---------------------------------------------------------------------------
# WaveNet denoiser(镜像 wavenet.py;Diffusion-SVC-v1 / so-vits 浅扩散)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """WaveNet 残差块(NLC 版,镜像 wavenet.py::ResidualBlock)。

    dilated conv + gated tanh/sigmoid + skip projection。torch 版所有层
    dilation=1(kernel=3),参数化支持任意 kernel/dilation。
    """

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
        """x: [B, T, C];conditioner: [B, T, encoder_hidden];
        diffusion_step: [B, C](mlp 输出)→ (x', skip) [B, T, C]。"""
        diffusion_step = self.diffusion_projection(diffusion_step)[:, None, :]  # [B,1,C]
        conditioner = self.conditioner_projection(conditioner)   # [B, T, 2C]
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner                    # [B, T, 2C]
        C = self.residual_channels
        gate, filt = y[..., :C], y[..., C:]                       # [B, T, C] each
        y = mx.sigmoid(gate) * mx.tanh(filt)                      # gated
        y = self.output_projection(y)                             # [B, T, 2C]
        residual, skip = y[..., :C], y[..., C:]
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    """WaveNet denoiser(NLC 版,镜像 wavenet.py::WaveNet)。

    forward(spec [B,T,M], diffusion_step [B], cond [B,T,n_hidden]) -> [B,T,M]
    """

    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.in_dims = in_dims
        self.n_chans = n_chans
        self.input_projection = nn.Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        # [Linear(n_chans, n_chans*4), Mish, Linear(n_chans*4, n_chans)]
        self.mlp = [
            nn.Linear(n_chans, n_chans * 4),
            nn.Mish(),
            nn.Linear(n_chans * 4, n_chans),
        ]
        self.residual_layers = [
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=dilation,
                kernel_size=kernel_size,
            )
            for _ in range(n_layers)
        ]
        self.skip_projection = nn.Conv1d(n_chans, n_chans, 1)
        self.output_projection = nn.Conv1d(n_chans, in_dims, 1)
        # torch wavenet 把 output_projection 初始化为 0(与 ckpt 无关,构造对齐)
        # mlx:直接置零 weight(训练/随机初始化场景数值一致)
        self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond):
        """spec: [B, T, M];diffusion_step: [B] step;cond: [B, T, n_hidden]。

        Returns [B, T, M]。
        """
        x = self.input_projection(spec)      # [B, T, C]
        x = nn.relu(x)
        step_emb = self.diffusion_embedding(diffusion_step)   # [B, C]
        step_emb = self.mlp[0](step_emb)
        step_emb = self.mlp[1](step_emb)
        step_emb = self.mlp[2](step_emb)                      # [B, C]

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, step_emb)
            skip.append(skip_connection)

        x = mx.sum(mx.stack(skip), axis=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = nn.relu(x)
        x = self.output_projection(x)        # [B, T, M]
        return x


# ---------------------------------------------------------------------------
# DDPM sampler(镜像 model.py::GaussianDiffusion;dpm-solver/unipc 移除)
# ---------------------------------------------------------------------------

def _exists(x):
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if callable(d) else d


def _extract(a, t, x_shape):
    """a: [T] buffer;t: [B] int 索引 → 广播形状 [B,1,1,...]。

    与 torch ``a.gather(-1, t).reshape(b, 1*(len(x_shape)-1))`` 等价。
    """
    b = t.shape[0]
    out = mx.take(a, t, axis=0)               # [B]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM 采样核心(NLC 版)。denoise_fn 签名 (x [B,T,M], t [B], cond [B,T,C])。

    全程 fp32:DDPM/PLMS 迭代数值敏感。
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

        self.num_timesteps = int(timesteps)
        self.k_step = k_step
        self.noise_list = []

        def _buf(a):
            return mx.array(a, dtype=mx.float32)

        self.betas = _buf(betas)
        self.alphas_cumprod = _buf(alphas_cumprod)
        self.alphas_cumprod_prev = _buf(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = _buf(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = _buf(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = _buf(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = _buf(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = _buf(np.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = _buf(posterior_variance)
        self.posterior_log_variance_clipped = _buf(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = _buf(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = _buf(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.spec_min = _buf(np.zeros((1, 1, out_dims)) + spec_min)
        self.spec_max = _buf(np.zeros((1, 1, out_dims)) + spec_max)

    # -- helpers ------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        noise = _default(noise, lambda: mx.random.normal(shape=x_start.shape, dtype=mx.float32))
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

    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = mx.random.normal(shape=x.shape, dtype=mx.float32)
        nonzero_mask = (1.0 - (t == 0).astype(mx.float32)).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * mx.exp(0.5 * model_log_variance) * noise

    def p_sample_ddim(self, x, t, interval, cond):
        a_t = _extract(self.alphas_cumprod, t, x.shape)
        a_prev = _extract(
            self.alphas_cumprod, mx.maximum(t - interval, 0), x.shape
        )
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = mx.sqrt(a_prev) * (
            x / mx.sqrt(a_t)
            + (mx.sqrt((1 - a_prev) / a_prev) - mx.sqrt((1 - a_t) / a_t))
            * noise_pred
        )
        return x_prev

    def p_sample_plms(self, x, t, interval, cond):
        """PLMS 采样(Pseudo Numerical Methods for Diffusion Models)。"""

        def get_x_pred(x, noise_t, t):
            a_t = _extract(self.alphas_cumprod, t, x.shape)
            a_prev = _extract(
                self.alphas_cumprod, mx.maximum(t - interval, 0), x.shape
            )
            a_t_sq, a_prev_sq = mx.sqrt(a_t), mx.sqrt(a_prev)
            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1
                / (a_t_sq * (mx.sqrt((1 - a_prev) * a_t) + mx.sqrt((1 - a_t) * a_prev)))
                * noise_t
            )
            return x + x_delta

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(
                x_pred, mx.maximum(t - interval, 0), cond=cond
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
        if len(noise_list) > 4:
            del noise_list[0]
        return x_prev

    def sample(
        self,
        condition,
        gt_spec=None,
        infer_speedup=10,
        method="ddim",
        k_step=None,
        use_tqdm=False,
    ):
        """condition [B, T, C] → denormalized mel [B, T, M]。全程 fp32。"""
        b, T = condition.shape[0], condition.shape[1]
        cond = condition.astype(mx.float32)
        shape = (b, T, self.out_dims)

        if gt_spec is None or k_step is None:
            t = self.k_step
            x = mx.random.normal(shape, dtype=mx.float32)
        else:
            t = int(k_step)
            norm_spec = self.norm_spec(gt_spec)          # [B, T, M]
            x = self.q_sample(
                x_start=norm_spec,
                t=mx.array([t - 1], dtype=mx.int32),
            )

        if method == "plms":
            self.noise_list = []
            for i in range(t - infer_speedup, -1, -infer_speedup):
                x = self.p_sample_plms(
                    x, mx.full((b,), i, dtype=mx.int32),
                    infer_speedup, cond=cond,
                )
                mx.eval(x)
        elif method == "ddim":
            for i in range(t - infer_speedup, -1, -infer_speedup):
                x = self.p_sample_ddim(
                    x, mx.full((b,), i, dtype=mx.int32),
                    infer_speedup, cond=cond,
                )
                mx.eval(x)  # 强制计算,释放上步中间结果(降峰值)
        elif method == "ddpm":
            for i in range(t - 1, -1, -1):
                x = self.p_sample(
                    x, mx.full((b,), i, dtype=mx.int32), cond
                )
                mx.eval(x)
        else:
            raise NotImplementedError(method)

        return self.denorm_spec(x)                        # [B, T, M]

    # -- 归一化(与 torch 采样器一致)-------------------------------------

    def norm_spec(self, x):
        x = x.astype(mx.float32)
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# ---------------------------------------------------------------------------
# DiffusionModel(镜像 model.py::DiffusionModel;推理只做 sample)
# ---------------------------------------------------------------------------

class DiffusionModel(nn.Module):
    """Diffusion-SVC 风格 DDPM 模型(mlx 推理集)。

    condition: units/f0/volume 嵌入(+ 可选 spk / pitch aug),去噪器 +
    DDPM 采样。训练 forward 不实现(mlx 只服务推理)。
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

        # condition 嵌入(Unit2MelV2)
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

        # denoiser
        if denoiser == "wavenet":
            denoise_fn = WaveNet(
                in_dims=out_dims,
                n_layers=wn_layers,
                n_chans=wn_chans,
                n_hidden=n_hidden,
            )
        else:  # convnext(默认)
            denoise_fn = ConvNext(
                mel_channels=out_dims,
                dim=cn_chans,
                mlp_factor=mlp_factor,
                condition_dim=n_hidden,
                num_layers=cn_layers,
                dilation_cycle=4,
                gradient_checkpointing=gradient_checkpointing,
            )

        # sampler(推理)
        self.decoder = GaussianDiffusion(
            denoise_fn,
            out_dims=out_dims,
            timesteps=timesteps,
            k_step=k_step,
            max_beta=max_beta,
            spec_min=spec_min,
            spec_max=spec_max,
        )

    def _embed_condition(self, units, f0, volume, spk_id, aug_shift=None):
        """→ [B, T, n_hidden] condition 嵌入(镜像 torch _embed_condition)。"""
        units = units.astype(mx.float32)
        if f0.ndim == 2:
            f0 = f0[:, :, None]
        if volume.ndim == 2:
            volume = volume[:, :, None]
        f0 = f0.astype(mx.float32)
        volume = volume.astype(mx.float32)

        x = (
            self.unit_embed(units)
            + self.f0_embed(mx.log(1 + f0 / 700))
            + self.volume_embed(volume)
        )
        if self.spk_embed is not None and spk_id is not None:
            # usvc 约定:0-based speaker id。spk_id [B] → [B, 1]
            spk = spk_id if spk_id.ndim > 1 else spk_id[:, None]
            x = x + self.spk_embed(spk.astype(mx.int32))
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift.astype(mx.float32) / 5)
        return x

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
        """units [B,T,D], f0 [B,T] 或 [B,T,1], volume 同 f0, spk_id [B]。

        gt_spec [B,T,M] 可选(部分加噪,浅扩散 k_step<1000)。
        Returns denormalized mel [B, T, M](fp32)。
        """
        x = self._embed_condition(units, f0, volume, spk_id, aug_shift)
        return self.decoder.sample(
            x,
            gt_spec=gt_spec,
            infer_speedup=infer_speedup,
            method=method,
            k_step=k_step,
            use_tqdm=use_tqdm,
        )

    # -- ckpt 加载 ----------------------------------------------------------

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str | Path,
        *,
        key_prefix: str | None = "model",
        strict: bool = False,
        dtype=None,
        k_step: int | None = None,
    ):
        """从 torch diffusion ckpt 加载(推理)。

        权重键形如 ``model.unit_embed.weight`` / ``model.decoder.denoise_fn.*``
        (lightning wrapper)。也兼容 ``module.`` 前缀或无前缀。

        Args:
            ckpt_path: torch ckpt。
            key_prefix: ckpt 顶层前缀('model'|'module'|None)。
            strict: 是否严格匹配(默认 False,容忍非持久 buffer 缺失)。
            dtype: 目标精度(默认 fp32;扩散采样必须 fp32)。
            k_step: 采样步数上限。None 时从 ckpt 反推(优先 hyper_parameters/
                ``decoder.betas`` 长度,失败回退 1000)。
        """
        from .._convert import load_torch_ckpt_to_mlx

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype or mx.float32)
        if key_prefix:
            prefix = key_prefix + "."
            stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not stripped:
                # 前缀不匹配,尝试无前缀(裸 DiffusionModel sd)
                stripped = sd
            sd = stripped
        cfg = cls._infer_config(sd)
        if k_step is not None:
            cfg["k_step"] = int(k_step)
        obj = cls(**cfg)
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        """从权重 shape 反推 DiffusionModel 构造参数。"""
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731

        uw = shape("unit_embed.weight")                     # (n_hidden, n_units)
        if uw is None:
            raise ValueError("diffusion ckpt missing unit_embed.weight")
        n_units, n_hidden = uw[1], uw[0]

        sw = shape("spk_embed.weight")                      # (n_spk, n_hidden)
        n_spk = int(sw[0]) if sw is not None else 1

        aew = shape("aug_shift_embed.weight")
        use_pitch_aug = aew is not None

        # denoiser 类型:residual_layers.0 有 dilated_conv → wavenet,有 dwconv → convnext
        wn0 = shape("decoder.denoise_fn.residual_layers.0.dilated_conv.weight")
        if wn0 is not None:
            denoiser = "wavenet"
            in_w = shape("decoder.denoise_fn.input_projection.weight")  # (C,1,M)
            wn_chans = int(in_w[0])
            out_dims = int(in_w[2])
            n = 0
            while shape(f"decoder.denoise_fn.residual_layers.{n}.dilated_conv.weight") is not None:
                n += 1
            wn_layers = n
            return dict(
                n_units=n_units, n_spk=n_spk, out_dims=out_dims,
                n_hidden=n_hidden, use_pitch_aug=use_pitch_aug,
                k_step=1000, denoiser="wavenet",
                wn_layers=wn_layers, wn_chans=wn_chans,
            )
        op = shape("decoder.denoise_fn.output_projection.2.weight")    # (M,1,dim)
        if op is None:
            raise ValueError(
                "diffusion ckpt has neither wavenet nor convnext denoiser keys"
            )
        out_dims = int(op[0])
        cn_chans = int(op[2])
        p1 = shape("decoder.denoise_fn.residual_layers.0.pwconv1.weight")  # (4d,d)
        mlp_factor = int(p1[0] // p1[1]) if p1 is not None else 4
        n = 0
        while shape(f"decoder.denoise_fn.residual_layers.{n}.dwconv.weight") is not None:
            n += 1
        cn_layers = n
        return dict(
            n_units=n_units, n_spk=n_spk, out_dims=out_dims,
            n_hidden=n_hidden, use_pitch_aug=use_pitch_aug,
            k_step=1000, denoiser="convnext",
            cn_layers=cn_layers, cn_chans=cn_chans, mlp_factor=mlp_factor,
        )
