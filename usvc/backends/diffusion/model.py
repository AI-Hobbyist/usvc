"""Diffusion-SVC style DDPM model for usvc (convnext denoiser + GaussianDiffusion
sampler; extracted from Diffusion-SVC diffusion/convnext.py + diffusion/diffusion.py,
fish-diffusion derived). dpm-solver/unipc removed — ddim/plms/ddpm retained.

Training loss reuses usvc.train.objective.DDPMObjective — math identical to
Diffusion-SVC's GaussianDiffusion.p_losses: betas=linspace(1e-4,0.02,1000),
x0=norm(mel) (spec_min=-12, spec_max=2 -> [-1,1]), t~randint(0,1000),
noise=randn_like(x0), xt=sqrt(acp[t])*x0+sqrt(1-acp[t])*noise,
loss=MSE(denoise_fn(xt,t,cond),noise). Only layout differs: p_losses uses
[B,1,M,T], DDPMObjective [B,T,M] (transposed in _denoise_wrapper), so the
loss is numerically identical.
"""
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

from ...nn.normalization import MelNormalizer
from ...train.objective import DDPMObjective

try:
    from usvc._triton.dwconv1d import dwconv1d as _triton_dwconv1d
    _HAS_TRITON_DWCONV = True
except Exception:
    _HAS_TRITON_DWCONV = False


def _dwconv_apply(x, weight, bias, dilation):
    # Triton depthwise conv1d（bit 级等价，fwd ~2.3×），PyTorch conv1d 兜底；
    # contiguous() 满足 CUDA-graph 捕获布局检查（compile-friendly custom op）
    if _HAS_TRITON_DWCONV:
        try:
            return _triton_dwconv1d(x.contiguous(), weight, bias, dilation)
        except Exception:
            pass
    return F.conv1d(x, weight, bias, groups=weight.shape[0],
                    dilation=dilation, padding=dilation * (weight.shape[2] - 1) // 2)


class DiffusionEmbedding(nn.Module):
    def __init__(self, d_denoiser):
        super().__init__()
        self.dim = d_denoiser

    def forward(self, x):
        hd = self.dim // 2
        dt = x.dtype if x.is_floating_point else torch.float32
        emb = torch.exp(torch.arange(hd, device=x.device, dtype=dt)
                        * -(math.log(10000) / (hd - 1)))
        emb = x[:, None].to(dt) * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, dilation=1, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, groups=dim, dilation=dilation,
                                padding=int(dilation * (7 - 1) / 2))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim))
                      if layer_scale_init_value is not None and layer_scale_init_value > 0
                      else None)
        self.diffusion_step_projection = nn.Conv1d(dim, dim, 1)
        self.condition_projection = nn.Conv1d(dim, dim, 1)

    def forward(self, x, condition=None, diffusion_step=None):
        residual = x
        x = x + self.diffusion_step_projection(diffusion_step) \
              + self.condition_projection(condition)
        if getattr(self, "_use_triton_dwconv", False):
            x = _dwconv_apply(x, self.dwconv.weight, self.dwconv.bias, self.dwconv.dilation[0])
        else:
            x = self.dwconv(x)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x.transpose(1, 2)))))  # (B,C,T)->(B,T,C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B,T,C)->(B,C,T)
        return residual + x


class ConvNext(nn.Module):  # ConvNeXt denoiser；输入/输出 [B,M,T]（支持 [B,1,M,T]）
    def __init__(self, mel_channels=128, dim=512, mlp_factor=4, condition_dim=256,
                 num_layers=20, dilation_cycle=4, gradient_checkpointing=False):
        super().__init__()
        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = nn.Sequential(
            DiffusionEmbedding(dim), nn.Linear(dim, dim * mlp_factor),
            nn.GELU(), nn.Linear(dim * mlp_factor, dim))
        self.conditioner_projection = nn.Sequential(
            nn.Conv1d(condition_dim, dim * mlp_factor, 1), nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1))
        self.residual_layers = nn.ModuleList(
            ConvNeXtBlock(dim=dim, intermediate_dim=dim * mlp_factor,
                          dilation=2 ** (i % dilation_cycle))
            for i in range(num_layers))
        self.output_projection = nn.Sequential(
            nn.Conv1d(dim, dim, 1), nn.GELU(), nn.Conv1d(dim, mel_channels, 1))
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, spec, diffusion_step, cond):
        use_4_dim = spec.dim() == 4  # DiffSVC 兼容 [B,1,M,T]
        if use_4_dim:
            spec = spec[:, 0]
        assert spec.dim() == 3, f"mel must be 3 dim tensor, but got {spec.dim()}"
        x = F.gelu(self.input_projection(spec))
        # diffusion_step 为 long；embedding 固定 f32 计算 → 显式 cast 到 x.dtype，
        # 让后续 Linear 保持 half/bf16 推理 dtype
        ds = self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(-1)
        cond = self.conditioner_projection(cond)
        for l in self.residual_layers:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(l, x, cond, ds)
            else:
                x = l(x, cond, ds)
        x = self.output_projection(x)
        return x[:, None] if use_4_dim else x


def _extract(a, t, x_shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def _noise_like(shape, device, repeat=False):
    if repeat:
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1)))
    return torch.randn(shape, device=device)


class GaussianDiffusion(nn.Module):  # DDPM 采样核心（训练 loss 在 DDPMObjective）
    def __init__(self, denoise_fn, out_dims=128, timesteps=1000, k_step=1000,
                 max_beta=0.02, spec_min=-12, spec_max=2):
        super().__init__()
        self.denoise_fn, self.out_dims = denoise_fn, out_dims
        # f64 预计算再 f32 落盘：与 np.linspace(1e-4,0.02)/cumprod/log 参考逐位一致（已验）
        b = torch.linspace(1e-4, max_beta, timesteps, dtype=torch.float64)
        ac = torch.cumprod(1.0 - b, 0)
        acp = torch.cat((b.new_tensor([1.0]), ac[:-1]))
        pv = b * (1.0 - acp) / (1.0 - ac)  # posterior variance
        for k, v in dict(
            betas=b.float(),
            alphas_cumprod=ac.float(),
            alphas_cumprod_prev=acp.float(),
            sqrt_alphas_cumprod=ac.sqrt().float(),
            sqrt_one_minus_alphas_cumprod=(1.0 - ac).sqrt().float(),
            log_one_minus_alphas_cumprod=(1.0 - ac).log().float(),
            sqrt_recip_alphas_cumprod=ac.reciprocal().sqrt().float(),
            sqrt_recipm1_alphas_cumprod=(1.0 / ac - 1).sqrt().float(),
            posterior_variance=pv.float(),
            posterior_log_variance_clipped=pv.clamp(min=1e-20).log().float(),
            posterior_mean_coef1=(b * acp.sqrt() / (1.0 - ac)).float(),
            posterior_mean_coef2=((1.0 - acp) * (1.0 - b).sqrt() / (1.0 - ac)).float(),
        ).items():
            self.register_buffer(k, v)
        self.num_timesteps = timesteps
        self.k_step = k_step
        self.noise_list = deque(maxlen=4)  # PLMS 历史噪声
        self.register_buffer("spec_min", torch.FloatTensor([spec_min])[None, None, :out_dims])
        self.register_buffer("spec_max", torch.FloatTensor([spec_max])[None, None, :out_dims])

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (_extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (_extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        return (_extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t,
                _extract(self.posterior_variance, t, x_t.shape),
                _extract(self.posterior_log_variance_clipped, t, x_t.shape))

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        return self.q_posterior(x_start=self.predict_start_from_noise(x, t=t, noise=noise_pred),
                                x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = _noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        a_t = _extract(self.alphas_cumprod, t, x.shape).to(x.dtype)
        a_prev = _extract(self.alphas_cumprod,
                          torch.max(t - interval, torch.zeros_like(t)), x.shape).to(x.dtype)
        noise_pred = self.denoise_fn(x, t, cond=cond)
        return a_prev.sqrt() * (x / a_t.sqrt()
                                + (((1 - a_prev) / a_prev).sqrt() - ((1 - a_t) / a_t).sqrt())
                                * noise_pred)

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        def get_x_pred(x, noise_t, t):
            a_t = _extract(self.alphas_cumprod, t, x.shape).to(x.dtype)
            a_prev = _extract(self.alphas_cumprod,
                              torch.max(t - interval, torch.zeros_like(t)), x.shape).to(x.dtype)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
                * noise_t)
            return x + x_delta

        noise_pred = self.denoise_fn(x, t, cond=cond)
        nl = self.noise_list
        if len(nl) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(
                x_pred, torch.maximum(t - interval, torch.zeros_like(t)), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(nl) == 1:
            noise_pred_prime = (3 * noise_pred - nl[-1]) / 2
        elif len(nl) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * nl[-1] + 5 * nl[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * nl[-1] + 37 * nl[-2] - 9 * nl[-3]) / 24
        x_prev = get_x_pred(x, noise_pred_prime, t)
        nl.append(noise_pred)
        return x_prev

    @torch.no_grad()
    def sample(self, condition, gt_spec=None, infer_speedup=10, method="ddim",
               k_step=None, use_tqdm=False):
        cond = condition.transpose(1, 2)  # [B,T,C] -> [B,C,T]
        b, device = condition.shape[0], condition.device
        shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])
        if gt_spec is None or k_step is None:
            t, x = self.k_step, torch.randn(shape, device=device, dtype=condition.dtype)
        else:
            t = k_step
            x = self.q_sample(x_start=self.norm_spec(gt_spec).transpose(1, 2)[:, None, :, :],
                              t=torch.tensor([t - 1], device=device).long())
        bar = None
        if use_tqdm:
            from tqdm import tqdm
            bar = tqdm(desc="sample time step", total=t // infer_speedup)
        if method == "plms":
            self.noise_list = deque(maxlen=4)
            for i in reversed(range(0, t, infer_speedup)):
                x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long),
                                       infer_speedup, cond=cond)
                if bar: bar.update(1)
        elif method == "ddim":
            for i in reversed(range(0, t, infer_speedup)):
                x = self.p_sample_ddim(x, torch.full((b,), i, device=device, dtype=torch.long),
                                       infer_speedup, cond=cond)
                if bar: bar.update(1)
        elif method == "ddpm":
            for i in reversed(range(0, t)):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
                if bar: bar.update(1)
        else:
            raise NotImplementedError(method)
        if bar: bar.close()
        return self.denorm_spec(x.squeeze(1).transpose(1, 2))  # [B,1,M,T]->[B,T,M]

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class DiffusionModel(nn.Module):  # Unit2MelV2 提取：cond embeds + denoiser + DDPM loss
    def __init__(self, n_units=768, n_spk=1, out_dims=128, n_hidden=256,
                 use_pitch_aug=False, timesteps=1000, max_beta=0.02,
                 spec_min=-12, spec_max=2, cn_layers=20, cn_chans=384,
                 mlp_factor=4, gradient_checkpointing=False, k_step=1000,
                 denoiser="convnext", wn_layers=20, wn_chans=384):
        super().__init__()
        self.n_spk, self.out_dims, self.n_hidden = n_spk, out_dims, n_hidden
        self.use_pitch_aug, self.k_step = use_pitch_aug, k_step
        self.unit_embed = nn.Linear(n_units, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False) if use_pitch_aug else None
        self.spk_embed = nn.Embedding(n_spk, n_hidden) if (n_spk is not None and n_spk > 1) else None
        if denoiser == "wavenet":
            from .wavenet import WaveNet
            denoise_fn = WaveNet(in_dims=out_dims, n_layers=wn_layers,
                                 n_chans=wn_chans, n_hidden=n_hidden)
        else:  # convnext（默认）
            denoise_fn = ConvNext(mel_channels=out_dims, dim=cn_chans, mlp_factor=mlp_factor,
                                  condition_dim=n_hidden, num_layers=cn_layers,
                                  dilation_cycle=4,
                                  gradient_checkpointing=gradient_checkpointing)
        self.decoder = GaussianDiffusion(denoise_fn, out_dims=out_dims, timesteps=timesteps,
                                         k_step=k_step, max_beta=max_beta,
                                         spec_min=spec_min, spec_max=spec_max)
        self.objective = DDPMObjective(MelNormalizer(spec_min, spec_max),
                                       timesteps=timesteps, max_beta=max_beta)
        self.objective.register_buffers(self)

    def _embed_condition(self, units, f0, volume, spk_id, aug_shift):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume.dim() == 2:
            volume = volume.unsqueeze(-1)
        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.spk_embed is not None and spk_id is not None:
            # spk_id [B] -> [B,1]：embedding([B]) 只有 T==B 时才偶然对齐
            spk = spk_id if spk_id.dim() > 1 else spk_id.unsqueeze(1)
            x = x + self.spk_embed(spk)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        return x

    def _denoise_wrapper(self, xt, t, cond):  # [B,T,M] <-> [B,1,M,T] 布局适配
        return self.decoder.denoise_fn(xt.transpose(1, 2).unsqueeze(1), t, cond)[:, 0].transpose(1, 2)

    def forward(self, units, f0, volume, spk_id, gt_spec, aug_shift=None, frame_len=None):
        cond = self._embed_condition(units, f0, volume, spk_id, aug_shift).transpose(1, 2)  # [B,T,H]->[B,H,T]
        mask = None
        if frame_len is not None:
            from ...train.objective import lens_to_mask
            # frame 掩码：padding mel=0 是响亮的非静音目标，会污染梯度
            mask = lens_to_mask(frame_len, length=gt_spec.shape[1]).to(gt_spec.device)  # frame_len 可能在 CPU
        loss = self.objective.compute_loss(gt_spec, self._denoise_wrapper, cond,
                                           t_max=self.k_step, mask=mask)
        return loss, {"eps_loss": loss.item()}

    @torch.no_grad()
    def sample(self, units, f0, volume, spk_id, aug_shift=None, gt_spec=None,
               infer_speedup=10, method="ddim", k_step=None, use_tqdm=False):
        return self.decoder.sample(self._embed_condition(units, f0, volume, spk_id, aug_shift),
                                   gt_spec=gt_spec, infer_speedup=infer_speedup,
                                   method=method, k_step=k_step, use_tqdm=use_tqdm)