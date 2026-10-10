"""ddsp6 backend model: CombSubSuperFast + velocity net + RectifiedFlow.

Extracted from DDSP-SVC-6.1/6.2/6.3: reflow/vocoder.py -> Unit2Wav,
reflow/reflow.py -> RectifiedFlow, reflow/lynxnet.py -> LYNXNet (6.1/6.2),
reflow/lynxnet2.py -> LYNXNet2 (6.3, default), ddsp/vocoder.py ->
CombSubSuperFast, ddsp/unit2control.py -> Unit2Control,
ddsp/model_conformer_naive.py -> ConformerNaiveEncoder,
nsf_hifigan/nvSTFT.py -> MelExtractor (pure torch + librosa).
Class/attribute names are kept identical so original checkpoints load
directly (``ddsp_model.*``, ``reflow_model.*``, ``reflow_model.velocity_fn.*``).
torchaudio removed: Resample (audio pre-staged at target sr), Spectrogram
(RSSLoss/SSSLoss unused by 6.3's train loop), nvSTFT (reimplemented).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...nn._commons import calc_same_padding


class SwiGLU(nn.Module):
    # fp16 overflow guards differ per source: 'ratio'=lynxnet2, 'float'=conformer 6.3, None=6.1/6.2.
    # split not chunk: ONNX export compatibility.
    def __init__(self, mode='ratio', dim=-1):
        super().__init__(); self.mode, self.dim = mode, dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            if self.mode == 'ratio':
                a, b = torch.aminmax(out.detach()); c, d = torch.aminmax(gate.detach())
                m = torch.max(-a, b).float() * torch.max(-c, d).float()
                if m > 1000:
                    r = (1000 / m).half(); gate *= r
                    return (out * gate).clamp(-1000 * r, 1000 * r) / r
            elif self.mode == 'float':
                if torch.max(-out.detach().amin(), out.detach().amax()) * torch.max(-gate.detach().amin(), gate.detach().amax()) > 1000:
                    return (out.float() * gate.float()).clamp(-1000, 1000).half()
        return out * gate


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, x):
        device = x.device; half = self.dim // 2
        # floating grid in input dtype (half/bf16 inference); long steps train in fp32.
        dtype = x.dtype if x.is_floating_point else torch.float32
        emb = torch.exp(torch.arange(half, device=device, dtype=dtype) * (-math.log(10000) / (half - 1)))
        emb = x[:, None].to(dtype) * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__(); self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0.):
        super().__init__(); inner = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim), Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim), Transpose((1, 2)),
            nn.Linear(dim, inner * 2), SwiGLU(), nn.Linear(inner, inner * 2), SwiGLU(), nn.Linear(inner, dim),
            nn.Dropout(dropout) if float(dropout) > 0 else nn.Identity())

    def forward(self, x):
        return x + self.net(x)


class LYNXNet2(nn.Module):
    def __init__(self, in_dims, dim_cond, n_layers=6, n_chans=512, n_dilates=1, dropout=0.):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans), nn.Linear(n_chans, n_chans * 4), nn.GELU(), nn.Linear(n_chans * 4, n_chans))
        self.residual_layers = nn.ModuleList([LYNXNet2Block(n_chans, n_dilates, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_chans); self.output_projection = nn.Linear(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        use_4d = spec.dim() == 4  # DiffSVC compat [B, 1, M, T]
        x = spec[:, 0] if use_4d else spec
        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"
        x = self.input_projection(x.transpose(1, 2))
        x = x + self.conditioner_projection(cond.transpose(1, 2)) + self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(1)
        for layer in self.residual_layers:
            x = layer(x)
        x = self.output_projection(self.norm(x)).transpose(1, 2)  # [B, 128, T]
        return x[:, None] if use_4d else x


# 6.1/6.2 velocity net (LYNXNet2 replaced it in 6.3): Conv1d input + gelu,
# residual layer post-adds t/cond, post-norm on channel-last, no fp16 clamp.
class LYNXConvModule(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, in_norm=False, activation='PReLU', dropout=0.):
        super().__init__(); inner = dim * expansion_factor
        acts = {'SiLU': nn.SiLU, 'ReLU': nn.ReLU, 'PReLU': lambda: nn.PReLU(inner)}
        act = acts.get(activation if activation is not None else 'PReLU')
        if act is None:
            raise ValueError(f'{activation} is not a valid activation')
        pad = calc_same_padding(kernel_size)
        self.net = nn.ModuleList((
            nn.LayerNorm(dim), Transpose((1, 2)),
            nn.Conv1d(dim, inner * 2, 1), SwiGLU(None, 1),
            nn.Conv1d(inner, inner, kernel_size, padding=pad[0], groups=inner), act(),
            nn.Conv1d(inner, dim, 1), Transpose((1, 2)),
            nn.Dropout(dropout) if float(dropout) > 0 else nn.Identity()))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class LYNXNetResidualLayer(nn.Module):
    def __init__(self, dim_cond, dim, expansion_factor, kernel_size=31, in_norm=False, activation='PReLU', dropout=0.):
        super().__init__()
        self.diffusion_projection = nn.Conv1d(dim, dim, 1)
        self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
        self.convmodule = LYNXConvModule(dim, expansion_factor, kernel_size, in_norm, activation, dropout)

    def forward(self, x, conditioner, diffusion_step):
        res_x = x.transpose(1, 2)
        x = (x + self.diffusion_projection(diffusion_step) + self.conditioner_projection(conditioner)).transpose(1, 2)
        return (self.convmodule(x) + res_x).transpose(1, 2)  # (batch, length, dim)


class LYNXNet(nn.Module):
    def __init__(self, in_dims, dim_cond, n_feats=1, n_layers=6, n_chans=512, n_dilates=2, in_norm=False, activation='PReLU', dropout=0.):
        super().__init__()
        self.input_projection = nn.Conv1d(in_dims * n_feats, n_chans, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans), nn.Linear(n_chans, n_chans * 4), nn.GELU(), nn.Linear(n_chans * 4, n_chans))
        self.residual_layers = nn.ModuleList([
            LYNXNetResidualLayer(dim_cond, n_chans, n_dilates, in_norm=in_norm, activation=activation, dropout=dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Conv1d(n_chans, in_dims * n_feats, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        use_4d = spec.dim() == 4  # DiffSVC compat [B, 1, M, T]
        x = spec[:, 0] if use_4d else spec
        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"
        x = F.gelu(self.input_projection(x))  # [B, residual_channel, T]
        d = self.diffusion_embedding(diffusion_step.to(x.dtype)).unsqueeze(-1)
        for layer in self.residual_layers:
            x = layer(x, cond, d)
        x = self.output_projection(self.norm(x.transpose(1, 2)).transpose(1, 2))  # [B, 128, T]
        return x[:, None] if use_4d else x


class ConformerConvModule(nn.Module):
    # 6.1/6.2 conv1d blocks default expansion_factor=2; 6.3 linear blocks default 1.
    def __init__(self, dim, expansion_factor=None, kernel_size=31, dropout=0., use_norm=False,
                 conv_model_type='mode1', conv_block='linear'):
        super().__init__()
        if conv_block not in ('linear', 'conv1d'):
            raise ValueError(f'{conv_block} is not a valid conv_block')
        if conv_model_type == 'mode2':
            raise NotImplementedError('mode2 not implemented yet')
        if conv_model_type != 'mode1':
            raise ValueError(f'{conv_model_type} is not a valid conv_model_type')
        inner = dim * (2 if expansion_factor is None and conv_block == 'conv1d' else 1 if expansion_factor is None else expansion_factor)
        if conv_block == 'conv1d':  # ConformerConvModule-6.1/6.2: depthwise conv + PReLU, channel SwiGLU.
            self.net = nn.ModuleList((
                nn.LayerNorm(dim) if use_norm else nn.Identity(), Transpose((1, 2)),
                nn.Conv1d(dim, inner * 2, 1), SwiGLU(None, 1),
                nn.Conv1d(inner, inner, kernel_size, padding=kernel_size // 2, groups=inner), nn.PReLU(inner),
                nn.Conv1d(inner, dim, 1), Transpose((1, 2)), nn.Dropout(dropout)))
        else:  # 6.3 linear blocks.
            self.net = nn.ModuleList((
                nn.LayerNorm(dim) if use_norm else nn.Identity(), Transpose((1, 2)),
                nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim), Transpose((1, 2)),
                nn.Linear(dim, inner * 2), SwiGLU('float'), nn.Linear(inner, inner * 2), SwiGLU('float'), nn.Linear(inner, dim),
                nn.Dropout(dropout)))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads=8, use_norm=False, conv_only=False, conv_dropout=0.,
                 atten_dropout=0.1, conv_block='linear'):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model, use_norm=use_norm, dropout=conv_dropout, conv_block=conv_block)
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(dim_model, num_heads, dim_model * 4, atten_dropout, activation='gelu')
            self.norm = nn.LayerNorm(dim_model)
        else:
            self.attn = None

    def forward(self, x, mask=None):
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask)
        return x + self.conformer(x)


class ConformerNaiveEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, use_norm=False, conv_only=False, conv_dropout=0.,
                 atten_dropout=0., conv_block='linear'):
        super().__init__()
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.encoder_layers = nn.ModuleList([
            CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout, conv_block)
            for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


def _split_to_dict(tensor, tensor_splits):
    return dict(zip(tensor_splits, torch.split(tensor, list(tensor_splits.values()), dim=-1)))


class Unit2Control(nn.Module):
    def __init__(self, input_channel, block_size, n_spk, output_splits, num_layers=3, dim_model=256,
                 use_norm=False, use_attention=False, use_pitch_aug=False, conv_block='linear'):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed, self.phase_embed, self.volume_embed = [nn.Linear(1, dim_model) for _ in range(3)]
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, dim_model)
        self.aug_shift_embed = nn.Linear(1, dim_model, bias=False) if use_pitch_aug else None
        self.stack = nn.Sequential(nn.utils.weight_norm(nn.Conv1d(input_channel, 512, 3, 1, 1)), nn.PReLU(512),
                                   nn.utils.weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.stack2 = nn.Sequential(nn.utils.weight_norm(nn.Conv1d(2 * block_size, 512, 3, 1, 1)), nn.PReLU(512),
                                    nn.utils.weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.decoder = ConformerNaiveEncoder(num_layers, 8, dim_model, use_norm, not use_attention, 0, 0.1, conv_block)
        self.norm = nn.LayerNorm(dim_model)
        self.dense_out = nn.utils.weight_norm(nn.Linear(dim_model, sum(output_splits.values())))

    def forward(self, units, source, noise, volume, spk_id=None, spk_mix_dict=None, aug_shift=None):
        exciter = torch.cat((source, noise), dim=-1).transpose(1, 2)
        x = self.stack(units.transpose(1, 2)) + self.stack2(exciter)
        x = x.transpose(1, 2) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    x = x + v * self.spk_embed(torch.tensor([[k]], dtype=torch.long, device=units.device))
            else:
                x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.norm(self.decoder(x))
        e = self.dense_out(x)
        return _split_to_dict(e, self.output_splits), x


class CombSubSuperFast(nn.Module):
    def __init__(self, sampling_rate, block_size, win_length, n_unit=256, n_spk=1, num_layers=3,
                 dim_model=256, use_norm=False, use_attention=False, use_pitch_aug=False, conv_block='linear',
                 glu_type='softsign_glu'):
        super().__init__()
        self.register_buffer('sampling_rate', torch.tensor(sampling_rate))
        self.register_buffer('block_size', torch.tensor(block_size))
        self.register_buffer('win_length', torch.tensor(win_length))
        self.register_buffer('window', torch.hann_window(win_length))
        self._spectral_fp32 = conv_block == '6.5'
        s = win_length // 2 + 1
        splits = dict(harmonic_magnitude=s, harmonic_phase=s, noise_magnitude=s, noise_phase=s)
        if conv_block == '6.5':
            from .v65 import Unit2Control65
            self.unit2ctrl = Unit2Control65(n_unit, block_size, n_spk, splits,
                                           num_layers, dim_model, use_pitch_aug, glu_type)
        else:
            self.unit2ctrl = Unit2Control(n_unit, block_size, n_spk, splits,
                                          num_layers, dim_model, use_norm, use_attention, use_pitch_aug, conv_block)

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
        return torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None,
                aug_shift=None, initial_phase=None, infer=True, **kwargs):
        combtooth = self.fast_source_gen(f0_frames)  # combtooth exciter
        noise = torch.randn_like(combtooth)  # noise exciter
        ctrls, hidden = self.unit2ctrl(units_frames, combtooth.unfold(1, self.block_size, self.block_size),
                                       noise.unfold(1, self.block_size, self.block_size), volume_frames,
                                       spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift)
        if self._spectral_fp32:
            # AMP linear outputs may be BF16; PyTorch cannot multiply BComplex32.
            # Keep the 6.5 spectral synthesis in FP32 without detaching gradients.
            ctrls = {key: value.float() for key, value in ctrls.items()}
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * math.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)
        noise_filter = torch.exp(ctrls['noise_magnitude'] + 1.j * math.pi * ctrls['noise_phase']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)
        mode = 'reflect' if combtooth.shape[-1] > self.win_length // 2 else 'constant'
        stft = lambda s: torch.stft(s, n_fft=self.win_length, win_length=self.win_length, hop_length=self.block_size,
                                    window=self.window, center=True, return_complex=True, pad_mode=mode)
        sig = stft(combtooth) * src_filter.permute(0, 2, 1) + stft(noise) * noise_filter.permute(0, 2, 1)
        return torch.istft(sig, n_fft=self.win_length, win_length=self.win_length, hop_length=self.block_size,
                           window=self.window, center=True), hidden


class RectifiedFlow(nn.Module):
    # NOTE on FlowMatchingObjective (usvc/train/objective.py): same math, NOT reused
    # because bit-parity with DDSP-SVC-6.3 requires: t in [t_start, 1) clipped to
    # [1e-7, 1-1e-7] (no t_start offset there), the SAME t for xt/velocity input
    # (x1000) and l2_lognorm weights (objective re-samples t for weights), and the
    # 4-D [B,1,M,T] layout. Verbatim impl keeps the loss bit-comparable.
    def __init__(self, velocity_fn, out_dims=128, spec_min=-12, spec_max=2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims, self.spec_min, self.spec_max = out_dims, spec_min, spec_max

    def reflow_loss(self, x_1, t, cond, loss_type='l2_lognorm', mask=None):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v = self.velocity_fn(x_t, 1000 * t, cond)
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

    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, 1000 * t, cond) * dt
        t += dt
        return x, t

    def sample_rk4(self, x, t, dt, cond):
        k1 = self.velocity_fn(x, 1000 * t, cond)
        k2 = self.velocity_fn(x + .5 * k1 * dt, 1000 * (t + .5 * dt), cond)
        k3 = self.velocity_fn(x + .5 * k2 * dt, 1000 * (t + .5 * dt), cond)
        k4 = self.velocity_fn(x + k3 * dt, 1000 * (t + dt), cond)
        x += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        t += dt
        return x, t

    def forward(self, condition, gt_spec=None, infer=True, infer_step=10, method='euler',
                t_start=0.0, use_tqdm=True, mask=None):
        cond = condition.transpose(1, 2)  # [B, H, T]
        b, device = condition.shape[0], condition.device
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec).transpose(1, 2)[:, None]  # [B, 1, M, T]
            t = torch.clip(t_start + (1.0 - t_start) * torch.rand(b, device=device), 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, t, cond=cond, mask=mask)
        shape = (b, 1, self.out_dims, cond.shape[2])  # [B, 1, M, T]
        if gt_spec is None:
            x = torch.randn(shape, device=device, dtype=cond.dtype)
            t = torch.full((b,), 0, device=device)
            dt = 1.0 / infer_step
        else:
            x = t_start * self.norm_spec(gt_spec).transpose(1, 2)[:, None] + (1 - t_start) * torch.randn(shape, device=device, dtype=cond.dtype)
            t = torch.full((b,), t_start, device=device)
            dt = (1.0 - t_start) / infer_step
        step = {'euler': self.sample_euler, 'rk4': self.sample_rk4}.get(method)
        if step is None:
            raise NotImplementedError(method)
        it = tqdm(range(infer_step), desc='sample time step', total=infer_step) if use_tqdm else range(infer_step)
        for _ in it:
            x, t = step(x, t, dt, cond)
        x = torch.clamp(x.squeeze(1).transpose(1, 2), -1.0, 1.0)  # clamp before denorm: avoid vocoder overshoot/clip
        return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# nvSTFT.get_mel replica (pure torch + librosa): 128 mels, n_fft=2048, hop=512, fmin=40, fmax=16000.
class MelExtractor:
    def __init__(self, sr=44100, n_mels=128, n_fft=2048, win_size=2048, hop_length=512,
                 fmin=40.0, fmax=16000.0, clip_val=1e-5):
        self.sr, self.n_mels, self.n_fft, self.win_size, self.hop_length = sr, n_mels, n_fft, win_size, hop_length
        self.fmin, self.fmax, self.clip_val, self._cache = fmin, fmax, clip_val, {}

    def _ensure(self, device):
        key = str(device)
        if key not in self._cache:
            import librosa
            basis = torch.from_numpy(librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                                         fmin=self.fmin, fmax=self.fmax)).float().to(device)
            self._cache[key] = (basis, torch.hann_window(self.win_size).to(device))
        return self._cache[key]

    def extract(self, y):
        # y [B, L] -> log-mel [B, T, M]. Deliberately NOT inference_mode: the
        # original runs extract inside training forward with autograd, so
        # ddsp/reflow losses backprop through the mel front-end into the DDSP
        # model; an inference guard here breaks LYNXNet backward and silently
        # detaches DDSP from training. Sampling callers wrap in no_grad.
        mel_basis, hann = self._ensure(y.device)
        pad_left = (self.win_size - self.hop_length) // 2  # nvSTFT center=False grid
        pad_right = max((self.win_size - self.hop_length + 1) // 2, self.win_size - y.size(-1) - pad_left)
        mode = 'reflect' if pad_right < y.size(-1) else 'constant'
        y = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)
        spec = torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_size, window=hann,
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        return torch.log(torch.clamp(torch.matmul(mel_basis, spec), min=self.clip_val)).transpose(1, 2)


class Unit2Wav(nn.Module):
    def __init__(self, sampling_rate, block_size, win_length, n_unit, n_spk, use_norm=False,
                 use_attention=False, use_pitch_aug=False, out_dims=128, n_aux_layers=3, n_aux_chans=256,
                 n_layers=6, n_chans=512, mel_extractor=None, velocity_net="LYNXNet2", glu_type='softsign_glu'):
        super().__init__()
        self.sampling_rate, self.block_size = sampling_rate, block_size
        self.velocity_net = velocity_net
        if velocity_net not in ("LYNXNet", "LYNXNet2", "LYNXNet2_65"):
            raise ValueError(f"unknown velocity_net {velocity_net!r}")
        if velocity_net == "LYNXNet2_65" and use_attention:
            raise ValueError("DDSP-SVC 6.5 does not have an attention backbone")
        conv_block = "6.5" if velocity_net == "LYNXNet2_65" else "conv1d" if velocity_net == "LYNXNet" else "linear"
        self.ddsp_model = CombSubSuperFast(
            sampling_rate, block_size, win_length, n_unit, n_spk,
            n_aux_layers if n_aux_layers is not None else 3,
            n_aux_chans if n_aux_chans is not None else 256,
            use_norm, use_attention, use_pitch_aug, conv_block=conv_block, glu_type=glu_type)
        if velocity_net == "LYNXNet2_65":
            from .v65 import LYNXNet2_65, RectifiedFlow65
            self.reflow_model = RectifiedFlow65(
                LYNXNet2_65(out_dims, out_dims, n_layers, n_chans, glu_type=glu_type), out_dims=out_dims)
        else:
            velocity_cls = LYNXNet if velocity_net == "LYNXNet" else LYNXNet2
            self.reflow_model = RectifiedFlow(
                velocity_cls(in_dims=out_dims, dim_cond=out_dims, n_layers=n_layers, n_chans=n_chans), out_dims=out_dims)
        self.mel_extractor = mel_extractor if mel_extractor is not None else MelExtractor(
            sr=sampling_rate, n_mels=out_dims, n_fft=win_length, win_size=win_length, hop_length=block_size)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0,
                silence_front=0, use_tqdm=True, frame_len=None):
        ddsp_wav, hidden = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                           aug_shift=aug_shift, infer=infer)
        extractor = vocoder if vocoder is not None else self.mel_extractor
        start = int(silence_front * self.sampling_rate / self.block_size) * self.block_size
        ddsp_mel = None if extractor is None else extractor.extract(ddsp_wav[:, start:]).to(ddsp_wav.dtype)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            if t_start < 1.0:
                from ...train.objective import lens_to_mask
                _mask = None if frame_len is None else lens_to_mask(frame_len, length=gt_spec.shape[1]).to(gt_spec.device)  # frame_len may be CPU
                reflow_loss = self.reflow_model(ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False, mask=_mask)
            else:
                reflow_loss = torch.tensor(0)
            return ddsp_loss, reflow_loss
        if gt_spec is not None and ddsp_mel is None:
            ddsp_mel = gt_spec
        mel = self.reflow_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_step=infer_step, method=method,
                                t_start=t_start, use_tqdm=use_tqdm) if t_start < 1.0 else ddsp_mel
        if return_wav:
            # Original calls the frozen NSF-HiFiGAN generator here; usvc does
            # not bundle those weights, so wav-domain synthesis is unavailable.
            raise NotImplementedError("return_wav=True requires the frozen NSF-HiFiGAN generator (pretrain/nsf_hifigan), which usvc does not bundle.")
        return mel


# DDSP + reflow with usvc training interface. velocity_net: LYNXNet (6.1/6.2) | LYNXNet2 (6.3, default).
# Defaults match DDSP-SVC-6.3 configs/reflow.yaml: 44100/512/2048, contentvec768l12tta2x
# (n_units=768), use_norm=True, use_attention=False, use_pitch_aug=True, n_aux_layers=6,
# n_aux_chans=512, n_layers=6, n_chans=1024, out_dims=128.
class DDSP6Model(Unit2Wav):
    def __init__(self, n_units=768, n_spk=1, win_length=2048, sampling_rate=44100, block_size=512,
                 use_norm=True, use_attention=False, use_pitch_aug=True, out_dims=128, n_aux_layers=6,
                 n_aux_chans=512, n_layers=6, n_chans=1024, t_start=0.0, mel_extractor=None, velocity_net="LYNXNet2",
                 glu_type='softsign_glu'):
        super().__init__(sampling_rate, block_size, win_length, n_units, n_spk, use_norm, use_attention,
                         use_pitch_aug, out_dims, n_aux_layers, n_aux_chans, n_layers, n_chans, mel_extractor, velocity_net,
                         glu_type)
        self.t_start = t_start

    def forward(self, units, f0, volume, spk_id, gt_spec, aug_shift=None, frame_len=None, lambda_ddsp=1.0, t_start=None):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume.dim() == 2:
            volume = volume.unsqueeze(-1)
        if t_start is None:
            t_start = self.t_start
        ddsp_loss, reflow_loss = super().forward(units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
                                                 gt_spec=gt_spec, infer=False, t_start=t_start, frame_len=frame_len)
        reflow_loss = reflow_loss.to(ddsp_loss.device)
        total = lambda_ddsp * ddsp_loss + reflow_loss
        return total, {"ddsp_loss": float(ddsp_loss.item()), "reflow_loss": float(reflow_loss.item())}

    def sample(self, units, f0, volume, spk_id, aug_shift=None, infer_step=10, method='euler', t_start=None, use_tqdm=False):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
        if volume.dim() == 2:
            volume = volume.unsqueeze(-1)
        if t_start is None:
            t_start = self.t_start
        return super().forward(units, f0, volume, spk_id=spk_id, aug_shift=aug_shift, gt_spec=None, infer=True,
                               infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
