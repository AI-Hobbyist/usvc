"""DDSP-SVC 6.5 core (local reference revision f794231).

Adapted from reflow/lynxnet2.py, ddsp/unit2control.py and reflow/reflow.py.
Speaker IDs use usvc zero-based indexing; padding masks are handled separately
from the upstream dual-timestep selection mask.

MIT License

Copyright (c) 2023 yxlllc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from torch.nn.utils import weight_norm
import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATanGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate):
        atan_gate = torch.atan(gate)
        decay_out = out / gate.square().add(1.0)
        ctx.save_for_backward(decay_out, atan_gate)
        return out * atan_gate

    @staticmethod
    def backward(ctx, grad_output):
        decay_out, atan_gate = ctx.saved_tensors
        grad_out_part = grad_output * atan_gate
        grad_gate_part = grad_output * decay_out
        return grad_out_part, grad_gate_part


class ATanGLU(nn.Module):
    # ArcTan-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return ATanGLUFunction.apply(out, gate)


class SoftSignGLUFunction(torch.autograd.Function):
    """ATanGLUFunction-style memory trick for SoftSignGLU.

    softsign'(x) = 1/(1+|x|)^2 = (1-|softsign(x)|)^2, so both partial
    derivatives of y = out * softsign(gate) are precomputable in forward:
      dy/dout = softsign(gate)
      dy/dgate = out * (1-|softsign(gate)|)^2
    Saves 2 tensors (vs 3 for naive autograd) and backward is two pure
    multiplies with no softsign recompute.
    """
    @staticmethod
    def forward(ctx, out, gate):
        ss_gate = torch.nn.functional.softsign(gate)
        decay_out = out * (1.0 - ss_gate.abs()).square()
        ctx.save_for_backward(ss_gate, decay_out)
        return out * ss_gate

    @staticmethod
    def backward(ctx, grad_output):
        ss_gate, decay_out = ctx.saved_tensors
        return grad_output * ss_gate, grad_output * decay_out


class SoftSignGLU(nn.Module):
    """Gated Linear Unit with SoftSign gate: out * softsign(gate).

    More numerically stable than ATanGLU (no approximation needed in
    Triton kernels) while providing similar gating behavior.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        if self.training:
            return SoftSignGLUFunction.apply(out, gate)
        else:
            return out * torch.nn.functional.softsign(gate)


class DoubleSoftSignGLUFunction(torch.autograd.Function):
    """Memory-optimized backward for DoubleSoftSignGLU (same trick).

    Operates on the WHOLE Linear output x = [out | gate] (softsign is
    elementwise, so softsign-then-split == split-then-softsign). With
    a = softsign(out), b = softsign(gate), y = a * b:
      dy/dout = b * (1-|a|)^2
      dy/dgate = a * (1-|b|)^2
    Both partials are precomputed into ONE [.., 2N] tensor in forward, so
    backward is a single multiply against the (broadcast) upstream grad —
    no softsign recompute, no cat.
    """
    @staticmethod
    def forward(ctx, x, dim):
        ss = torch.nn.functional.softsign(x)
        a, b = torch.split(ss, ss.size(dim) // 2, dim=dim)
        decay_sq = (1.0 - ss.abs()).square()
        da, db = torch.split(decay_sq, decay_sq.size(dim) // 2, dim=dim)
        # decay = [b*(1-|a|)^2 | a*(1-|b|)^2], written into decay_sq's halves
        da.mul_(b)
        db.mul_(a)
        ctx.save_for_backward(decay_sq)
        ctx.dim = dim
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        decay, = ctx.saved_tensors
        grad_x = decay * torch.cat([grad_output, grad_output], dim=ctx.dim)
        return grad_x, None


class DoubleSoftSignGLU(nn.Module):
    """FastWaveD-style double-gated unit: softsign applied to the whole
    Linear output, then split and multiplied:
      y = softsign(out) * softsign(gate)
    Output is bounded in (-1, 1) since both branches saturate.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.training:
            return DoubleSoftSignGLUFunction.apply(x, self.dim)
        # softsign whole tensor first (one elementwise op), then split
        ss = torch.nn.functional.softsign(x)
        out, gate = torch.split(ss, ss.size(self.dim) // 2, dim=self.dim)
        return out * gate


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        emb = x.unsqueeze(-1) * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


def _make_glu(glu_type):
    if glu_type == 'atanglu':
        return ATanGLU()
    elif glu_type == 'softsign_glu':
        return SoftSignGLU()
    elif glu_type == 'double_softsign_glu':
        return DoubleSoftSignGLU()
    else:
        raise ValueError(f'{glu_type} is not a valid activation')


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0., glu_type='softsign_glu'):
        super().__init__()
        self.dim = dim
        self.glu_type = glu_type
        inner_dim = int(dim * expansion_factor)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            _make_glu(glu_type),
            nn.Linear(inner_dim, inner_dim * 2),
            _make_glu(glu_type),
            nn.Linear(inner_dim, dim),
            _dropout
        )

    def forward(self, x):
        y = F.rms_norm(x, (x.size(-1), ))
        return x + self.net(y)


class LYNXNet2_65(nn.Module):
    def __init__(self, in_dims, dim_cond, n_layers=6, n_chans=512, n_dilates=1, dropout=0., glu_type='softsign_glu'):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.glu_type = glu_type
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
                    dropout=dropout,
                    glu_type=glu_type
                )
                for i in range(n_layers)
            ]
        )
        self.output_projection = nn.Linear(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, diffusion_step_2=None, mask=None):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :param mask: [B, T]
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
        x = x + self.conditioner_projection(cond.transpose(1, 2)) # [B, T, C]

        if mask is not None:
            step = torch.cat((diffusion_step, diffusion_step_2), dim=0)
            step = self.diffusion_embedding(step)
            step, step_2 = torch.split(step, x.shape[0], dim=0) #[B, 1, C]
            mask = mask.to(x).unsqueeze(-1) # [B, T, 1]
            x = x + step + (step_2 - step) * mask
        else:
            x = x + self.diffusion_embedding(diffusion_step)

        for layer in self.residual_layers:
            x = layer(x)

        # post-norm
        x = F.rms_norm(x, (x.size(-1), ))

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        return x[:, None] if use_4_dim else x
def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Unit2Control65(nn.Module):
    def __init__(
            self,
            input_channel,
            block_size,
            n_spk,
            output_splits,
            num_layers=3,
            dim_model=256,
            use_pitch_aug=False,
            glu_type='softsign_glu'):
        super().__init__()
        self.output_splits = output_splits
        self.volume_embed = nn.Linear(1, dim_model)
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, dim_model)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, dim_model, bias=False)
        else:
            self.aug_shift_embed = None

        self.stack = nn.Sequential(
                weight_norm(nn.Conv1d(input_channel, 512, 3, 1, 1)),
                nn.PReLU(num_parameters=512),
                weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.stack2 = nn.Sequential(
                weight_norm(nn.Conv1d(2 * block_size, 512, 3, 1, 1)),
                nn.PReLU(num_parameters=512),
                weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=dim_model,
                    expansion_factor=1,
                    kernel_size=31,
                    glu_type=glu_type
                )
                for i in range(num_layers)
            ]
        )
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = nn.Linear(dim_model, self.n_out)

    def forward(self, units, source, noise, volume, spk_id = None, spk_mix_dict = None, aug_shift = None):

        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''
        exciter = torch.cat((source, noise), dim=-1).transpose(1,2)
        x = self.stack(units.transpose(1, 2)) + self.stack2(exciter)
        x = x.transpose(1, 2) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                x = x + self.spk_embed(spk_id.reshape(-1, 1))
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        for layer in self.residual_layers:
            x = layer(x)
        x = F.rms_norm(x, (x.size(-1), ))
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        return controls, x
class RectifiedFlow65(nn.Module):
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

    def reflow_loss(self, x_1, t1, cond, t2=None, mask=None, loss_type='l2_lognorm', valid_mask=None):
        t = t1 if mask is None else t1 + (t2 - t1) * mask
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, :] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t1, cond, None if t2 is None else 1000 * t2, mask)

        if loss_type == 'l1':
            element_loss = (x_1 - x_0 - v_pred).abs()
        elif loss_type in ('l2', 'l2_lognorm'):
            element_loss = F.mse_loss(x_1 - x_0, v_pred, reduction='none')
        else:
            raise NotImplementedError(loss_type)
        if loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / ( 1 - t)) ** 2)
            element_loss = weights[:, None, None, :] * element_loss
        if valid_mask is None:
            return element_loss.mean()
        valid = valid_mask.to(element_loss)[:, None, None, :]
        return (element_loss * valid).sum() / (valid.sum() * self.out_dims + 1e-8)

    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, 1000 * t, cond) * dt
        t += dt
        return x, t

    def sample_rk4(self, x, t, dt ,cond):
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
                use_tqdm=True, mask=None):
        cond = condition.transpose(1, 2) # [B, H, T]
        b, _, n_frames = cond.shape
        device = condition.device
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec)
            x_1 = x_1.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            t = t_start + (1.0 - t_start) * torch.rand(b, 1, device=device)
            t = torch.clip(t, 1e-7, 1-1e-7)
            t2 = t_start + (1.0 - t_start) * torch.rand(b, 1, device=device)
            t2 = torch.clip(t2, 1e-7, 1-1e-7)
            time_mask = (torch.rand(b, n_frames, device=device) < 0.25).float()
            return self.reflow_loss(x_1, t, cond=cond, t2=t2, mask=time_mask, valid_mask=mask)
        else:
            shape = (b, 1, self.out_dims, n_frames) # [B, 1, M, T]

            # initial condition and step size of the ODE
            if gt_spec is None:
                x = torch.randn(shape, device=device)
                t = torch.full((b,), 0.0, device=device, dtype=torch.float32)
                dt = 1.0 / infer_step
            else:
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :] # [B, 1, M, T]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device)
                t = torch.full((b,), t_start, device=device, dtype=torch.float32)
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
