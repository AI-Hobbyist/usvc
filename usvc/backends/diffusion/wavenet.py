"""WaveNet denoiser (Diffusion-SVC-v1 / so-vits shallow diffusion).
Copied verbatim from Diffusion-SVC-v1 diffusion/wavenet.py (MD5-identical to
so-vits-svc diffusion/wavenet.py), so class/param names match original
checkpoints exactly.

    WaveNet(in_dims=128, n_layers=20, n_chans=384, n_hidden=256)
    forward(spec [B,1,M,T], diffusion_step [B,1], cond [B,M,T]) -> [B,1,M,T]
"""
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Mish

# channels_last 推理辅助统一在 usvc/nn/cl_helper.py（rvc/sovits/vocoder/wavenet 共享）
from ...nn.cl_helper import conv2d_cl


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        hd = self.dim // 2
        emb = math.log(10000) / (hd - 1)
        emb = torch.exp(torch.arange(hd, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels,
                                      kernel_size=3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def _fwd(self, x, conditioner, diffusion_step, conv):
        ds = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        if x.dim() == 4:
            ds = ds[:, :, :, None]  # CL 布局 [B,C,1,1]
        y = conv(x + ds, self.dilated_conv) + conv(conditioner, self.conditioner_projection)
        gate, filter = torch.split(y, [self.residual_channels] * 2, dim=1)  # torch.split 而非 chunk，避免 onnx::Slice
        y = conv(torch.sigmoid(gate) * torch.tanh(filter), self.output_projection)
        residual, skip = torch.split(y, [self.residual_channels] * 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

    def forward(self, x, conditioner, diffusion_step):
        return self._fwd(x, conditioner, diffusion_step, lambda t, c: c(t))

    def forward_cl(self, x, conditioner, diffusion_step, conv2d_fn):
        return self._fwd(x, conditioner, diffusion_step, conv2d_fn)


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4), Mish(), nn.Linear(n_chans * 4, n_chans))
        self.residual_layers = nn.ModuleList(
            ResidualBlock(encoder_hidden=n_hidden, residual_channels=n_chans, dilation=1)
            for _ in range(n_layers))
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        if getattr(self, "_use_cl", False):
            return self.forward_cl(spec, diffusion_step, cond)
        x = F.relu(self.input_projection(spec.squeeze(1)))  # [B,1,M,T] -> [B,M,T]
        ds = self.mlp(self.diffusion_embedding(diffusion_step))
        skip = []
        for l in self.residual_layers:
            x, s = l(x, cond, ds)
            skip.append(s)
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)[:, None, :, :]  # [B,M,T] -> [B,1,M,T]

    def forward_cl(self, spec, diffusion_step, cond):
        # 网络内零布局转换：spec [B,1,M,T] -> [B,M,1,T] CL，mel 作通道
        x = spec.transpose(1, 2).to(memory_format=torch.channels_last)
        x = F.relu(conv2d_cl(x, self.input_projection))
        ds = self.mlp(self.diffusion_embedding(diffusion_step))
        cond4 = cond.unsqueeze(2).to(memory_format=torch.channels_last)  # [B,M,1,T]
        skip = []
        for l in self.residual_layers:
            x, s = l.forward_cl(x, cond4, ds, conv2d_cl)
            skip.append(s)
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = F.relu(conv2d_cl(x, self.skip_projection))
        return conv2d_cl(x, self.output_projection).transpose(1, 2)  # [B,128,1,T]->[B,1,128,T]