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


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward_cl(self, x, conditioner, diffusion_step, conv2d_cl):
        """channels_last 版（x/conditioner 为 [B,C,1,T] CL）。"""
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = conv2d_cl(conditioner, self.conditioner_projection)
        y = x + diffusion_step[:, :, :, None]

        y = conv2d_cl(y, self.dilated_conv) + conditioner

        # split dim=1 (channel) — same as NCHW
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = conv2d_cl(y, self.output_projection)

        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def _conv2d_cl(self, x4, conv):
        """Conv1d as Conv2d(1,k) on a channels_last [B,C,1,T] tensor.

        Keeps 1D weight shape (state_dict 兼容); cuDNN picks the NHWC
        tensorop path with zero nchw<->nhwc conversions inside the network.
        """
        w = conv.weight[:, :, None, :]  # [out, in, 1, k]
        return F.conv2d(
            x4, w, conv.bias,
            stride=1,
            padding=(0, conv.padding[0]),
            dilation=(1, conv.dilation[0]),
        )

    def forward_cl(self, spec, diffusion_step, cond):
        """channels_last 版 forward：网络内零布局转换（等价于 forward）。"""
        # spec [B,1,M,T] -> [B,M,1,T]（mel 作通道）-> channels_last
        x = spec.transpose(1, 2).to(memory_format=torch.channels_last)
        x = self._conv2d_cl(x, self.input_projection)   # [B,C,1,T] CL

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        cond4 = cond.unsqueeze(2).to(memory_format=torch.channels_last)  # [B,M,1,T]
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer.forward_cl(
                x, cond4, diffusion_step, self._conv2d_cl)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self._conv2d_cl(x, self.skip_projection)
        x = F.relu(x)
        x = self._conv2d_cl(x, self.output_projection)  # [B,128,1,T]
        return x.transpose(1, 2)  # [B,1,128,T]

    def forward(self, spec, diffusion_step, cond):
        if getattr(self, "_use_cl", False):
            return self.forward_cl(spec, diffusion_step, cond)
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]
