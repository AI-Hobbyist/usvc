import torch
import torch.nn as nn
from .constants import N_MELS


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
        )
        # shortcut key exists only when in!=out; Identity keeps the key set identical
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else nn.Identity()

    def forward(self, x): out = self.conv(x); return out + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)] + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])
        self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

    def forward(self, x):
        for conv in self.conv: x = conv(x)
        return (x, self.pool(x)) if self.pool is not None else x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride, (1, 1), output_padding=(0, 1) if stride == (1, 2) else (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)] + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv in self.conv2: x = conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList([ResEncoderBlock(out_channels * 2 ** (i - 1) if i else in_channels, out_channels * 2 ** i, kernel_size, n_blocks, momentum) for i in range(n_encoders)])
        self.latent_channels = [[out_channels * 2 ** i, in_size // 2 ** i] for i in range(n_encoders)]
        self.out_size, self.out_channel = in_size // 2 ** n_encoders, out_channels * 2 ** n_encoders

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            _, x = layer(x)
            concat_tensors.append(_)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels if i == 0 else out_channels, out_channels, None, n_blocks, momentum) for i in range(n_inters)])

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResDecoderBlock(in_channels // 2 ** i, in_channels // 2 ** (i + 1), stride, n_blocks, momentum) for i in range(n_decoders)])

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class TimbreFilter(nn.Module):
    def __init__(self, latent_rep_channels):
        super().__init__()
        self.layers = nn.ModuleList([ConvBlockRes(rep[0], rep[0]) for rep in latent_rep_channels])

    def forward(self, x_tensors):
        return [layer(t) for layer, t in zip(self.layers, x_tensors)]


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.decoder(self.intermediate(x), self.tf(concat_tensors))
        return x


class DeepUnet0(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)  # kept: ckpt-free attr, never used in forward (RVC original)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        return self.decoder(self.intermediate(x), concat_tensors)