"""MLX 版 RMVPE F0 提取器(E2E0,无 TimbreFilter;镜像 features/rmvpe/:
model.py / deepunet.py / seq.py / spec.py)。配置(E2E0 推理实例):
n_blocks=4, n_gru=1, kernel_size=(2,2), en_de_layers=5, inter_layers=4,
in_channels=1, en_out_channels=16。

关键适配:torch NCHW ↔ MLX NHWC(mel (B,128,T) 经 _to_nhwc → (B,T_pad,128,1),
T pad 到 32 倍数,对齐 torch mel2hidden 的 F.pad);顺序容器用普通 list 属性
(键名 = 属性名+索引,与 torch 对齐;mlx Sequential 是 layers.* 不匹配);
mlx GRU 丢 bias_hh 的 r/z 段 → 手写 cell 复刻 torch 公式(r/z/n 门序,
weight_ih_l0/_reverse 命名与 torch 一致);BatchNorm momentum=0.01;
ConvTranspose2d (I,O,kH,kW)→(O,kH,kW,I) 在 from_ckpt 单独转置(启发式对它错)。
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ._ops import hann_window, stft_center

__all__ = ["RMVPE", "E2E0", "DeepUnet0", "BiGRU", "MelSpectrogram"]

# ---- 常量(镜像 features/rmvpe/constants.py) ------------------------------
SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = 8000
WINDOW_LENGTH = 1024


def _gru_init(rows: int, cols: int):
    bound = 1.0 / math.sqrt(cols)
    return mx.random.uniform(shape=(rows, cols), low=-bound, high=bound)


class _GRU(nn.Module):
    """单层双向 GRU cell。参数命名与 torch nn.GRU(bidirectional=True) 完全
    对齐(weight_ih_l0/_reverse + bias_*),ckpt 直接 load_weights 无 remap。
    门序(torch):3*H 行按 [r; z; n]。"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.input_size = input_size
        self.hidden_size = H
        for name in ("weight_ih_l0", "weight_hh_l0",
                     "weight_ih_l0_reverse", "weight_hh_l0_reverse"):
            cols = input_size if name.startswith("weight_ih") else H
            setattr(self, name, _gru_init(3 * H, cols))
        for name in ("bias_ih_l0", "bias_hh_l0",
                     "bias_ih_l0_reverse", "bias_hh_l0_reverse"):
            setattr(self, name, mx.zeros((3 * H,)))

    def _cell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        """单步 GRU(torch 公式逐位):
        r=sigmoid(x·W_irᵀ+b_ir+h·W_hrᵀ+b_hr);z=sigmoid(同理);
        n=tanh(x·W_inᵀ+b_in+r*(h·W_hnᵀ+b_hn));h'=(1-z)*n+z*h"""
        H = self.hidden_size
        xp, hp = x @ w_ih.T + b_ih, h @ w_hh.T + b_hh  # (B, 3H)
        xr, xz, xn = mx.split(xp, 3, axis=-1)
        hr, hz, hn = mx.split(hp, 3, axis=-1)
        n = mx.tanh(xn + mx.sigmoid(xr + hr) * hn)
        return (1 - mx.sigmoid(xz + hz)) * n + mx.sigmoid(xz + hz) * h

    def _scan(self, x, w_ih, w_hh, b_ih, b_hh):
        B, T, _ = x.shape
        h = mx.zeros((B, self.hidden_size))
        outs = []
        for t in range(T):
            h = self._cell(x[:, t], h, w_ih, w_hh, b_ih, b_hh)
            outs.append(h)
        return mx.stack(outs, axis=1)  # (B, T, H)

    def bidirectional(self, x):
        fwd = self._scan(x, self.weight_ih_l0, self.weight_hh_l0,
                         self.bias_ih_l0, self.bias_hh_l0)
        bwd = mx.flip(self._scan(mx.flip(x, axis=1), self.weight_ih_l0_reverse,
                                 self.weight_hh_l0_reverse,
                                 self.bias_ih_l0_reverse, self.bias_hh_l0_reverse),
                      axis=1)
        return mx.concatenate([fwd, bwd], axis=-1)


class BiGRU(nn.Module):

    def __init__(self, input_features: int, hidden_features: int, num_layers: int = 1):
        super().__init__()
        self.gru = _GRU(input_features, hidden_features)

    def __call__(self, x):
        return self.gru.bidirectional(x)


# --- U-Net(镜像 deepunet.py) ----------------------------------------------

class ConvBlockRes(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = [
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.relu,
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.relu,
        ]
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def __call__(self, x):
        h = x
        for layer in self.conv:
            h = layer(h)
        return h + self.shortcut(x) if self.is_shortcut else h + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = [ConvBlockRes(in_channels, out_channels, momentum)] + [
            ConvBlockRes(out_channels, out_channels, momentum)
            for _ in range(n_blocks - 1)
        ]
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size, stride=kernel_size)

    def __call__(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Module):
    """ConvTranspose2d+BN+ReLU → cat skip → n_blocks ConvBlockRes(NHWC cat)。"""

    def __init__(self, in_channels, out_channels, stride,
                 n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = [
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=stride,
                               padding=(1, 1), output_padding=out_padding,
                               bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.relu,
        ]
        self.conv2 = [ConvBlockRes(out_channels * 2, out_channels, momentum)] + [
            ConvBlockRes(out_channels, out_channels, momentum)
            for _ in range(n_blocks - 1)
        ]

    def __call__(self, x, concat_tensor):
        for layer in self.conv1:
            x = layer(x)
        x = mx.concatenate([x, concat_tensor], axis=-1)  # NHWC channel 在末维
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, in_size, n_encoders, kernel_size,
                 n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm(in_channels, momentum=momentum)
        self.layers = []
        self.latent_channels = []
        for _ in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels,
                                               kernel_size, n_blocks,
                                               momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def __call__(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            _, x = self.layers[i](x)
            concat_tensors.append(_)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = [ResEncoderBlock(in_channels, out_channels, None,
                                       n_blocks, momentum)] + [
            ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            for _ in range(n_inters - 1)
        ]

    def __call__(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = []
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride,
                                               n_blocks, momentum))
            in_channels = out_channels

    def __call__(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet0(nn.Module):

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                 in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size,
                               n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2,
                                         self.encoder.out_channel,
                                         inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers,
                               kernel_size, n_blocks)

    def __call__(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        return self.decoder(x, concat_tensors)


class MelSpectrogram(nn.Module):
    """镜像 spec.py:torch.stft(center=True)→ _ops.stft_center(镜像补 n_fft//2,
    torch center=True 默认 reflect pad);librosa **htk=True** mel basis
    (⚠ usvc mel.py 是 htk=False);hann 与 torch.hann_window 逐位一致。"""

    def __init__(self, n_mel_channels: int = N_MELS,
                 sampling_rate: int = SAMPLE_RATE,
                 win_length: int = WINDOW_LENGTH, hop_length: int = 160,
                 n_fft: int | None = None, mel_fmin: float = MEL_FMIN,
                 mel_fmax: float = MEL_FMAX, clamp: float = 1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        import librosa
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft,
                                        n_mels=n_mel_channels, fmin=mel_fmin,
                                        fmax=mel_fmax, htk=True)
        self.mel_basis = mx.array(mel_basis.astype(np.float32))  # [n_mels, F]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.hann = hann_window(win_length)

    def __call__(self, audio, keyshift=0, speed=1, center=True):
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None, :]
        B, L = audio.shape
        if keyshift != 0:
            factor = 2 ** (keyshift / 12)
            n_fft_new = int(round(self.n_fft * factor))
            win_new = int(round(self.win_length * factor))
            hop_new = int(round(self.hop_length * speed))
            hann = hann_window(win_new)
        else:
            n_fft_new, win_new, hop_new = self.n_fft, self.win_length, self.hop_length
            hann = self.hann
        mels = []
        for b in range(B):
            spec = stft_center(audio[b], n_fft_new, hop_new, win_new, hann)
            if keyshift != 0:
                size = self.n_fft // 2 + 1
                if spec.shape[0] < size:
                    spec = mx.pad(spec, [(0, size - spec.shape[0]), (0, 0)])
                spec = spec[:size, :] * (self.win_length / win_new)  # amplitude comp
            mels.append(mx.log(mx.maximum(self.mel_basis @ spec, self.clamp)))
        mel = mx.stack(mels, axis=0)  # [B, n_mels, T]
        return mel[0] if squeeze else mel


class E2E0(nn.Module):
    """torch model.py::E2E0:mel → unet → cnn → flatten → BiGRU → Linear →
    Sigmoid → (B, T_pad, 360)。"""

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers,
                              in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = [BiGRU(3 * N_MELS, 256, n_gru),
                       nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()]
        else:
            self.fc = [nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()]

    def _to_nhwc(self, mel):
        """mel (B,128,T) → (B, T_pad, 128, 1);T pad 到 32 倍数保证 U-Net
        skip concat 的 H/W 对齐(torch mel2hidden 同样 F.pad)。"""
        mel = mel.transpose(0, 2, 1)  # (B, T, 128)
        B, T, C = mel.shape
        t_pad = 32 * ((T - 1) // 32 + 1)
        if t_pad > T:
            mel = mx.pad(mel, [(0, 0), (0, t_pad - T), (0, 0)])
        return mel[:, :, :, None]

    def __call__(self, mel):
        x = self.unet(self._to_nhwc(mel))  # (B, T_pad, 128, 16)
        x = self.cnn(x)  # (B, T_pad, 128, 3)
        B, Tp, W, C = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(B, Tp, 3 * W)  # torch flatten(-2)→(B,T,3*W)
        for layer in self.fc:
            x = layer(x)
        return x  # (B, T_pad, 360)


class RMVPE(nn.Module):

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16,
                 hop_length: int = 160):
        super().__init__()
        self.model = E2E0(n_blocks, n_gru, kernel_size, en_de_layers,
                          inter_layers, in_channels, en_out_channels)
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH,
                                            hop_length, None, MEL_FMIN, MEL_FMAX)
        self.hop_length = hop_length

    def _to_nhwc(self, mel):
        return self.model._to_nhwc(mel)

    @property
    def unet(self):
        return self.model.unet

    @property
    def cnn(self):
        return self.model.cnn

    @property
    def fc(self):
        return self.model.fc

    def mel2hidden(self, mel):
        return self.model(mel)[:, :mel.shape[-1]]

    def __call__(self, mel):
        return self.mel2hidden(mel)

    def infer_from_audio(self, audio, sample_rate: int = SAMPLE_RATE,
                         thred: float = 0.03):
        import librosa
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None, :]
        audio = audio.astype(mx.float32)
        if sample_rate != SAMPLE_RATE:
            wav_np = librosa.resample(np.array(audio[0]), orig_sr=sample_rate,
                                      target_sr=SAMPLE_RATE, res_type="soxr_lq")
            audio = mx.array(wav_np.astype(np.float32))[None, :]
        return self.decode(self.mel2hidden(self.mel_extractor(audio, center=True)),
                           thred=thred)

    def decode(self, hidden, thred: float = 0.03):
        """salience (B,T,360) → f0(纯 numpy;镜像 to_local_average_f0:局部
        加权平均于 **cents** 域再 f0=10*2^(cents/1200);⚠ 直接返回 cents 会
        差一个数量级(实测 mlx 6007Hz vs torch 321Hz 实为同一 f0 的 cents))。"""
        hidden_np = np.array(hidden)
        cents = 20 * np.arange(N_CLASS) + 1997.3794084376191  # bin → cents
        out = []
        for b in range(hidden_np.shape[0]):
            sal = hidden_np[b]  # (T, 360)
            f0_b = np.zeros(sal.shape[0], dtype=np.float32)
            for t in range(sal.shape[0]):
                c = int(sal.argmax(axis=-1)[t])
                start = max(0, c - 4)
                end = min(N_CLASS, c + 5)
                w = sal[t, start:end]
                if np.max(w) > thred:
                    f0_b[t] = 10.0 * 2.0 ** ((float(np.sum(w * cents[start:end]))
                                              / float(np.sum(w))) / 1200.0)
            out.append(f0_b)
        return out[0] if len(out) == 1 else np.stack(out)

    @staticmethod
    def from_ckpt(ckpt_path: str | Path, dtype=None) -> "RMVPE":
        """torch RMVPE ckpt(外层 {"model": state_dict})→ MLX RMVPE:
        Conv2d (O,C,kH,kW)→(O,kH,kW,C) 启发式;ConvTranspose2d
        (I,O,kH,kW)→(O,kH,kW,I) 专用转置(启发式对它错,键 = *.conv1.0.weight);
        GRU 2D 权重原样(键名已对齐);统一加 model. 前缀。"""
        from ._convert import load_torch_sd, to_mlx_sd
        from ._ops import merge_weight_norm

        m = RMVPE()
        merged = merge_weight_norm(load_torch_sd(ckpt_path))
        merged = {("model." + k): v for k, v in merged.items()}
        convt_keys = {k for k in merged if k.endswith(".conv1.0.weight")}
        conv2d_keys = ({k for k in merged
                        if k.endswith(".weight") and np.asarray(merged[k]).ndim == 4}
                       - convt_keys)
        mlx_sd = to_mlx_sd(merged, conv_transpose_keys=conv2d_keys, dtype=dtype)
        for k in convt_keys:
            arr = mx.array(np.transpose(np.asarray(merged[k]), (1, 2, 3, 0)))
            if dtype is not None:
                arr = arr.astype(dtype)
            mlx_sd[k] = arr
        m.load_weights(list(mlx_sd.items()), strict=False)
        m.eval()
        return m