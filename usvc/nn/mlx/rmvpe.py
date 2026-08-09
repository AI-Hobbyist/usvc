"""MLX 版 RMVPE F0 提取器(E2E0,无 TimbreFilter)。

镜像 ``usvc/features/rmvpe/`` 的 torch 实现:
  - ``model.py``: E2E0 = unet(DeepUnet0) + cnn + fc(BiGRU + Linear + Sigmoid)
  - ``deepunet.py``: DeepUnet0 = Encoder + Intermediate + Decoder
  - ``seq.py``: BiGRU(手写双向)
  - ``spec.py``: MelSpectrogram(center=True STFT + librosa htk mel)

配置(E2E0 推理实例): n_blocks=4, n_gru=1, kernel_size=(2,2),
en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16。

关键 MLX 适配(映射手册 + 实测):
  - 输入布局: torch NCHW (B,1,T,128) ↔ MLX NHWC (B,T,128,1)。
    mel (B,128,T) 经 ``_to_nhwc`` → (B, T_pad, 128, 1),T pad 到 32 的倍数
    (与 torch ``mel2hidden`` 的 ``F.pad`` 一致,保证 U-Net 各层 concat 对齐)。
  - mlx ``nn.Sequential`` 键名是 ``layers.*``,与 torch ``0/1`` 不匹配 →
    所有顺序容器用**普通 list 属性**(键名 = 属性名 + 索引,与 torch ckpt 对齐)。
  - mlx GRU 丢失 bias_hh 的 r/z 段 → 手写 GRU cell 复刻 torch 公式
    (r/z/n 门序,weight_ih_l0/weight_hh_l0 + _reverse 参数命名与 torch 一致)。
  - BatchNorm: mlx ``nn.BatchNorm``(NHWC),momentum=0.01(RMVPE 特定)。
  - ConvTranspose2d 权重: torch (I,O,kH,kW) → mlx (O,kH,kW,I),
    在 ``from_ckpt`` 里单独转置(``_convert`` 启发式 (0,2,3,1) 对它错误)。
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


# ---------------------------------------------------------------------------
# 手写双向 GRU(torch 公式逐位复刻)
# ---------------------------------------------------------------------------

def _gru_init(rows: int, cols: int):
    """与 torch GRU 近似的初始化(uniform,结构测试够用)。"""
    bound = 1.0 / math.sqrt(cols)
    return mx.random.uniform(shape=(rows, cols), low=-bound, high=bound)


class _GRU(nn.Module):
    """单层 GRU cell(前向+反向两套权重)。

    参数命名与 torch ``nn.GRU(..., bidirectional=True)`` 的 state_dict
    完全对齐: weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0 及
    ``_reverse`` 后缀,因此 ckpt 直接 load_weights 无需 key remap。

    门序(torch): weight_ih/weight_hh 的 3*H 行按 [r; z; n] 排列。
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.input_size = input_size
        self.hidden_size = H
        for name in (
            "weight_ih_l0", "weight_hh_l0",
            "weight_ih_l0_reverse", "weight_hh_l0_reverse",
        ):
            rows = 3 * H
            cols = input_size if name.startswith("weight_ih") else H
            setattr(self, name, _gru_init(rows, cols))
        for name in (
            "bias_ih_l0", "bias_hh_l0",
            "bias_ih_l0_reverse", "bias_hh_l0_reverse",
        ):
            setattr(self, name, mx.zeros((3 * H,)))

    def _cell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        """单步 GRU(与 torch 公式逐位一致)。

        torch:
          r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
          z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
          n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
          h' = (1 - z) * n + z * h
        """
        H = self.hidden_size
        xp = x @ w_ih.T + b_ih          # (B, 3H)
        hp = h @ w_hh.T + b_hh          # (B, 3H)
        xr, xz, xn = mx.split(xp, 3, axis=-1)
        hr, hz, hn = mx.split(hp, 3, axis=-1)
        r = mx.sigmoid(xr + hr)
        z = mx.sigmoid(xz + hz)
        n = mx.tanh(xn + r * hn)
        return (1 - z) * n + z * h

    def _scan(self, x, w_ih, w_hh, b_ih, b_hh):
        """沿时间轴扫描(cumsum 式顺序依赖,惰性建图)。"""
        B, T, _ = x.shape
        h = mx.zeros((B, self.hidden_size))
        outs = []
        for t in range(T):
            h = self._cell(x[:, t], h, w_ih, w_hh, b_ih, b_hh)
            outs.append(h)
        return mx.stack(outs, axis=1)   # (B, T, H)

    def bidirectional(self, x):
        """前向 + 反向(时间反转)拼接 → (B, T, 2H)。"""
        fwd = self._scan(x, self.weight_ih_l0, self.weight_hh_l0,
                         self.bias_ih_l0, self.bias_hh_l0)
        x_rev = mx.flip(x, axis=1)
        bwd = self._scan(x_rev, self.weight_ih_l0_reverse,
                         self.weight_hh_l0_reverse,
                         self.bias_ih_l0_reverse, self.bias_hh_l0_reverse)
        bwd = mx.flip(bwd, axis=1)
        return mx.concatenate([fwd, bwd], axis=-1)


class BiGRU(nn.Module):
    """对应 torch ``features/rmvpe/seq.py::BiGRU``(bidirectional=True)。

    键名 fc.0.gru.*(torch 里 BiGRU 包装了 self.gru = nn.GRU)。"""
    def __init__(self, input_features: int, hidden_features: int, num_layers: int = 1):
        super().__init__()
        self.gru = _GRU(input_features, hidden_features)

    def __call__(self, x):
        return self.gru.bidirectional(x)


# ---------------------------------------------------------------------------
# U-Net(镜像 features/rmvpe/deepunet.py)
# ---------------------------------------------------------------------------

class ConvBlockRes(nn.Module):
    """Conv2d + BN + ReLU + Conv2d + BN + ReLU(可选 shortcut 1x1)。

    torch 用 nn.Sequential,键名 conv.0/conv.1/...;MLX 用 list 属性保持对齐。
    """

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
        if self.is_shortcut:
            return h + self.shortcut(x)
        return h + x


class ResEncoderBlock(nn.Module):
    """n_blocks 个 ConvBlockRes + 可选 AvgPool2d。"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = [ConvBlockRes(in_channels, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
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
    """ConvTranspose2d + BN + ReLU,再 cat skip,再 n_blocks 个 ConvBlockRes。"""

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
        self.conv2 = [ConvBlockRes(out_channels * 2, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def __call__(self, x, concat_tensor):
        for layer in self.conv1:
            x = layer(x)
        # NHWC: channel 在最后一维
        x = mx.concatenate([x, concat_tensor], axis=-1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):
    """入口 BN + n_encoders 个 ResEncoderBlock(每级通道翻倍、HW 减半)。"""

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
    """n_inters 个无池化 ResEncoderBlock。"""

    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = [ResEncoderBlock(in_channels, out_channels, None,
                                       n_blocks, momentum)]
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None,
                                               n_blocks, momentum))

    def __call__(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    """n_decoders 个 ResDecoderBlock,通道逐级减半。"""

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
    """Encoder → Intermediate → Decoder(E2E0 用,无 TimbreFilter)。"""

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
        x = self.decoder(x, concat_tensors)
        return x


# ---------------------------------------------------------------------------
# MelSpectrogram(center=True STFT + librosa htk mel)
# ---------------------------------------------------------------------------

class MelSpectrogram(nn.Module):
    """镜像 ``features/rmvpe/spec.py::MelSpectrogram``。

    差异:
      - torch.stft(center=True) → ``_ops.stft_center``(两端零补 n_fft//2)
      - librosa htk mel basis(htk=True!与 usvc mel.py 的 htk=False 不同)
      - hann 窗与 torch.hann_window 逐位一致(``_ops.hann_window``)
    """

    def __init__(
        self,
        n_mel_channels: int = N_MELS,
        sampling_rate: int = SAMPLE_RATE,
        win_length: int = WINDOW_LENGTH,
        hop_length: int = 160,
        n_fft: int | None = None,
        mel_fmin: float = MEL_FMIN,
        mel_fmax: float = MEL_FMAX,
        clamp: float = 1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        import librosa
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax, htk=True,
        )
        self.mel_basis = mx.array(mel_basis.astype(np.float32))  # [n_mels, n_fft//2+1]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.hann = hann_window(win_length)

    def __call__(self, audio, keyshift=0, speed=1, center=True):
        """audio: 1d [L] 或 2d [B, L] float → mel [B, n_mels, T](频域在前)。"""
        import mlx.core as mx
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None, :]  # [1, L]
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
            # spec: [freq, frames]
            if keyshift != 0:
                size = self.n_fft // 2 + 1
                resize = spec.shape[0]
                if resize < size:
                    spec = mx.pad(spec, [(0, size - resize), (0, 0)])
                spec = spec[:size, :] * (self.win_length / win_new)
            mel_output = self.mel_basis @ spec  # [n_mels, frames]
            mels.append(mx.log(mx.maximum(mel_output, self.clamp)))
        mel = mx.stack(mels, axis=0)  # [B, n_mels, T]
        return mel[0] if squeeze else mel


# ---------------------------------------------------------------------------
# E2E0 / RMVPE
# ---------------------------------------------------------------------------

class E2E0(nn.Module):
    """对应 torch ``features/rmvpe/model.py::E2E0``。

    forward: mel (B, 128, T) → unet → cnn → flatten → BiGRU → Linear → Sigmoid
    → (B, T, 360)。
    """

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers,
                              in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = [
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            ]
        else:
            self.fc = [
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            ]

    def _to_nhwc(self, mel):
        """mel (B, 128, T) → (B, T_pad, 128, 1) NHWC,T pad 到 32 倍数。

        pad 保证 U-Net 各层 skip concat 的 H/W 对齐(torch mel2hidden 同样
        ``F.pad(mel, (0, 32*ceil - n_frames))``)。"""
        mel = mel.transpose(0, 2, 1)            # (B, T, 128)
        B, T, C = mel.shape
        t_pad = 32 * ((T - 1) // 32 + 1)
        if t_pad > T:
            mel = mx.pad(mel, [(0, 0), (0, t_pad - T), (0, 0)])
        return mel[:, :, :, None]               # (B, T_pad, 128, 1)

    def __call__(self, mel):
        """mel (B, 128, T) → salience (B, T_pad, 360)。"""
        x = self._to_nhwc(mel)                  # (B, T_pad, 128, 1)
        x = self.unet(x)                        # (B, T_pad, 128, 16)
        x = self.cnn(x)                         # (B, T_pad, 128, 3)
        B, Tp, W, C = x.shape
        # torch: cnn(...).transpose(1,2) (B,3,T,W) → (B,T,3,W);flatten(-2) → (B,T,3*W)
        x = x.transpose(0, 1, 3, 2)             # (B, Tp, 3, W)
        x = x.reshape(B, Tp, 3 * W)             # (B, Tp, 384)
        for layer in self.fc:
            x = layer(x)
        return x                                # (B, T_pad, 360)


class RMVPE(nn.Module):
    """MLX 版 RMVPE F0 估计器(E2E0 模型)。

    用法::
        m = RMVPE.from_ckpt("pretrained/rmvpe/model.pt")
        hidden = m(mel)          # mel (B, 128, T) → (B, T_pad, 360)
        f0 = m.decode(hidden, ...)
    """

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16,
                 hop_length: int = 160):
        super().__init__()
        self.model = E2E0(n_blocks, n_gru, kernel_size, en_de_layers,
                          inter_layers, in_channels, en_out_channels)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length,
            None, MEL_FMIN, MEL_FMAX,
        )
        self.hop_length = hop_length

    # -- 布局辅助(结构测试/外部调用用) ------------------------------------
    def _to_nhwc(self, mel):
        """mel (B, 128, T) → (B, T_pad, 128, 1)。"""
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

    # -- 推理 --------------------------------------------------------------

    def mel2hidden(self, mel):
        """mel (B, 128, T) → hidden (B, T, 360),截断回原帧数。

        对应 torch ``inference.py::RMVPE.mel2hidden``(pad → model → 截断)。
        """
        n_frames = mel.shape[-1]
        return self.model(mel)[:, :n_frames]

    def __call__(self, mel):
        """等价 mel2hidden(mel)。"""
        return self.mel2hidden(mel)

    def infer_from_audio(self, audio, sample_rate: int = SAMPLE_RATE,
                         thred: float = 0.03):
        """audio: 1d [L] 或 2d [B, L](16kHz 优先)→ f0 [T]。

        非 16kHz 经 librosa resample(torch 版用 soxr_lq)。mel/STFT 保持
        fp32,低精度只应用在 E2E0 前。
        """
        import librosa
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None, :]
        audio = audio.astype(mx.float32)
        if sample_rate != SAMPLE_RATE:
            wav_np = librosa.resample(
                np.array(audio[0]), orig_sr=sample_rate, target_sr=SAMPLE_RATE,
                res_type="soxr_lq",
            )
            audio = mx.array(wav_np.astype(np.float32))[None, :]
        mel = self.mel_extractor(audio, center=True)   # [B, 128, T]
        hidden = self.mel2hidden(mel)
        return self.decode(hidden, thred=thred)

    def decode(self, hidden, thred: float = 0.03):
        """salience (B, T, 360) → f0 (T,)(局部加权 argmax 解码,纯 numpy)。"""
        import numpy as np
        hidden_np = np.array(hidden)                    # (B, T, 360)
        cents = 20 * np.arange(N_CLASS) + 1997.3794084376191  # bin → cents
        out = []
        for b in range(hidden_np.shape[0]):
            sal = hidden_np[b]                          # (T, 360)
            centers = sal.argmax(axis=-1)               # (T,)
            f0_b = np.zeros(sal.shape[0], dtype=np.float32)
            for t in range(sal.shape[0]):
                c = int(centers[t])
                start = max(0, c - 4)
                end = min(N_CLASS, c + 5)
                w = sal[t, start:end]
                ps = float(np.sum(w * cents[start:end]))
                ws = float(np.sum(w))
                f0_b[t] = ps / ws if np.max(w) > thred else 0.0
            out.append(f0_b)
        return out[0] if len(out) == 1 else np.stack(out)

    # -- 权重加载 -----------------------------------------------------------

    @staticmethod
    def from_ckpt(ckpt_path: str | Path, dtype=None) -> "RMVPE":
        """加载 torch RMVPE ckpt(外层 {"model": state_dict})→ MLX RMVPE。

        处理:
          - Conv2d 权重 torch (O,C,kH,kW) → mlx (O,kH,kW,C)(启发式)
          - ConvTranspose2d 权重 torch (I,O,kH,kW) → mlx (O,kH,kW,I)
            (专用转置;启发式 (0,2,3,1) 对转置卷积错误)
          - weight_norm 合并(本模型无,但走统一管道安全)
          - GRU 权重 (2D) 原样(键名已对齐)
        """
        from ._convert import load_torch_sd, to_mlx_sd
        from ._ops import merge_weight_norm

        m = RMVPE()
        np_sd = load_torch_sd(ckpt_path)                # 解包 {"model": ...}
        merged = merge_weight_norm(np_sd)
        # torch ckpt 键是 E2E0 的 state_dict(unet.*/cnn.*/fc.*);mlx 侧
        # RMVPE 包装 self.model = E2E0,故加 model. 前缀(与 torch
        # ``self.model.load_state_dict(ckpt["model"])`` 语义一致)。
        merged = {("model." + k): v for k, v in merged.items()}
        # 分离 ConvTranspose2d 键(decoder 的 conv1.0)
        convt_keys = {k for k in merged if k.endswith(".conv1.0.weight")}
        # 普通 Conv2d 键:ndim==4 且非转置卷积
        conv2d_keys = (
            {k for k in merged if k.endswith(".weight") and np.asarray(merged[k]).ndim == 4}
            - convt_keys
        )
        mlx_sd = to_mlx_sd(merged, conv_transpose_keys=conv2d_keys, dtype=dtype)
        # 手动转置 ConvTranspose2d:torch (I,O,kH,kW) → mlx (O,kH,kW,I)
        for k in convt_keys:
            w = np.transpose(np.asarray(merged[k]), (1, 2, 3, 0))
            arr = mx.array(w)
            if dtype is not None:
                arr = arr.astype(dtype)
            mlx_sd[k] = arr
        m.load_weights(list(mlx_sd.items()), strict=False)
        m.eval()
        return m
