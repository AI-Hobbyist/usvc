"""MLX 版 FCPE F0 提取器(CFNaiveMelPE + ConformerNaiveEncoder + Wav2MelModule;
镜像 features/fcpe/:models.py / model_conformer_naive.py / mel_extractor.py /
models_infer.py)。配置反推自 pretrained/fcpe/fcpe.pt(config_dict):
mel sr=16k, num_mels=128, n_fft=win=1024, hop=160, fmin=0, fmax=8000;
model out_dims=360, hidden_dims=512, n_layers=6, n_heads=8, f0_min=32.7,
f0_max=1975.5, use_fa_norm=true, conv_only=true。

关键适配:conv_only=true → attn 分支 None;mel 提取 center=False + 手动 pad
(nvstft_pad 同款),librosa htk=False(fmin=0;⚠ RMVPE 是 htk=True),magnitude
=sqrt(re^2+im^2+1e-9)(torch.stft 语义);GroupNorm(4,hidden) 用
pytorch_compatible=True;output_proj weight_norm 由 from_ckpt 合并(dim=0);
Sequential/ModuleList → list 属性 + None 占位(键名与 torch 索引对齐);
Conv1d (O,C,K)→(O,K,C) 统一转置。
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ["FCPE", "CFNaiveMelPE", "ConformerNaiveEncoder", "CFNEncoderLayer",
           "ConformerConvModule", "DepthWiseConv1d", "FCPEMelExtractor"]

# ---- 配置(pretrained/fcpe/fcpe.pt config_dict) --------------------------
SAMPLE_RATE, N_MELS, N_FFT, WIN_SIZE, HOP_SIZE = 16000, 128, 1024, 1024, 160
MEL_FMIN, MEL_FMAX = 0.0, 8000.0
N_CLASS, HIDDEN_DIMS, N_LAYERS, N_HEADS = 360, 512, 6, 8
F0_MIN, F0_MAX = 32.7, 1975.5
THRESHOLD = 0.006  # torch InferCFNaiveMelPE.forward 默认阈值


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int,
                 padding: int, groups: int):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size,
                              padding=padding, groups=groups)

    def __call__(self, x):
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """torch ``net`` Sequential 键名对齐(NLC 无 Transpose;列表索引 =
    Sequential 索引,None 占位无参层 net.1/3/5/7/8)。"""

    def __init__(self, dim: int, expansion_factor: int = 2,
                 kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = kernel_size // 2  # calc_same_padding 奇数核 → pad[0]
        self.inner_dim = inner_dim
        self.net = [
            nn.LayerNorm(dim),                       # net.0
            None,                                    # net.1 Transpose
            nn.Conv1d(dim, inner_dim * 2, 1),        # net.2
            None,                                    # net.3 GLU(见 __call__)
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size,
                            padding=padding, groups=inner_dim),  # net.4
            None,                                    # net.5 SiLU
            nn.Conv1d(inner_dim, dim, 1),            # net.6
            None, None,                              # net.7 Transpose, net.8 Dropout
        ]

    def __call__(self, x):
        x = self.net[0](x)
        x = self.net[2](x)
        a, b = mx.split(x, 2, axis=-1)  # GLU(dim=1): 前 inner * sigmoid(后 inner)
        x = self.net[4](a * mx.sigmoid(b))
        return self.net[6](nn.silu(x))


class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, num_heads: int = 8,
                 use_norm: bool = False, conv_only: bool = False,
                 conv_dropout: float = 0.0, atten_dropout: float = 0.0):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model)
        self.norm = nn.LayerNorm(dim_model)
        # conv_dropout>0 时 torch 包 Sequential(ConvModule, Dropout);
        # 本 ckpt conv_dropout=0;attn=None 因 conv_only=true(FCPE ckpt)
        self.attn = None

    def __call__(self, x, mask=None):
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask=mask)
        return x + self.conformer(x)


class ConformerNaiveEncoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, dim_model: int,
                 use_norm: bool = False, conv_only: bool = False,
                 conv_dropout: float = 0.0, atten_dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.conv_only = conv_only
        self.encoder_layers = [
            CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only,
                            conv_dropout, atten_dropout)
            for _ in range(num_layers)
        ]

    def __call__(self, x, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


class CFNaiveMelPE(nn.Module):
    """input_stack → ConformerNaiveEncoder → norm → output_proj → sigmoid;
    mlx NLC 直接吃 (B,T,C),torch input_stack 吃 (B,C,T) 再转回(键名一致)。"""

    def __init__(self, input_channels: int = N_MELS, out_dims: int = N_CLASS,
                 hidden_dims: int = HIDDEN_DIMS, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, f0_max: float = F0_MAX,
                 f0_min: float = F0_MIN, use_fa_norm: bool = True,
                 conv_only: bool = True, conv_dropout: float = 0.0,
                 atten_dropout: float = 0.0, use_harmonic_emb: bool = False):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.use_fa_norm = use_fa_norm
        self.conv_only = conv_only
        self.harmonic_emb = None  # use_harmonic_emb=False(FCPE ckpt)
        self.input_stack = [  # torch Sequential 索引 0/1/2/3
            nn.Conv1d(input_channels, hidden_dims, 3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_dims, pytorch_compatible=True),
            None,  # nn.LeakyReLU
            nn.Conv1d(hidden_dims, hidden_dims, 3, stride=1, padding=1),
        ]
        self.net = ConformerNaiveEncoder(
            num_layers=n_layers, num_heads=n_heads, dim_model=hidden_dims,
            use_norm=use_fa_norm, conv_only=conv_only,
            conv_dropout=conv_dropout, atten_dropout=atten_dropout)
        self.norm = nn.LayerNorm(hidden_dims)
        # output_proj 是 weight_norm(Linear);from_ckpt 合并 weight_g/v(dim=0)
        self.output_proj = nn.Linear(hidden_dims, out_dims)

    def __call__(self, x):
        x = self.input_stack[0](x)
        x = self.input_stack[1](x)
        x = mx.maximum(x, 0.01 * x)  # LeakyReLU(negative_slope=0.01)
        x = self.input_stack[3](x)
        x = self.net(x)
        return mx.sigmoid(self.output_proj(self.norm(x)))

    def cent_table(self) -> mx.array:
        """cent bin 表(360,):linspace(1200*log2(f0_min/10), ..., out_dims)。
        numpy 求值与 torch linspace float64 差异 <1e-4,argmax decode 无影响。"""
        c_min = 1200.0 * np.log2(self.f0_min / 10.0)
        c_max = 1200.0 * np.log2(self.f0_max / 10.0)
        return mx.array(np.linspace(c_min, c_max, self.out_dims).astype(np.float32))


class FCPEMelExtractor(nn.Module):
    """center=False 手动 pad STFT + librosa mel(htk=False;⚠ RMVPE htk=True)。
    pad=nvstft_pad 同款(reflect/constant fallback);mag=sqrt(re^2+im^2+1e-9);
    log(clamp(·,1e-5));输出 (B,T,n_mels);n_frames 对齐 L//hop+1(截断/补尾帧)。"""

    def __init__(self, sr: int = SAMPLE_RATE, n_mels: int = N_MELS,
                 n_fft: int = N_FFT, win_size: int = WIN_SIZE,
                 hop_length: int = HOP_SIZE, fmin: float = MEL_FMIN,
                 fmax: float = MEL_FMAX, clip_val: float = 1e-5):
        super().__init__()
        import librosa
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=fmin, fmax=fmax, htk=False)
        self.mel_basis = mx.array(mel_basis.astype(np.float32))  # [n_mels, F]
        self.sr = sr
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.clip_val = clip_val

    def _stft_mag(self, wav: mx.array) -> mx.array:
        """1d wav(已 pad)→ mag [n_fft//2+1, frames](sqrt(re^2+im^2+1e-9))。"""
        win, hop = self.win_size, self.hop_length
        n_frames = max(1, (wav.shape[-1] - win) // hop + 1)
        idx = (mx.arange(n_frames) * hop)[:, None] + mx.arange(win)[None, :]
        frames = wav[idx] * (0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(win) / win))
        spec = mx.fft.rfft(frames, n=self.n_fft)
        re, im = spec.real, spec.imag
        return mx.sqrt(re * re + im * im + 1e-9).T

    def __call__(self, audio: mx.array) -> mx.array:
        """audio (B, L) 或 (L,)@16k → mel (B, T, n_mels)(调用方负责 resample)。"""
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None, :]
        B, L = audio.shape
        win, hop = self.win_size, self.hop_length
        pad_left = (win - hop) // 2
        pad_right = max((win - hop + 1) // 2, win - L - pad_left)
        from ._ops import pad_reflect1d_fallback
        mels = []
        for b in range(B):
            mel = self.mel_basis @ self._stft_mag(
                pad_reflect1d_fallback(audio[b], pad_left, pad_right))
            mels.append(mx.log(mx.maximum(mel, self.clip_val)).T)  # [frames, n_mels]
        mel = mx.stack(mels, axis=0)  # [B, T, n_mels]
        n_frames = L // hop + 1
        if mel.shape[1] > n_frames:
            mel = mel[:, :n_frames, :]
        elif mel.shape[1] < n_frames:
            mel = mx.concatenate(
                [mel, mx.broadcast_to(mel[:, -1:, :],
                                      (B, n_frames - mel.shape[1], mel.shape[-1]))],
                axis=1)
        return mel[0] if squeeze else mel


class FCPE(nn.Module):
    """F0 估计器(CFNaiveMelPE + mel 提取)。用法:m=FCPE.from_ckpt(path);
    f0=m.infer_from_audio(wav_16k, sample_rate=16000)  # [T] Hz
    infer 对齐 torch InferCFNaiveMelPE.infer:local_argmax, threshold=0.006,
    f0=10*2**(cents/1200), uv(f0<f0_min)→0。"""

    def __init__(self, hop_length: int = HOP_SIZE,
                 mel_cfg: dict | None = None):
        super().__init__()
        self.model = CFNaiveMelPE()
        self.mel_extractor = FCPEMelExtractor()
        self.hop_length = hop_length
        self.f0_min = F0_MIN
        self._cent = self.model.cent_table()

    def infer_from_audio(self, audio, sample_rate: int = SAMPLE_RATE,
                         threshold: float = THRESHOLD):
        """audio 1d [L] → f0 [T](Hz,0=无声);非 16k 经 librosa soxr_hq。"""
        import librosa
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None, :]
        audio = audio.astype(mx.float32)
        if sample_rate != SAMPLE_RATE:
            wav_np = librosa.resample(np.array(audio[0]), orig_sr=sample_rate,
                                      target_sr=SAMPLE_RATE, res_type="soxr_hq")
            audio = mx.array(wav_np.astype(np.float32))[None, :]
        return self.decode(self.model(self.mel_extractor(audio)),
                           threshold=threshold)  # [B, T, 360] → f0

    def decode(self, latent, threshold: float = THRESHOLD):
        """salience (B, T, 360) → f0(纯 numpy;镜像 latent2cents_local_decoder:
        局部 9-bin argmax±4 加权平均于 cents 域,confident<=threshold → 0)。"""
        latent_np = np.array(latent)
        cent = np.array(self._cent)
        out = []
        for b in range(latent_np.shape[0]):
            y = latent_np[b]  # (T, 360)
            T = y.shape[0]
            confident = y.max(axis=-1)
            max_idx = y.argmax(axis=-1)
            idx = np.clip(np.arange(0, 9)[None, :] + (max_idx - 4)[:, None],
                          0, self.model.out_dims - 1)  # (T, 9)
            y_l = y[np.arange(T)[:, None], idx]
            cents = (y_l * cent[idx]).sum(axis=-1) / (y_l.sum(axis=-1) + 1e-12)
            f0 = 10.0 * 2.0 ** (cents / 1200.0)
            f0[confident <= threshold] = 0.0
            f0[f0 < self.f0_min] = 0.0  # uv 过滤
            out.append(f0.astype(np.float32))
        return out[0] if len(out) == 1 else np.stack(out)

    @staticmethod
    def from_ckpt(ckpt_path: str | Path, dtype=None) -> "FCPE":
        """torch FCPE ckpt(顶层 {model, config_dict})→ MLX FCPE:
        merge_weight_norm(output_proj,dim=0) + Conv (O,C,K)→(O,K,C)(ndim==3
        启发式)+ model. 前缀;cent_table 等 buffer 运行时重建不加载。"""
        from ._convert import load_torch_sd, to_mlx_sd
        from ._ops import merge_weight_norm

        m = FCPE()
        merged = merge_weight_norm(load_torch_sd(ckpt_path))
        conv_keys = {k for k, v in merged.items()
                     if k.endswith(".weight") and np.asarray(v).ndim == 3}
        mlx_sd = to_mlx_sd(merged, conv_transpose_keys=conv_keys, dtype=dtype)
        m.load_weights(list({("model." + k): v for k, v in mlx_sd.items()}.items()),
                       strict=False)
        m.eval()
        return m