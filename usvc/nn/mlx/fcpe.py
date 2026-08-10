"""MLX 版 FCPE F0 提取器(CFNaiveMelPE + ConformerNaiveEncoder)。

镜像 ``usvc/features/fcpe/`` 的 torch 实现:
  - ``models.py``: CFNaiveMelPE(input_stack + ConformerNaiveEncoder + norm + output_proj)
  - ``model_conformer_naive.py``: ConformerNaiveEncoder(6 层 CFNEncoderLayer)
  - ``mel_extractor.py``: Wav2MelModule(center=False 手动 pad STFT + librosa mel)
  - ``models_infer.py``: spawn_infer_model_from_pt(权重加载与推理)

配置(从 ``pretrained/fcpe/fcpe.pt`` config_dict 反推):
  mel: sr=16000, num_mels=128, n_fft=1024, win_size=1024, hop_size=160,
       fmin=0, fmax=8000
  model: CFNaiveMelPE, out_dims=360, hidden_dims=512, n_layers=6, n_heads=8,
         f0_min=32.7, f0_max=1975.5, use_fa_norm=true, conv_only=true

关键 MLX 适配:
  - 本 ckpt 是 **conv_only=true**(无 attention 权重)。CFNEncoderLayer 的
    attn 分支置 None,只保留 conformer 残差;use_fa_norm=true 仅影响
    FastAttention 分支,conv_only 下不生效。
  - mel 提取与 RMVPE 不同:**center=False** + 手动 pad(``nvstft_pad``
    同款 reflect/constant fallback),librosa mel **fmin=0 / htk=False**,
    magnitude = sqrt(re^2+im^2+1e-9)(torch.stft 语义,非纯 abs)。
  - GroupNorm(4, hidden):mlx ``nn.GroupNorm`` 沿最后一维(NLC),torch
    GroupNorm 同样沿 channel 维 —— 布局一致,键名 weight/bias 相同。
  - weight_norm:``output_proj`` 是 weight_norm(Linear),from_ckpt 里用
    ``merge_weight_norm`` 合并(dim=0)。
  - 顺序容器(Sequential / ModuleList)用 **list 属性**,键名与 torch 索引
    完全对齐(含 None 占位,如 conformer net.1/3/5/7 无参)。
  - Conv1d 权重:torch (O,C,K) → mlx (O,K,C),统一转置。
  - decode:local_argmax(9-bin 局部加权平均,cents 域),与 torch
    ``latent2cents_local_decoder`` 逐位一致,threshold=0.006,
    f0_min 过滤(uv → 0)。
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ["FCPE", "CFNaiveMelPE", "ConformerNaiveEncoder",
           "CFNEncoderLayer", "ConformerConvModule", "DepthWiseConv1d",
           "FCPEMelExtractor"]

# ---- 配置(pretrained/fcpe/fcpe.pt config_dict) --------------------------
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
WIN_SIZE = 1024
HOP_SIZE = 160
MEL_FMIN = 0.0
MEL_FMAX = 8000.0
N_CLASS = 360
HIDDEN_DIMS = 512
N_LAYERS = 6
N_HEADS = 8
F0_MIN = 32.7
F0_MAX = 1975.5
# torch InferCFNaiveMelPE.forward 默认阈值
THRESHOLD = 0.006


# ---------------------------------------------------------------------------
# DepthWiseConv1d / ConformerConvModule(镜像 model_conformer_naive.py)
# ---------------------------------------------------------------------------

class DepthWiseConv1d(nn.Module):
    """分组卷积包装(torch 同名类;键名 = conv.weight)。"""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int,
                 padding: int, groups: int):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size,
                              padding=padding, groups=groups)

    def __call__(self, x):
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """Conformer 卷积模块(torch ``net`` Sequential 的键名对齐)。

    torch: LayerNorm → Transpose → Conv1d(dim→2*inner) → GLU →
           DepthWiseConv1d → SiLU → Conv1d(inner→dim) → Transpose → Dropout
    mlx:   NLC 布局,无 Transpose;GLU 用 axis=-1(等价 torch dim=1)。
    列表 + None 占位让键名与 torch Sequential 索引一致
    (net.0/norm, net.2/conv, net.4.conv, net.6/conv)。
    """

    def __init__(self, dim: int, expansion_factor: int = 2,
                 kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = kernel_size // 2          # calc_same_padding 奇数核 → pad[0]
        self.inner_dim = inner_dim
        # 列表索引 ↔ torch Sequential 索引;None 占位无参层(Transpose/GLU/SiLU/Dropout)
        self.net = [
            nn.LayerNorm(dim),              # net.0
            None,                            # net.1 Transpose(mlx 不需要)
            nn.Conv1d(dim, inner_dim * 2, 1),  # net.2
            None,                            # net.3 GLU(mlx 用 axis=-1,见 __call__)
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size,
                            padding=padding, groups=inner_dim),  # net.4
            None,                            # net.5 SiLU
            nn.Conv1d(inner_dim, dim, 1),    # net.6
            None,                            # net.7 Transpose(mlx 不需要)
            None,                            # net.8 Dropout(推理恒等)
        ]

    def __call__(self, x):
        x = self.net[0](x)                  # LayerNorm
        x = self.net[2](x)                  # (B, T, 2*inner)
        # GLU(dim=1) torch: 前 inner 通道 * sigmoid(后 inner 通道)
        a, b = mx.split(x, 2, axis=-1)
        x = a * mx.sigmoid(b)               # (B, T, inner)
        x = self.net[4](x)                  # DepthWiseConv1d
        x = nn.silu(x)
        x = self.net[6](x)                  # (B, T, dim)
        return x


class CFNEncoderLayer(nn.Module):
    """Conformer 编码层(残差)。conv_only=true 时无 attention。"""

    def __init__(self, dim_model: int, num_heads: int = 8,
                 use_norm: bool = False, conv_only: bool = False,
                 conv_dropout: float = 0.0, atten_dropout: float = 0.0):
        super().__init__()
        self.conformer = ConformerConvModule(dim_model)
        self.norm = nn.LayerNorm(dim_model)
        # torch: conv_dropout>0 时包 Sequential(ConvModule, Dropout);
        # 本 ckpt conv_dropout=0,直接是 ConformerConvModule。
        self.attn = None                    # conv_only=true(FCPE ckpt)

    def __call__(self, x, mask=None):
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask=mask)
        x = x + self.conformer(x)
        return x


class ConformerNaiveEncoder(nn.Module):
    """N 层 CFNEncoderLayer。"""

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


# ---------------------------------------------------------------------------
# CFNaiveMelPE(镜像 models.py)
# ---------------------------------------------------------------------------

class CFNaiveMelPE(nn.Module):
    """input_stack → ConformerNaiveEncoder → norm → output_proj → sigmoid。

    forward 输入 mel (B, T, input_channels),输出 latent (B, T, out_dims)。
    torch 侧 input_stack 吃 (B,C,T) 再 transpose 回 (B,T,C);mlx NLC 直接吃
    (B,T,C),键名一致(input_stack.0/1/3)。
    """

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
        self.harmonic_emb = None            # use_harmonic_emb=False(FCPE ckpt)

        # input_stack: Conv1d → GroupNorm(4) → LeakyReLU → Conv1d
        self.input_stack = [
            nn.Conv1d(input_channels, hidden_dims, 3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_dims),
            None,                            # nn.LeakyReLU(torch Sequential 索引 2)
            nn.Conv1d(hidden_dims, hidden_dims, 3, stride=1, padding=1),
        ]
        self.net = ConformerNaiveEncoder(
            num_layers=n_layers, num_heads=n_heads, dim_model=hidden_dims,
            use_norm=use_fa_norm, conv_only=conv_only,
            conv_dropout=conv_dropout, atten_dropout=atten_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dims)
        # output_proj: weight_norm(Linear);ckpt 里是 weight_g/weight_v,
        # from_ckpt 用 merge_weight_norm 合并(dim=0)。
        self.output_proj = nn.Linear(hidden_dims, out_dims)

    def __call__(self, x):
        """x: mel (B, T, input_channels) → latent (B, T, out_dims)。"""
        for layer in self.input_stack:
            if layer is not None:
                x = layer(x)
        x = self.net(x)
        x = self.norm(x)
        x = self.output_proj(x)
        return mx.sigmoid(x)

    # -- cent 表(推理 decode 用) ------------------------------------------

    def cent_table(self) -> mx.array:
        """cent bin 表(360,),与 torch 注册 buffer 逐位一致。

        torch: linspace(f0_to_cent(f0_min), f0_to_cent(f0_max), out_dims),
        其中 f0_to_cent = 1200*log2(f0/10)。用 numpy 在构造时求值(与 torch
        linspace 的 float64 差异 < 1e-4,对 argmax decode 无影响)。
        """
        c_min = 1200.0 * np.log2(self.f0_min / 10.0)
        c_max = 1200.0 * np.log2(self.f0_max / 10.0)
        return mx.array(np.linspace(c_min, c_max, self.out_dims).astype(np.float32))


# ---------------------------------------------------------------------------
# FCPE mel 提取(Wav2MelModule / MelModule 镜像)
# ---------------------------------------------------------------------------

class FCPEMelExtractor(nn.Module):
    """center=False 手动 pad STFT + librosa mel(htk=False, fmin=0)。

    镜像 ``mel_extractor.py``:
      - pad_left=(win-hop)//2, pad_right=max((win-hop+1)//2, win-L-pad_left)
        mode = reflect if pad_right < L else constant(``nvstft_pad``)
      - magnitude = sqrt(re^2 + im^2 + 1e-9)
      - keyshift=0 时直接 mel_basis @ spec,log(clamp(x, 1e-5))
      - 输出 (B, T, n_mels)(transpose 后时间在前)
      - n_frames 对齐:audio_res//hop + 1,截断或补尾帧
    """

    def __init__(self, sr: int = SAMPLE_RATE, n_mels: int = N_MELS,
                 n_fft: int = N_FFT, win_size: int = WIN_SIZE,
                 hop_length: int = HOP_SIZE, fmin: float = MEL_FMIN,
                 fmax: float = MEL_FMAX, clip_val: float = 1e-5):
        super().__init__()
        import librosa
        mel_basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
            htk=False,  # FCPE 用默认(与 RMVPE 的 htk=True 不同)
        )
        self.mel_basis = mx.array(mel_basis.astype(np.float32))  # [n_mels, n_fft//2+1]
        self.sr = sr
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.clip_val = clip_val

    def _stft_mag(self, wav: mx.array) -> mx.array:
        """1d wav(已 pad)→ magnitude [n_fft//2+1, frames](sqrt(re^2+im^2+1e-9))。"""
        win = self.win_size
        hop = self.hop_length
        n_frames = (wav.shape[-1] - win) // hop + 1
        if n_frames <= 0:
            n_frames = 1
        # 手动 unfold(frames[i] = wav[i*hop : i*hop+win])
        starts = mx.arange(n_frames) * hop
        offsets = mx.arange(win)
        idx = starts[:, None] + offsets[None, :]
        frames = wav[idx]                                  # [n_frames, win]
        window = 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(win) / win)
        frames = frames * window
        spec = mx.fft.rfft(frames, n=self.n_fft)           # [n_frames, n_fft//2+1] cplx
        # torch.stft magnitude: sqrt(re^2 + im^2 + 1e-9)
        re, im = spec.real, spec.imag
        mag = mx.sqrt(re * re + im * im + 1e-9)
        return mag.T                                       # [n_fft//2+1, frames]

    def __call__(self, audio: mx.array) -> mx.array:
        """audio (B, L) 或 (L,)→ mel (B, T, n_mels)(16k 域)。

        与 torch Wav2MelModule 一致:输入已假设 @16k(调用方 resample)。
        """
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
            padded = pad_reflect1d_fallback(audio[b], pad_left, pad_right)
            spec = self._stft_mag(padded)                  # [513, frames]
            mel = self.mel_basis @ spec                    # [n_mels, frames]
            mel = mx.log(mx.maximum(mel, self.clip_val))
            mels.append(mel.T)                             # [frames, n_mels]
        mel = mx.stack(mels, axis=0)                       # [B, T, n_mels]
        # n_frames 对齐(audio_res // hop + 1):截断或补尾帧
        n_frames = L // hop + 1
        if mel.shape[1] > n_frames:
            mel = mel[:, :n_frames, :]
        elif mel.shape[1] < n_frames:
            pad = mel[:, -1:, :]
            reps = n_frames - mel.shape[1]
            mel = mx.concatenate([mel, mx.broadcast_to(pad, (B, reps, mel.shape[-1]))], axis=1)
        return mel[0] if squeeze else mel


# ---------------------------------------------------------------------------
# FCPE(镜像 models_infer.py::InferCFNaiveMelPE)
# ---------------------------------------------------------------------------

class FCPE(nn.Module):
    """MLX 版 FCPE F0 估计器(CFNaiveMelPE + mel 提取)。

    用法::
        m = FCPE.from_ckpt("pretrained/fcpe/fcpe.pt")
        f0 = m.infer_from_audio(mx.array(wav_16k), sample_rate=16000)  # [T] Hz

    infer 语义对齐 torch ``InferCFNaiveMelPE.infer``:
      - decoder="local_argmax", threshold=0.006
      - f0 = 10 * 2**(cents/1200)
      - uv = f0 < f0_min → 0
    """

    def __init__(self, hop_length: int = HOP_SIZE,
                 mel_cfg: dict | None = None):
        super().__init__()
        self.model = CFNaiveMelPE()
        self.mel_extractor = FCPEMelExtractor()
        self.hop_length = hop_length
        self.f0_min = F0_MIN
        self._cent = self.model.cent_table()

    # -- 推理 --------------------------------------------------------------

    def infer_from_audio(self, audio, sample_rate: int = SAMPLE_RATE,
                         threshold: float = THRESHOLD):
        """audio 1d [L] → f0 [T](Hz,0=无声)。

        非 16k 输入经 librosa 重采样到 16k(soxr_hq)。mel/模型保持 fp32。
        """
        import librosa
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None, :]
        audio = audio.astype(mx.float32)
        if sample_rate != SAMPLE_RATE:
            wav_np = librosa.resample(
                np.array(audio[0]), orig_sr=sample_rate, target_sr=SAMPLE_RATE,
                res_type="soxr_hq",
            )
            audio = mx.array(wav_np.astype(np.float32))[None, :]
        mel = self.mel_extractor(audio)        # [B, T, 128]
        latent = self.model(mel)               # [B, T, 360]
        return self.decode(latent, threshold=threshold)

    def decode(self, latent, threshold: float = THRESHOLD):
        """salience (B, T, 360) → f0 (T,)(local_argmax,纯 numpy)。

        镜像 torch ``latent2cents_local_decoder``:
          局部 9-bin 窗口(argmax ± 4),加权平均在 cents 域,
          f0 = 10 * 2**(cents/1200);confident <= threshold → 无声(0)。
        """
        import numpy as np
        latent_np = np.array(latent)                      # (B, T, 360)
        cent = np.array(self._cent)                       # (360,)
        out = []
        for b in range(latent_np.shape[0]):
            y = latent_np[b]                              # (T, 360)
            T = y.shape[0]
            confident = y.max(axis=-1)                    # (T,)
            max_idx = y.argmax(axis=-1)                   # (T,)
            # local argmax 索引:arange(0,9) + (max_idx - 4),clamp [0, 359]
            idx = np.clip(np.arange(0, 9)[None, :] + (max_idx - 4)[:, None],
                          0, self.model.out_dims - 1)     # (T, 9)
            y_l = y[np.arange(T)[:, None], idx]           # (T, 9)
            c_l = cent[idx]                               # (T, 9)
            cents = (y_l * c_l).sum(axis=-1) / (y_l.sum(axis=-1) + 1e-12)
            f0 = 10.0 * 2.0 ** (cents / 1200.0)
            f0[confident <= threshold] = 0.0
            # uv 过滤(f0 < f0_min → 0)
            f0[f0 < self.f0_min] = 0.0
            out.append(f0.astype(np.float32))
        return out[0] if len(out) == 1 else np.stack(out)

    # -- 权重加载 -----------------------------------------------------------

    @staticmethod
    def from_ckpt(ckpt_path: str | Path, dtype=None) -> "FCPE":
        """加载 torch FCPE ckpt(顶层 {model: state_dict, config_dict})→ MLX FCPE。

        处理:
          - merge_weight_norm(output_proj.weight_g/v → weight,dim=0)
          - Conv1d 权重 torch (O,C,K) → mlx (O,K,C)(启发式 ndim==3)
          - buffers(cent_table / gaussian_blurred_cent_mask)由
            CFNaiveMelPE.cent_table 运行时重建,不加载
        """
        from ._convert import load_torch_sd, to_mlx_sd
        from ._ops import merge_weight_norm

        m = FCPE()
        np_sd = load_torch_sd(ckpt_path)      # 解包 {"model": ...}
        merged = merge_weight_norm(np_sd)
        # 本 ckpt 的 conv 权重键:input_stack.*/conformer net.*(ndim==3)
        conv_keys = {
            k for k, v in merged.items()
            if k.endswith(".weight") and np.asarray(v).ndim == 3
        }
        mlx_sd = to_mlx_sd(merged, conv_transpose_keys=conv_keys, dtype=dtype)
        # 模型键带 model. 前缀(FCPE 包装 self.model = CFNaiveMelPE)
        mlx_sd = {("model." + k): v for k, v in mlx_sd.items()}
        m.load_weights(list(mlx_sd.items()), strict=False)
        m.eval()
        return m
