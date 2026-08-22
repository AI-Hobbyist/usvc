"""MLX 版 RVC wav-domain(GeneratorNSF + RVCModel;镜像 usvc/backends/rvc/model.py)。
与 torch 差异:删 channels_last 路径;输入 NLC [B,T,C](torch NCL);g 接受
[B,gin] 自动转 [B,1,gin];weight_norm 由 _convert 加载时合并。
⚠ RVC SineGen 的 rand_ini 只加到第一帧(与 vocoder classic 不同)。"""
import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

from .._ops import interpolate_linear, interpolate_nearest, sequence_mask
from ..nsf_hifigan import ResBlock1, ResBlock2
from .._gan import DiscriminatorP, DiscriminatorS, MultiPeriodDiscriminator

__all__ = [
    "RVCSineGen", "RVCSourceModuleHnNSF", "GeneratorNSF",
    "RVCTextEncoder", "RVCModel", "f0_to_coarse",
]

LRELU_SLOPE = 0.1


class RVCSineGen(nn.Module):
    """RVC sine generator(镜像 rvc/model.py::SineGen,harmonic_num=0;
    rand_ini 只加第一帧 rad_values[:,0,:],与 vocoder classic 不同)。"""

    def __init__(self, samp_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp, self.noise_std, self.harmonic_num = sine_amp, noise_std, harmonic_num
        self.dim, self.sampling_rate, self.voiced_threshold = harmonic_num + 1, samp_rate, voiced_threshold

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).astype(f0.dtype)

    def __call__(self, f0, upp: int):
        """f0 [B, T_frames] → (sine_waves [B,L,dim], uv [B,L,1], noise [B,L,dim])。"""
        f0 = f0[:, None, :].transpose(0, 2, 1)  # → [B, T, 1]
        B, T, _ = f0.shape
        f0_buf = mx.zeros((B, T, self.dim))
        f0_buf[:, :, 0] = f0[:, :, 0]
        for idx in range(self.harmonic_num):
            f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
        rad_values = (f0_buf / self.sampling_rate) % 1.0
        rand_ini = mx.random.uniform(shape=(B, self.dim))
        rand_ini[:, 0] = 0.0
        rad_values = rad_values.at[:, 0, :].add(rand_ini)
        tmp_over_one = mx.cumsum(rad_values, axis=1) * upp
        tmp_t = interpolate_linear(tmp_over_one.transpose(0, 2, 1), size=T * upp,
                                   align_corners=True).transpose(0, 2, 1)
        rad_values = interpolate_nearest(
            rad_values.transpose(0, 2, 1), scale_factor=float(upp)
        ).transpose(0, 2, 1)
        tmp_over_one = tmp_t % 1.0
        # 相位卷绕检测(<0 处需 shift)
        tmp_diff = tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]
        cumsum_shift = mx.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = (tmp_diff < 0) * -1.0
        sine_waves = mx.sin(mx.cumsum(rad_values + cumsum_shift, axis=1) * 2 * math.pi)
        sine_waves = sine_waves * self.sine_amp
        uv = self._f02uv(f0)
        uv = interpolate_nearest(uv.transpose(0, 2, 1), scale_factor=float(upp)).transpose(0, 2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class RVCSourceModuleHnNSF(nn.Module):
    """返回 (sine_merge, None, None)(noise/uv 为 None,torch 版如此)。"""

    def __init__(self, sampling_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, add_noise_std: float = 0.003,
                 voiced_threshod: float = 0, is_half: bool = True):
        super().__init__()
        self.l_sin_gen = RVCSineGen(sampling_rate, harmonic_num, sine_amp,
                                    add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x, upp=1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        return self.l_tanh(self.l_linear(sine_wavs)), None, None


class GeneratorNSF(nn.Module):
    """RVC wav-domain 生成器(conv_pre/conv_post 无 weight_norm;
    conv_post bias=False;resblock 复用 vocoder)。"""

    def __init__(self, initial_channel: int, resblock: str,
                 resblock_kernel_sizes: list, resblock_dilation_sizes: list,
                 upsample_rates: list, upsample_initial_channel: int,
                 upsample_kernel_sizes: list, gin_channels: int, sr: int,
                 is_half: bool = False):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

        self.m_source = RVCSourceModuleHnNSF(sampling_rate=sr, harmonic_num=0, is_half=is_half)
        self.noise_convs = []
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i), c_cur,
                                   k, stride=u, padding=(k - u) // 2)
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2,
                              stride=stride_f0, padding=stride_f0 // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, config: dict | None = None,
                  key_prefix: str | None = "generator.dec", strict: bool = False):
        """RVC ckpt 生成器权重在 ``generator.dec.*``(usvc 训练格式)或 ``dec.*``。"""
        from .._convert import load_torch_ckpt_to_mlx
        sd = load_torch_ckpt_to_mlx(
            ckpt_path, convtranspose_keys=cls._ups_weights(ckpt_path, key_prefix)
        )
        cfg = config if config is not None else cls._infer_config(ckpt_path, key_prefix, sd=sd)
        obj = cls(**cfg)
        if any(k.startswith("generator.") for k in sd):
            sd = {k[len("generator."):]: v for k, v in sd.items()}
        dec_sd = {}
        for k, v in sd.items():
            dec_sd[k[len("dec."):] if k.startswith("dec.") else k] = v
        if not dec_sd:
            raise ValueError(
                f"no '{prefix or 'dec'}' keys found in {ckpt_path}; "
                f"got keys like: {list(sd)[:5]}"
            )
        obj.load_weights(list(dec_sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _ups_weights(ckpt_path: str | Path, key_prefix: str | None = None) -> set[str]:
        """ConvTranspose1d 权重键(ups.*.weight,merge 后需 (I,O,K)→(O,K,I))。"""
        from .._convert import load_torch_sd
        from .._ops import merge_weight_norm
        merged = merge_weight_norm(load_torch_sd(ckpt_path))
        pref = (key_prefix + ".") if key_prefix else ""
        return {
            k for k in merged
            if k.startswith(pref + "ups.") and k.endswith(".weight")
        }

    @staticmethod
    def _infer_config(ckpt_path: str | Path | None = None, key_prefix: str | None = None,
                      sd: dict | None = None) -> dict:
        """从权重 shape 反推构造参数(upsample_rates 用默认 [10,8,2,2] 等)。"""
        if sd is None:
            if ckpt_path is None:
                raise ValueError("either ckpt_path or sd must be given")
            from .._convert import load_torch_ckpt_to_mlx
            sd = load_torch_ckpt_to_mlx(ckpt_path)
        prefix = (key_prefix + ".") if key_prefix else ""
        cand = {}
        for k, v in sd.items():
            if prefix and k.startswith(prefix):
                cand[k[len(prefix):]] = v
            elif not prefix and ".dec." in k:
                cand[k.split(".dec.")[-1]] = v
        shape = lambda k: cand[k].shape if k in cand else None  # noqa: E731
        conv_pre_w = shape("conv_pre.weight")
        conv_post_w = shape("conv_post.weight")
        cond_w = shape("cond.weight")
        if conv_pre_w is None or conv_post_w is None:
            raise ValueError(
                f"cannot infer GeneratorNSF config from {ckpt_path}: missing conv_pre/conv_post"
            )
        initial_channel = conv_pre_w[2]   # mlx (O, K, C)
        upsample_initial_channel = conv_pre_w[0]
        ups = sorted([k for k in cand if k.startswith("ups.") and k.endswith(".weight")],
                     key=lambda k: int(k.split(".")[1]))
        num_ups = len(ups)
        # mlx ConvTranspose1d weight (out, k, in)(torch (in,out,k) 已转置)
        kernel_sizes = [int(cand[uk].shape[1]) for uk in ups]
        # upsample_rates 无法从 shape 推断(在 stride 里)
        if num_ups == 4:
            upsample_rates = [10, 8, 2, 2]
        elif num_ups == 3:
            upsample_rates = [10, 8, 2]
        else:
            upsample_rates = [8, 6, 4, 2][:num_ups]
        gin_channels = cond_w[2] if cond_w is not None else 0  # mlx (O, 1, C) 的 C=gin
        return dict(
            initial_channel=initial_channel,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=kernel_sizes or [20, 16, 4, 4][:num_ups],
            gin_channels=gin_channels,
            sr=48000,
        )

    def __call__(self, x, f0, g=None, n_res=None):
        """x [B,T,initial]; f0 [B,T_frames] Hz; g [B,gin] 或 [B,1,gin] → [B,T_out,1]。"""
        har_source, _, _ = self.m_source(f0, self.upp)   # [B, L, 1] (NLC)
        if n_res is not None:
            n = int(n_res)
            if n * self.upp != har_source.shape[1]:
                har_source = interpolate_linear(
                    har_source.transpose(0, 2, 1), size=n * self.upp, align_corners=True
                ).transpose(0, 2, 1)
            if n != x.shape[1]:
                x = interpolate_linear(x.transpose(0, 2, 1), size=n,
                                       align_corners=True).transpose(0, 2, 1)
        x = self.conv_pre(x)
        if g is not None:
            if g.ndim == 2:
                g = g[:, None, :]  # [B, 1, gin]
            x = x + self.cond(g)
        for i, (ups, noise_conv) in enumerate(zip(self.ups, self.noise_convs)):
            if i < self.num_upsamples:
                x = ups(nn.leaky_relu(x, self.lrelu_slope))
                x_source = noise_conv(har_source)
                minl = min(x.shape[1], x_source.shape[1])
                x = x[:, :minl, :] + x_source[:, :minl, :]
                xs = None
                l = [i * self.num_kernels + j for j in range(self.num_kernels)]
                for j, resblock in enumerate(self.resblocks):
                    if j in l:
                        rb = resblock(x)
                        xs = rb if xs is None else xs + rb
                x = xs / self.num_kernels
        return mx.tanh(self.conv_post(nn.leaky_relu(x, self.lrelu_slope)))


def f0_to_coarse(f0, f0_min: float = 50.0, f0_max: float = 1100.0):
    """f0 [B,T] Hz → tokens [1,255](mel 刻度)。
    ⚠ 与 sovits.f0_to_coarse 数学等价但 FP 舍入不同:220k 样本实测 1 个
    round 边界 token 差(f0≈942.9Hz, rvc=229 vs sovits=228);不可合并。"""
    f0 = f0.astype(mx.float32)
    f0_mel_min = 1127.0 * mx.log(1.0 + f0_min / 700.0)
    f0_mel_max = 1127.0 * mx.log(1.0 + f0_max / 700.0)
    f0_mel = 1127.0 * mx.log(1.0 + f0 / 700.0)
    f0_mel = mx.where(
        f0_mel > 0,
        (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0,
        f0_mel,
    )
    return mx.round(mx.clip(f0_mel, 1.0, 255.0)).astype(mx.int32)


class RVCTextEncoder(nn.Module):
    """emb_phone + emb_pitch + Encoder(window_size=10)+ proj。"""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int, p_dropout: float, f0: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        from .sovits import Encoder  # RVC window_size=10(so-vits 4)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers,
                               kernel_size, float(p_dropout), window_size=10)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(self, phone, pitch, lengths, skip_head=None, vol_emb=None):
        x = self.emb_phone(phone) if pitch is None else self.emb_phone(phone) + self.emb_pitch(pitch)
        if vol_emb is not None:
            x = x + vol_emb
        x = self.lrelu(x * math.sqrt(self.hidden_channels))   # [b, t, h]
        x_mask = sequence_mask(lengths, x.shape[1])[..., None].astype(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        if skip_head is not None:
            head = int(skip_head)
            x = x[:, head:, :]
            x_mask = x_mask[:, head:, :]
        stats = self.proj(x) * x_mask
        m = stats[..., :self.out_channels]
        logs = stats[..., self.out_channels:]
        return m, logs, x_mask


class RVCModel(nn.Module):
    """RVC v2 推理(emb_g + enc_p + flow + dec;z = m 确定性,无采样噪声)。
    默认参数对应 RVC configs/v2/48k.json:inter=192, hidden=192, filter=768,
    n_heads=2, n_layers=6, kernel=3, upsample=[12,10,2,2], gin=256, sr=48k。"""

    def __init__(
        self,
        n_units=768,
        spec_channels=1025,
        segment_size=20,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(12, 10, 2, 2),
        upsample_initial_channel=512,
        upsample_kernel_sizes=(24, 20, 4, 4),
        spk_embed_dim=1,
        gin_channels=256,
        sr=48000,
        is_half=False,
        vol_embedding=False,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.enc_p = RVCTextEncoder(n_units, inter_channels, hidden_channels,
                                    filter_channels, n_heads, n_layers,
                                    kernel_size, float(p_dropout))
        self.dec = GeneratorNSF(inter_channels, resblock,
                                list(resblock_kernel_sizes),
                                [list(d) for d in resblock_dilation_sizes],
                                list(upsample_rates), upsample_initial_channel,
                                list(upsample_kernel_sizes),
                                gin_channels=gin_channels, sr=sr, is_half=is_half)
        from .sovits import ResidualCouplingBlock  # flow 同 so-vits(mean_only,n_flows=4)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3,
                                          gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        self.vol_embedding = vol_embedding
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)
        from .sovits import EncoderQ as _EncoderQ  # 训练 posterior(同 torch rvc)
        self.enc_q = _EncoderQ(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                               gin_channels=gin_channels)
        self.spec_channels = spec_channels

    def infer(self, phone, f0, pitch, lengths, spk_id=0, skip_head=None,
              return_length=None, vol=None, noice_scale: float = 0.66666):
        """确定性推理链:enc_p → flow(reverse)→ dec;z = m + exp(logs)*noice_scale。"""
        g = self.emb_g(spk_id)
        if g.ndim == 1:
            g = g[None, :]
        g = g[:, None, :]  # [B, 1, gin](NLC)
        vol_emb = (
            self.emb_vol(vol[:, :, None])
            if vol is not None and self.vol_embedding
            else None
        )
        if skip_head is not None and return_length is not None:
            head = int(skip_head)
            length = int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m, logs, x_mask = self.enc_p(phone, pitch, lengths, skip_head=flow_head)
            z_p = (m + mx.exp(logs) * noice_scale) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, dec_head:dec_head + length, :]
            x_mask = x_mask[:, dec_head:dec_head + length, :]
            f0 = f0[:, head:head + length]
        else:
            m, logs, x_mask = self.enc_p(phone, pitch, lengths)
            z_p = (m + mx.exp(logs) * noice_scale) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        return self.dec(z * x_mask, f0, g=g)

    def f0_to_coarse(self, f0, f0_min=50.0, f0_max=1100.0):
        return f0_to_coarse(f0, f0_min, f0_max)

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, config: dict | None = None,
                  key_prefix: str | None = None, strict: bool = False):
        """权重在 ``generator.*``(usvc 训练格式)或裸键 ``dec.*``/``enc_p.*``。"""
        from .._convert import load_torch_ckpt_to_mlx
        obj = cls(**(config if config is not None else {}))
        prefix = key_prefix
        if prefix is None:
            from .._convert import load_torch_sd
            raw = load_torch_sd(ckpt_path)
            prefix = "generator" if any(k.startswith("generator.") for k in raw) else ""
        pref = (prefix + ".") if prefix else ""
        sd = load_torch_ckpt_to_mlx(
            ckpt_path,
            convtranspose_keys=GeneratorNSF._ups_weights(
                ckpt_path, key_prefix=(prefix + ".dec") if prefix else "dec"
            ),
        )
        model_sd = {}
        for k, v in sd.items():
            mk = k
            if pref and k.startswith(pref):
                mk = k[len(pref):]
            elif not pref and ("." in k and (k.startswith("enc_p.") or k.startswith("dec.")
                                             or k.startswith("flow.") or k.startswith("emb_g.")
                                             or k.startswith("enc_q."))):
                mk = k.split(".", 1)[-1]
            model_sd[mk] = v
        obj.load_weights(list(model_sd.items()), strict=strict)
        obj.eval()
        return obj


def _rvc_rand_slice_segments_with_pitch(z, pitch, x_lengths, segment_size):
    """随机 slice 段 → (z_slice, pitch_slice, ids_str)。"""
    B, T, C = z.shape
    ids_str_max = (x_lengths - segment_size + 1).astype(mx.int32)
    r = mx.random.uniform(shape=(B,)).astype(mx.float32) * ids_str_max.astype(mx.float32)
    ids_str = r.astype(mx.int32)
    idx = ids_str[:, None] + mx.arange(segment_size)[None, :]
    return mx.take_along_axis(z, idx[..., None], axis=1), \
        mx.take_along_axis(pitch, idx, axis=1), ids_str


def _rvc_train_forward(self, units, pitch_token, pitchf, spec, wave, spk_id,
                       frame_len=None, vol=None):
    """RVC 训练 forward → dict(wave_gen, z_p, m_p, logs_p, m_q, logs_q,
    z_mask, ids_slice, phone_lengths)。wave 未用(API parity)。"""
    B = units.shape[0]
    if spk_id is None:
        spk_id = mx.zeros((B,), dtype=mx.int32)
    if spk_id.ndim == 1:
        spk_id = spk_id[:, None]
    g = self.emb_g(spk_id)                     # [B, 1, gin](NLC)
    phone_lengths = (
        frame_len if frame_len is not None
        else mx.full((B,), units.shape[1], dtype=mx.int32)
    )
    spec_lengths = phone_lengths
    vol_emb = (
        self.emb_vol(vol[:, :, None])
        if vol is not None and self.vol_embedding
        else None
    )
    m_p, logs_p, x_mask = self.enc_p(units, pitch_token, phone_lengths, vol_emb=vol_emb)
    z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)
    z_slice, pitchf_slice, ids_slice = _rvc_rand_slice_segments_with_pitch(
        z, pitchf, spec_lengths, self.segment_size
    )
    o = self.dec(z_slice, pitchf_slice, g=g)   # [B, seg*hop, 1]
    return {
        "wave_gen": o,
        "z_p": z_p,
        "m_p": m_p,
        "logs_p": logs_p,
        "m_q": m_q,
        "logs_q": logs_q,
        "z_mask": y_mask,
        "ids_slice": ids_slice,
        "phone_lengths": phone_lengths,
    }


def _rvc_forward_train(self, units, pitch_token, pitchf, spec, wave, spk_id,
                       frame_len=None, vol=None):
    return _rvc_train_forward(self, units, pitch_token, pitchf, spec, wave, spk_id,
                              frame_len=frame_len, vol=vol)


RVCModel.forward_train = _rvc_forward_train


class MultiPeriodDiscriminatorV2(MultiPeriodDiscriminator):
    """RVC v2 MPD: periods (2,3,5,7,11,17,23,37)。"""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__(periods=(2, 3, 5, 7, 11, 17, 23, 37),
                         use_spectral_norm=use_spectral_norm)