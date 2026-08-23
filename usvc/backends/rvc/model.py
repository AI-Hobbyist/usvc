"""RVC v2 training model: SynthesizerTrnMs768NSFsid + MultiPeriodDiscriminatorV2.
Extracted verbatim from RVC-Project/Retrieval-based-Voice-Conversion-WebUI
(infer/module/models.py + modules.py; torchaudio/scipy removed). Shared
numerically-identical components come from sovits.model (Encoder, EncoderQ,
ResidualCouplingBlock, DiscriminatorS/P, sequence_mask, get_padding,
init_weights, rand_slice_segments_with_pitch, LRELU_SLOPE). RVC-specific
algorithms (differ from so-vits, kept verbatim): SineGen, SourceModuleHnNSF,
ResBlock1/2 (RVC signatures), GeneratorNSF (conv_pre/conv_post WITHOUT
weight_norm, conv_post bias=False, padding=(k-u)//2, m_source harmonic_num=0),
TextEncoder, RVCModel, MultiPeriodDiscriminatorV2 (periods [2,3,5,7,11,17,23,37]).
"""
from math import prod, sqrt

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from ...nn.cl_helper import conv2d_cl, convT2d_cl
from ..sovits.model import (
    DiscriminatorP,
    DiscriminatorS,
    Encoder,
    EncoderQ,
    LRELU_SLOPE,
    ResidualCouplingBlock,
    get_padding,
    init_weights,
    rand_slice_segments_with_pitch,
    sequence_mask,
)

sr2sr = {"32k": 32000, "40k": 40000, "48k": 48000}


class SineGen(nn.Module):
    # RVC sine excitation; Triton fused frame-domain kernel
    # (usvc._triton.fused_sinegen) is bit-equivalent to the sample-domain path
    # for the RVC config (harmonic_num=0 -> dim=1, rand_ini恒0) and drops the
    # old @torch.compiler.disable so GeneratorNSF can stay inside a CUDA graph.
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp, self.noise_std, self.harmonic_num = sine_amp, noise_std, harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate, self.voiced_threshold = samp_rate, voiced_threshold

    def forward(self, f0, upp):
        from ..._triton.fused_sinegen import fused_sinegen
        with torch.no_grad():
            f0_in = f0[:, :, 0] if f0.dim() == 3 else f0
            B, T = f0_in.shape
            # noise 在此生成传入 kernel（fixed seed 可复现）；返回接口与原版一致
            noise = torch.randn(B, T * upp, self.dim, device=f0_in.device, dtype=torch.float32)
            sine_waves = fused_sinegen(f0_in, upp, self.dim, self.sampling_rate,
                                       self.sine_amp, self.noise_std, rand_ini=None, noise=noise)
            uv = torch.ones_like(f0_in) * (f0_in > self.voiced_threshold)
            if uv.device.type == "privateuseone":  # DirectML
                uv = uv.float()
            uv = F.interpolate(uv.unsqueeze(-1).transpose(2, 1),
                               scale_factor=float(upp), mode="nearest").transpose(2, 1)
            return sine_waves.to(f0.dtype if f0.dim() == 2 else f0_in.dtype), uv, noise


class SourceModuleHnNSF(nn.Module):  # hn-nsf source，返回 (sine_merge, None, None)
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0, is_half=True):
        super().__init__()
        self.sine_amp, self.noise_std, self.is_half = sine_amp, add_noise_std, is_half
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)  # merge source harmonics into one excitation
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp=1):
        sine_wavs, _, _ = self.l_sin_gen(x, upp)
        return self.l_tanh(self.l_linear(sine_wavs.to(dtype=self.l_linear.weight.dtype))), None, None


def _wconv(channels, k, d):
    return weight_norm(Conv1d(channels, channels, k, 1, dilation=d, padding=get_padding(k, d)))


class ResBlock1(nn.Module):  # RVC hifigan resblock 1（dilation 组 + 1-dilation 组）
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([_wconv(channels, kernel_size, d) for d in dilation])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([_wconv(channels, kernel_size, 1) for _ in dilation])
        self.convs2.apply(init_weights)
        self.lrelu_slope = LRELU_SLOPE

    def _fwd(self, x, x_mask, cnv):
        m, s = (x_mask if x_mask is not None else torch.ones_like(x)), self.lrelu_slope
        for c1, c2 in zip(self.convs1, self.convs2):
            x = cnv(F.leaky_relu(cnv(F.leaky_relu(x, s) * m, c1), s) * m, c2) + x
        return x * m

    def forward(self, x, x_mask=None):
        return self._fwd(x, x_mask, lambda t, c: c(t))

    def forward_cl(self, x, x_mask=None):
        return self._fwd(x, x_mask, conv2d_cl)

    def remove_weight_norm(self):
        for l in (*self.convs1, *self.convs2):
            remove_weight_norm(l)


class ResBlock2(nn.Module):  # RVC hifigan resblock 2（单 dilation 组）
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([_wconv(channels, kernel_size, d) for d in dilation])
        self.convs.apply(init_weights)
        self.lrelu_slope = LRELU_SLOPE

    def _fwd(self, x, x_mask, cnv):
        m = x_mask if x_mask is not None else torch.ones_like(x)
        for c in self.convs:
            x = cnv(F.leaky_relu(x, self.lrelu_slope) * m, c) + x
        return x * m

    def forward(self, x, x_mask=None):
        return self._fwd(x, x_mask, lambda t, c: c(t))

    def forward_cl(self, x, x_mask=None):
        return self._fwd(x, x_mask, conv2d_cl)

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class GeneratorNSF(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, gin_channels, sr, is_half=False):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = nn.Upsample(scale_factor=prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0, is_half=is_half)
        self.noise_convs = nn.ModuleList()  # 注册序先于 conv_pre:同 sovits,保 clip 归约序
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c = upsample_initial_channel // 2 ** (i + 1)
            self.ups.append(weight_norm(ConvTranspose1d(
                upsample_initial_channel // 2 ** i, c, k, u, padding=(k - u) // 2)))
            self.noise_convs.append(
                Conv1d(1, c, s * 2, s, s // 2)
                if (s := prod(upsample_rates[i + 1:])) and i + 1 < self.num_upsamples
                else Conv1d(1, c, 1))
        self.resblocks = nn.ModuleList(
            resblock(upsample_initial_channel // 2 ** (i + 1), k, d)
            for i in range(self.num_upsamples)
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes))
        self.conv_post = Conv1d(upsample_initial_channel // 2 ** self.num_upsamples,
                                1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp, self.lrelu_slope = prod(upsample_rates), LRELU_SLOPE

    def _synth(self, x, f0, g, n_res, cnv, tcnv, pck, rblk, uncl):
        # autocast 下统一输入 dtype：rvc x_mask(fp32) 污染 z 链 → conv_pre/cond 走 tf32
        # （比 fp16 慢 2×）；显式 cast 让首个 conv 走 fp16 tensor core
        if torch.is_autocast_enabled():
            ac = torch.get_autocast_gpu_dtype()
            x, f0 = x.to(ac), f0.to(ac)
            if g is not None:
                g = g.to(ac)
        har, _, _ = self.m_source(f0, self.upp)
        har = har.transpose(1, 2)
        if n_res is not None:
            n = int(n_res.item()) if isinstance(n_res, torch.Tensor) else int(n_res)
            if n * self.upp != har.shape[-1]:
                har = F.interpolate(har, size=n * self.upp, mode="linear")
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")
        x, hs = pck(x), pck(har)
        x = cnv(x, self.conv_pre)
        if g is not None:
            x = x + cnv(pck(g), self.cond)
        for i, (ups, nconv) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = tcnv(x, ups)
            s = cnv(hs, nconv)
            m = min(x.shape[-1], s.shape[-1])
            x = x[..., :m] + s[..., :m]
            x = sum(rblk(r, x) for r in
                    self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]) / self.num_kernels
        return uncl(torch.tanh(cnv(F.leaky_relu(x), self.conv_post)))

    def forward(self, x, f0, g=None, n_res=None):
        if getattr(self, "_use_cl", False):
            return self.forward_cl(x, f0, g, n_res)
        return self._synth(x, f0, g, n_res,
                           lambda t, c: c(t), lambda t, c: c(t),
                           lambda t: t, lambda r, t: r(t), lambda t: t)

    def forward_cl(self, x, f0, g=None, n_res=None):
        # pck: [B,C,T] -> [B,C,1,T] channels_last（view，无拷贝）；uncl 收尾变回 [B,1,T]
        return self._synth(x, f0, g, n_res,
                           conv2d_cl, convT2d_cl,
                           lambda t: t[:, :, None, :].to(memory_format=torch.channels_last),
                           lambda r, t: r.forward_cl(t), lambda t: t[:, :, 0, :])

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class TextEncoder(nn.Module):  # RVC prior: emb_phone + emb_pitch -> FFT -> m, logs, x_mask
    def __init__(self, in_channels, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, f0=True):
        super().__init__()
        self.out_channels, self.hidden_channels = out_channels, hidden_channels
        self.filter_channels, self.n_heads, self.n_layers = filter_channels, n_heads, n_layers
        self.kernel_size, self.p_dropout = kernel_size, float(p_dropout)
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch token 1-255
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads,
                               n_layers, kernel_size, float(p_dropout),
                               window_size=10)  # RVC default (so-vits uses 4); rel-pos emb shape
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths, skip_head=None, vol_emb=None):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        if vol_emb is not None:
            x = x + vol_emb  # [b,t,h] vol_embedding 模式（与 so-vits emb_vol 同槽位，pre-encoder）
        x = self.lrelu(x * sqrt(self.hidden_channels))  # [b,t,h]
        x = torch.transpose(x, 1, -1)  # [b,h,t]
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        if skip_head is not None:
            head = int(skip_head.item()) if isinstance(skip_head, torch.Tensor) else int(skip_head)
            x, x_mask = x[:, :, head:], x_mask[:, :, head:]
        m, logs = torch.split(self.proj(x) * x_mask, self.out_channels, dim=1)
        return m, logs, x_mask


class RVCModel(nn.Module):
    """RVC v2 generator (SynthesizerTrnMs768NSFsid), usvc 训练接口。默认值来自
    RVC configs/v2/48k.json（spec_channels=1025, segment_size=20, inter/hidden 192,
    filter 768, n_heads 2, n_layers 6, resblock 1, kernels [3,7,11], dil [[1,3,5]×3],
    up [12,10,2,2]/[24,20,4,4], gin 256, sr 48k）。"""
    def __init__(self, n_units=768, spec_channels=1025, segment_size=20,
                 inter_channels=192, hidden_channels=192, filter_channels=768,
                 n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, resblock="1",
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 upsample_rates=(12, 10, 2, 2), upsample_initial_channel=512,
                 upsample_kernel_sizes=(24, 20, 4, 4), spk_embed_dim=1,
                 gin_channels=256, sr=48000, is_half=False, vol_embedding=False, **kwargs):
        super().__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.spec_channels, self.inter_channels = spec_channels, inter_channels
        self.hidden_channels, self.filter_channels = hidden_channels, filter_channels
        self.n_heads, self.n_layers = n_heads, n_layers
        self.kernel_size, self.p_dropout = kernel_size, float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = list(resblock_kernel_sizes)
        self.resblock_dilation_sizes = [list(d) for d in resblock_dilation_sizes]
        self.upsample_rates = list(upsample_rates)
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = list(upsample_kernel_sizes)
        self.segment_size, self.gin_channels, self.spk_embed_dim = segment_size, gin_channels, spk_embed_dim
        self.enc_p = TextEncoder(n_units, inter_channels, hidden_channels, filter_channels,
                                 n_heads, n_layers, kernel_size, float(p_dropout))
        self.dec = GeneratorNSF(inter_channels, resblock, self.resblock_kernel_sizes,
                                self.resblock_dilation_sizes, self.upsample_rates,
                                upsample_initial_channel, self.upsample_kernel_sizes,
                                gin_channels=gin_channels, sr=sr, is_half=is_half)
        self.enc_q = EncoderQ(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                              gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3,
                                          gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        self.vol_embedding = vol_embedding
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "enc_q"):
            self.enc_q.remove_weight_norm()

    def forward(self, units, pitch_token, pitchf, spec, wave=None, spk_id=None,
                frame_len=None, vol=None) -> dict:
        # 训练 forward；返回 dict 供 loss（wave_gen/z_p/m_p/logs_p/m_q/logs_q/
        # z_mask/ids_slice/phone_lengths）。wave 未用（API parity）
        if spk_id is None:
            spk_id = torch.zeros(units.shape[0], dtype=torch.long, device=units.device)
        g = self.emb_g(spk_id.unsqueeze(-1) if spk_id.dim() == 1 else spk_id) \
            .transpose(1, 2).contiguous()  # [B,1] -> [B,1,gin] -> [B,gin,1]
        phone_lengths = frame_len if frame_len is not None else torch.full(
            (units.shape[0],), units.shape[1], device=units.device, dtype=torch.long)
        vol_emb = self.emb_vol(vol[:, :, None]) if (vol is not None and self.vol_embedding) else None
        m_p, logs_p, x_mask = self.enc_p(units, pitch_token, phone_lengths, vol_emb=vol_emb)
        z, m_q, logs_q, y_mask = self.enc_q(spec, phone_lengths, g=g)  # RVC: spec 对齐 units 长度
        z_p = self.flow(z, y_mask, g=g)
        z_slice, pitchf_slice, ids_slice = rand_slice_segments_with_pitch(
            z, pitchf, phone_lengths, self.segment_size)
        return {"wave_gen": self.dec(z_slice, pitchf_slice, g=g), "z_p": z_p,
                "m_p": m_p, "logs_p": logs_p, "m_q": m_q, "logs_q": logs_q,
                "z_mask": y_mask, "ids_slice": ids_slice, "phone_lengths": phone_lengths}

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, skip_head=None,
              return_length=None, return_length2=None, vol=None):
        # RVC SynthesizerTrnMs256NSFsid.infer verbatim（流式 skip_head 分支）
        g = self.emb_g(sid).unsqueeze(-1)
        vol_emb = self.emb_vol(vol[:, :, None]) if (vol is not None and self.vol_embedding) else None
        if skip_head is not None and return_length is not None:
            head = int(skip_head.item()) if isinstance(skip_head, torch.Tensor) else int(skip_head)
            length = int(return_length.item()) if isinstance(return_length, torch.Tensor) else int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head, vol_emb=vol_emb)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head:dec_head + length]
            x_mask = x_mask[:, :, dec_head:dec_head + length]
            nsff0 = nsff0[:, head:head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, vol_emb=vol_emb)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)
        return o, x_mask, (z, z_p, m_p, logs_p)

    @staticmethod
    def f0_to_coarse(f0, f0_min=50.0, f0_max=1100.0):
        # RVC f0 量化（infer/rtrvc.py get_f0_post verbatim）。与 so-vits 公式数学
        # 等价但非 bit 等价（边界 token 不同），须保持独立：tests/test_mlx_ops_dedup.py
        # 守卫两者发散。恒走 fp32：autocast fp16 log/clamp 可越界溢出 256 词表
        with torch.autocast("cuda", enabled=False):
            f0 = f0.float()
            f0_mel_min = 1127 * torch.log(torch.tensor(1 + f0_min / 700))
            f0_mel_max = 1127 * torch.log(torch.tensor(1 + f0_max / 700))
            f0_mel = 1127 * torch.log(1 + f0 / 700)
            f0_mel = torch.where(f0_mel > 0,
                                 (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, f0_mel)
            return torch.round(f0_mel.clamp(min=1, max=255)).long()


class MultiPeriodDiscriminatorV2(nn.Module):  # RVC v2 MPD: S + P(2,3,5,7,11,17,23,37)
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
               for i in (2, 3, 5, 7, 11, 17, 23, 37)])

    def forward(self, y, y_hat):
        rs, gs, fr, fg = [], [], [], []
        for d in self.discriminators:
            yr, fmr = d(y)
            yg, fmg = d(y_hat)
            rs.append(yr); gs.append(yg); fr.append(fmr); fg.append(fmg)
        return rs, gs, fr, fg