"""rvc MLX GAN 训练 module。

build(cfg) -> (generator, discriminator, {"g": g_fn, "d": d_fn})。
loss 组成(镜像 torch RVCModule):
  G: lsgan_gen + 2.0*feature_match + c_mel*mel_L1 + c_kl*kl(无 lf0)
  D: lsgan_disc
mel 参数 RVC v2 48k:n_fft=2048/n_mels=128/sr=48000/hop=480/win=2048,
fmin=0/fmax=None,clip_val=2e-6,eps=2e-7。
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["build"]


def _slice_wave_real(wave_real, ids_slice, segment, hop):
    B = wave_real.shape[0]
    starts = ids_slice * hop
    idx = starts[:, None] + mx.arange(segment)[None, :]
    sliced = mx.take_along_axis(wave_real[..., 0], idx, axis=1)
    return sliced[..., None]


def _slice_mel(mel_full, ids_slice, seg_frames):
    B, M, T = mel_full.shape
    idx = ids_slice[:, None] + mx.arange(seg_frames)[None, :]
    idx = mx.broadcast_to(idx[:, None, :], (B, M, seg_frames))
    return mx.take_along_axis(mel_full, idx, axis=2)


def _lsgan_gen_loss(fake_logits):
    loss = mx.array(0.0)
    for df in fake_logits:
        loss = loss + ((df - 1.0) ** 2).mean()
    return loss


def _lsgan_disc_loss(real_logits, fake_logits):
    loss = mx.array(0.0)
    for dr, df in zip(real_logits, fake_logits):
        loss = loss + ((dr - 1.0) ** 2).mean()
        loss = loss + (df ** 2).mean()
    return loss


def _feature_match_loss(real_fmaps, fake_fmaps, weight=2.0):
    loss = mx.array(0.0)
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + mx.abs(f - mx.stop_gradient(r)).mean()
    return weight * loss


def _kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * mx.exp(-2 * logs_p)
    if z_mask is not None:
        kl = kl * z_mask
        return kl.sum() / (z_mask.sum() + 1e-8)
    return kl.mean()


def build(cfg):
    from ....nn.mlx.backends.rvc import RVCModel, MultiPeriodDiscriminatorV2
    from ....nn.mlx.backends.rvc import f0_to_coarse

    opts = dict(cfg.backend_opts or {})
    version = str(opts.get("rvc_version", "v2"))
    sr = int(opts.get("sr", 48000))
    default_n_units = 256 if version == "v1" else 768

    # RVC upsample 配置(镜像 torch RVCModule._RVC_UPSAMPLE)
    _RVC_UPSAMPLE = {
        ("v1", 32000): ([10, 4, 2, 2, 2], [16, 16, 4, 4, 4]),
        ("v1", 40000): ([10, 10, 2, 2], [16, 16, 4, 4]),
        ("v1", 48000): ([10, 6, 2, 2, 2], [16, 16, 4, 4, 4]),
        ("v2", 32000): ([10, 8, 2, 2], [20, 16, 4, 4]),
        ("v2", 48000): ([12, 10, 2, 2], [24, 20, 4, 4]),
    }
    if "upsample_rates" in opts and "upsample_kernel_sizes" in opts:
        up, k = opts["upsample_rates"], opts["upsample_kernel_sizes"]
    else:
        up, k = _RVC_UPSAMPLE[(version, sr)]

    generator = RVCModel(
        inter_channels=opts.get("inter_channels", 192),
        hidden_channels=opts.get("hidden_channels", 192),
        filter_channels=opts.get("filter_channels", 768),
        n_units=opts.get("n_units", default_n_units),
        spk_embed_dim=opts.get("num_speaker", 1),
        segment_size=opts.get("segment_size", 20),
        sr=sr,
        upsample_rates=up,
        upsample_kernel_sizes=k,
        vol_embedding=opts.get("vol_embedding", False),
    )
    discriminator = MultiPeriodDiscriminatorV2(
        use_spectral_norm=opts.get("use_spectral_norm", False),
    )

    c_mel = float(opts.get("c_mel", 45.0))
    c_kl = float(opts.get("c_kl", 1.0))

    mel_args = (
        opts.get("filter_length", 2048),
        opts.get("n_mel_channels", 128),
        sr,
        opts.get("hop_length", 480),
        opts.get("win_length", 2048),
        opts.get("mel_fmin", 0.0),
        opts.get("mel_fmax", None),
    )
    n_fft, n_mels, sr_c, hop, win, fmin, fmax = mel_args
    mel_clip_val = float(opts.get("mel_clip_val", 2e-6))
    mel_eps = float(opts.get("mel_eps", 2e-7))

    def _generator_out(g_model, batch):
        units = batch.units
        pitch_token = batch.pitch_token
        if pitch_token is None:
            pitch_token = f0_to_coarse(batch.f0)
        pitchf = batch.f0
        spec = batch.mel
        wave = batch.wav
        spk_id = batch.spk_id
        frame_len = batch.frame_len
        vol = batch.rms
        return g_model.forward_train(
            units, pitch_token, pitchf, spec, wave, spk_id,
            frame_len=frame_len, vol=vol,
        )

    def g_fn(g_model, d_model, batch):
        out = _generator_out(g_model, batch)
        wave_gen = out["wave_gen"]
        spec = batch.mel
        wave = batch.wav
        ids_slice = out["ids_slice"]
        segment = wave_gen.shape[1]

        if wave.ndim == 2:
            wave_real = wave[..., None]
        else:
            wave_real = wave
        wave_real_sliced = _slice_wave_real(wave_real, ids_slice, segment, hop)
        real_logits, fake_logits, real_fmaps, fake_fmaps = d_model(
            wave_real_sliced, wave_gen
        )

        loss_gen = _lsgan_gen_loss(fake_logits)
        loss_fm = _feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)

        loss_kl = _kl_loss(
            out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"]
        )

        from ....nn.mlx.stft import spec_to_mel, mel_spectrogram
        seg_frames = segment // hop
        mel_full = spec_to_mel(
            spec.transpose(0, 2, 1), n_fft, n_mels, sr_c, fmin, fmax,
            clip_val=mel_clip_val,
        )
        y_mel = _slice_mel(mel_full, ids_slice, seg_frames)
        y_hat_mel = mel_spectrogram(
            wave_gen[..., 0], n_fft, n_mels, sr_c, hop, win, fmin, fmax,
            clip_val=mel_clip_val, eps=mel_eps,
        )
        loss_mel = mx.abs(y_mel - y_hat_mel).mean()

        total = loss_gen + loss_fm + c_mel * loss_mel + c_kl * loss_kl
        return total

    def d_fn(d_model, g_model, batch):
        out = _generator_out(g_model, batch)
        wave_gen = mx.stop_gradient(out["wave_gen"])
        wave = batch.wav
        ids_slice = out["ids_slice"]
        segment = wave_gen.shape[1]
        if wave.ndim == 2:
            wave_real = wave[..., None]
        else:
            wave_real = wave
        wave_real_sliced = _slice_wave_real(wave_real, ids_slice, segment, hop)
        real_logits, fake_logits, _, _ = d_model(wave_real_sliced, wave_gen)
        return _lsgan_disc_loss(real_logits, fake_logits)

    return generator, discriminator, {"g": g_fn, "d": d_fn}
