"""sovits MLX GAN 训练 module。

build(cfg) -> (generator, discriminator, {"g": g_fn, "d": d_fn})。
loss 组成(镜像 torch SoVITSModule):
  G: lsgan_gen + 2.0*feature_match + c_mel*mel_L1 + c_kl*kl + lf0(若启用)
  D: lsgan_disc(real→1, fake→0)
mel 用 usvc/nn/mlx/stft.py(与 torch mel_spectrogram_torch 数值对齐)。
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["build"]


def _slice_wave_real(wave_real, ids_slice, segment, hop):
    """[B, T, 1] → slice segments。镜像 torch gather。"""
    B = wave_real.shape[0]
    starts = ids_slice * hop  # [B]
    idx = starts[:, None] + mx.arange(segment)[None, :]  # [B, seg]
    # wave_real [B, L, 1] → [B, seg]
    sliced = mx.take_along_axis(wave_real[..., 0], idx, axis=1)
    return sliced[..., None]  # [B, seg, 1]


def _slice_mel(mel_full, ids_slice, seg_frames):
    """[B, n_mels, T] mel → [B, n_mels, seg]。镜像 torch gather。"""
    B, M, T = mel_full.shape
    idx = ids_slice[:, None] + mx.arange(seg_frames)[None, :]  # [B, seg]
    idx = mx.broadcast_to(idx[:, None, :], (B, M, seg_frames))  # [B, M, seg]
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
    """so-vits/RVC simplified KL(sum/mask)。z_p: [B, T, C] NLC。"""
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * mx.exp(-2 * logs_p)
    if z_mask is not None:
        kl = kl * z_mask
        return kl.sum() / (z_mask.sum() + 1e-8)
    return kl.mean()


def build(cfg):
    from ....nn.mlx.backends.sovits import SoVITSModel, MultiPeriodDiscriminator

    opts = dict(cfg.backend_opts or {})
    spec_channels = opts.get("spec_channels", 1025)
    inter_channels = opts.get("inter_channels", 192)
    hidden_channels = opts.get("hidden_channels", 192)

    generator = SoVITSModel(
        n_units=opts.get("n_units", 768),
        spec_channels=spec_channels,
        segment_size=opts.get("segment_size", 20),
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=opts.get("filter_channels", 768),
        n_heads=opts.get("n_heads", 2),
        n_layers=opts.get("n_layers_dec", opts.get("n_layers", 6)),
        n_speakers=opts.get("num_speaker", 1),
        gin_channels=opts.get("gin_channels", 768),
        upsample_initial_channel=opts.get("upsample_initial_channel", 512),
        vol_embedding=opts.get("vol_embedding", False),
        use_automatic_f0_prediction=opts.get("use_automatic_f0_prediction", True),
    )
    discriminator = MultiPeriodDiscriminator(
        use_spectral_norm=opts.get("use_spectral_norm", False),
    )

    c_mel = float(opts.get("c_mel", 45.0))
    c_kl = float(opts.get("c_kl", 1.0))

    # mel 参数(so-vits config_template data section)
    mel_args = (
        opts.get("filter_length", 2048),
        opts.get("n_mel_channels", 80),
        opts.get("sampling_rate", 44100),
        opts.get("hop_length", 512),
        opts.get("win_length", 2048),
        opts.get("mel_fmin", 0.0),
        opts.get("mel_fmax", 22050.0),
    )
    n_fft, n_mels, sr, hop, win, fmin, fmax = mel_args
    # 训练 segment(样本数)
    segment_samples = int(opts.get("segment_size", 20)) * hop

    def _generator_out(g_model, batch):
        """G fwd once → out dict(训练 forward)。"""
        units = batch.units
        f0 = batch.f0
        spec = batch.mel           # [B, T, spec_channels]
        wave = batch.wav           # [B, L]
        spk_id = batch.spk_id
        frame_len = batch.frame_len
        vol = batch.rms
        uv = batch.uv
        return g_model.forward_train(
            units, f0, spec, wave, spk_id,
            frame_len=frame_len, vol=vol, uv=uv,
        )

    def g_fn(g_model, d_model, batch):
        out = _generator_out(g_model, batch)
        wave_gen = out["wave_gen"]                      # [B, seg_samples, 1]
        spec = batch.mel                               # [B, T, spec_channels] NLC
        wave = batch.wav                               # [B, L]
        ids_slice = out["ids_slice"]                   # [B]
        segment = wave_gen.shape[1]

        if wave.ndim == 2:
            wave_real = wave[..., None]
        else:
            wave_real = wave
        # slice real wave
        wave_real_sliced = _slice_wave_real(wave_real, ids_slice, segment, hop)
        # D on real(detach) and fake
        real_logits, fake_logits, real_fmaps, fake_fmaps = d_model(
            wave_real_sliced, wave_gen
        )

        loss_gen = _lsgan_gen_loss(fake_logits)
        loss_fm = _feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)

        # mel loss:y_mel = slice(spec_to_mel(spec)), y_hat_mel = mel(wave_gen)
        from ....nn.mlx.stft import spec_to_mel, mel_spectrogram
        seg_frames = segment // hop
        mel_full = spec_to_mel(spec.transpose(0, 2, 1), n_fft, n_mels, sr, fmin, fmax)
        # spec_to_mel 期望 [B, F, T],spec NLC [B, T, F] → transpose
        y_mel = _slice_mel(mel_full, ids_slice, seg_frames)
        y_hat_mel = mel_spectrogram(
            wave_gen[..., 0], n_fft, n_mels, sr, hop, win, fmin, fmax
        )
        loss_mel = mx.abs(y_mel - y_hat_mel).mean()

        loss_kl = _kl_loss(
            out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"]
        )

        loss_lf0 = mx.array(0.0)
        pred_lf0 = out.get("pred_lf0")
        if isinstance(pred_lf0, mx.array) and pred_lf0 is not None:
            loss_lf0 = ((pred_lf0 - out["lf0"]) ** 2).mean()

        total = loss_gen + loss_fm + c_mel * loss_mel + c_kl * loss_kl + loss_lf0
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
