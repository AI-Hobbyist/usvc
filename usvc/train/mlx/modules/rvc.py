"""rvc MLX GAN 训练 module。
build(cfg) -> (generator, discriminator, {"g": g_fn, "d": d_fn})。
loss(镜像 torch RVCModule):G = lsgan_gen + 2.0*feature_match + c_mel*mel_L1
+ c_kl*kl(无 lf0);D = lsgan_disc。
mel 参数 RVC v2 48k:n_fft=2048/n_mels=128/sr=48000/hop=480/win=2048,
fmin=0/fmax=None,clip_val=2e-6,eps=2e-7。"""
from __future__ import annotations
import mlx.core as mx
from ..losses import (
    feature_match_loss,
    kl_loss,
    lsgan_disc_loss,
    lsgan_gen_loss,
    slice_mel,
    slice_wave_real,
)
__all__ = ["build"]
def _d_slice(wave, wave_gen, ids_slice, hop):  # real wave 与 gen 对齐切片
    segment = wave_gen.shape[1]
    wave_real = wave[..., None] if wave.ndim == 2 else wave
    return _slice_wave_real(wave_real, ids_slice, segment, hop), segment
def build(cfg):
    from ....backends._defaults import RVC_UPSAMPLE
    from ....nn.mlx.backends.rvc import RVCModel, MultiPeriodDiscriminatorV2
    from ....nn.mlx.backends.rvc import f0_to_coarse
    opts = dict(cfg.backend_opts or {})
    version = str(opts.get("rvc_version", "v2"))
    sr = int(opts.get("sr", 48000))
    if "upsample_rates" in opts and "upsample_kernel_sizes" in opts:  # 默认表共享
        up, k = opts["upsample_rates"], opts["upsample_kernel_sizes"]
    else:
        up, k = RVC_UPSAMPLE[(version, sr)]
    generator = RVCModel(
        inter_channels=opts.get("inter_channels", 192), hidden_channels=opts.get("hidden_channels", 192), filter_channels=opts.get("filter_channels", 768),
        n_units=opts.get("n_units", 256 if version == "v1" else 768), spk_embed_dim=opts.get("num_speaker", 1), segment_size=opts.get("segment_size", 20),
        sr=sr, upsample_rates=up, upsample_kernel_sizes=k,
        vol_embedding=opts.get("vol_embedding", False)
    )
    discriminator = MultiPeriodDiscriminatorV2(
        use_spectral_norm=opts.get("use_spectral_norm", False))
    c_mel = float(opts.get("c_mel", 45.0))
    c_kl = float(opts.get("c_kl", 1.0))
    n_fft, n_mels, sr_c, hop, win, fmin, fmax = (
        opts.get("filter_length", 2048), opts.get("n_mel_channels", 128), sr,
        opts.get("hop_length", 480), opts.get("win_length", 2048),
        opts.get("mel_fmin", 0.0), opts.get("mel_fmax", None))
    mel_clip_val = float(opts.get("mel_clip_val", 2e-6))
    mel_eps = float(opts.get("mel_eps", 2e-7))
    def _gen_out(g_model, batch):
        pitch_token = batch.pitch_token
        if pitch_token is None:
            pitch_token = f0_to_coarse(batch.f0)
        spec = batch.mel
        spec = spec.transpose(0, 2, 1) if spec.ndim == 3 and spec.shape[1] == 1025 else spec
        return g_model.forward_train(batch.units, pitch_token, batch.f0, spec,
                                     batch.wav, batch.spk_id,
                                     frame_len=batch.frame_len, vol=batch.rms)
    def g_fn(g_model, d_model, batch):
        out = _gen_out(g_model, batch)
        wave_gen = out["wave_gen"]
        wave_real_sliced, segment = _d_slice(batch.wav, wave_gen, out["ids_slice"], hop)
        real_logits, fake_logits, real_fmaps, fake_fmaps = d_model(wave_real_sliced, wave_gen)
        loss_gen = _lsgan_gen_loss(fake_logits)
        loss_fm = _feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)
        loss_kl = _kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"],
                           out["z_mask"])
        from ....nn.mlx.stft import spec_to_mel, mel_spectrogram
        spec = batch.mel
        spec = spec.transpose(0, 2, 1) if spec.ndim == 3 and spec.shape[1] == 1025 else spec
        seg_frames = segment // hop
        y_mel = _slice_mel(spec_to_mel(spec.transpose(0, 2, 1), n_fft, n_mels, sr_c,
                                       fmin, fmax, clip_val=mel_clip_val),
                           out["ids_slice"], seg_frames)
        y_hat_mel = mel_spectrogram(wave_gen[..., 0], n_fft, n_mels, sr_c, hop, win,
                                    fmin, fmax, clip_val=mel_clip_val, eps=mel_eps)
        return loss_gen + loss_fm + c_mel * mx.abs(y_mel - y_hat_mel).mean() + c_kl * loss_kl
    def d_fn(d_model, g_model, batch):
        out = _gen_out(g_model, batch)
        wave_gen = mx.stop_gradient(out["wave_gen"])
        wave_real_sliced, _ = _d_slice(batch.wav, wave_gen, out["ids_slice"], hop)
        real_logits, fake_logits, _, _ = d_model(wave_real_sliced, wave_gen)
        return _lsgan_disc_loss(real_logits, fake_logits)
    return generator, discriminator, {"g": g_fn, "d": d_fn}
