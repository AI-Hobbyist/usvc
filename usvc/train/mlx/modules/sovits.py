"""sovits MLX GAN 训练 module。
build(cfg) -> (generator, discriminator, {"g": g_fn, "d": d_fn})。
loss(镜像 torch SoVITSModule):G = lsgan_gen + 2.0*feature_match +
c_mel*mel_L1 + c_kl*kl + lf0(若启用);D = lsgan_disc(real→1, fake→0)。
mel 用 usvc/nn/mlx/stft.py(与 torch mel_spectrogram_torch 数值对齐)。"""
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
    from ....nn.mlx.backends.sovits import SoVITSModel, MultiPeriodDiscriminator
    opts = dict(cfg.backend_opts or {})
    generator = SoVITSModel(
        n_units=opts.get("n_units", 768), spec_channels=opts.get("spec_channels", 1025), segment_size=opts.get("segment_size", 20),
        inter_channels=opts.get("inter_channels", 192), hidden_channels=opts.get("hidden_channels", 192), filter_channels=opts.get("filter_channels", 768),
        n_heads=opts.get("n_heads", 2), n_layers=opts.get("n_layers_dec", opts.get("n_layers", 6)), n_speakers=opts.get("num_speaker", 1),
        gin_channels=opts.get("gin_channels", 768), upsample_initial_channel=opts.get("upsample_initial_channel", 512), vol_embedding=opts.get("vol_embedding", False),
        use_automatic_f0_prediction=opts.get("use_automatic_f0_prediction", True),
    )
    discriminator = MultiPeriodDiscriminator(
        use_spectral_norm=opts.get("use_spectral_norm", False))
    c_mel = float(opts.get("c_mel", 45.0))
    c_kl = float(opts.get("c_kl", 1.0))
    # mel 参数(so-vits config_template data section)
    n_fft, n_mels, sr, hop, win, fmin, fmax = (
        opts.get("filter_length", 2048), opts.get("n_mel_channels", 80),
        opts.get("sampling_rate", 44100), opts.get("hop_length", 512),
        opts.get("win_length", 2048), opts.get("mel_fmin", 0.0),
        opts.get("mel_fmax", 22050.0))
    seg_samples = int(opts.get("segment_size", 20)) * hop  # 训练 segment(样本数)
    def _gen_out(g_model, batch):  # G fwd once → out dict(训练 forward)
        spec = batch.mel
        spec = spec.transpose(0, 2, 1) if spec.ndim == 3 and spec.shape[1] == 1025 else spec
        return g_model.forward_train(batch.units, batch.f0, spec, batch.wav,
                                     batch.spk_id, frame_len=batch.frame_len,
                                     vol=batch.rms, uv=batch.uv)
    def g_fn(g_model, d_model, batch):
        out = _gen_out(g_model, batch)
        wave_gen = out["wave_gen"]  # [B, seg_samples, 1]
        wave_real_sliced, segment = _d_slice(batch.wav, wave_gen, out["ids_slice"], hop)
        real_logits, fake_logits, real_fmaps, fake_fmaps = d_model(wave_real_sliced, wave_gen)
        loss_gen = _lsgan_gen_loss(fake_logits)
        loss_fm = _feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)
        from ....nn.mlx.stft import spec_to_mel, mel_spectrogram
        spec = batch.mel
        spec = spec.transpose(0, 2, 1) if spec.ndim == 3 and spec.shape[1] == 1025 else spec
        seg_frames = segment // hop
        y_mel = _slice_mel(spec_to_mel(spec.transpose(0, 2, 1), n_fft, n_mels, sr,
                                       fmin, fmax), out["ids_slice"], seg_frames)
        y_hat_mel = mel_spectrogram(wave_gen[..., 0], n_fft, n_mels, sr, hop, win,
                                    fmin, fmax)
        loss_mel = mx.abs(y_mel - y_hat_mel).mean()
        loss_kl = _kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"],
                           out["z_mask"])
        loss_lf0 = mx.array(0.0)
        pred_lf0 = out.get("pred_lf0")
        if isinstance(pred_lf0, mx.array) and pred_lf0 is not None:
            loss_lf0 = ((pred_lf0 - out["lf0"]) ** 2).mean()
        return loss_gen + loss_fm + c_mel * loss_mel + c_kl * loss_kl + loss_lf0
    def d_fn(d_model, g_model, batch):
        out = _gen_out(g_model, batch)
        wave_gen = mx.stop_gradient(out["wave_gen"])
        wave_real_sliced, _ = _d_slice(batch.wav, wave_gen, out["ids_slice"], hop)
        real_logits, fake_logits, _, _ = d_model(wave_real_sliced, wave_gen)
        return _lsgan_disc_loss(real_logits, fake_logits)
    return generator, discriminator, {"g": g_fn, "d": d_fn}
