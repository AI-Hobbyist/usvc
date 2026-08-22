"""GAN/VITS loss primitives shared by so-vits and RVC backends."""
import torch
import torch.nn.functional as F

from ..backends.sovits.model import mel_spectrogram, spec_to_mel


def lsgan_disc_loss(real_logits, fake_logits):
    # LSGAN: D wants Dr->1, Df->0; summed over discriminators (so-vits/RVC, NOT /N)
    loss = 0.0
    for dr, df in zip(real_logits, fake_logits):
        loss = loss + F.mse_loss(dr, torch.ones_like(dr)) + F.mse_loss(df, torch.zeros_like(df))
    return loss


def lsgan_gen_loss(fake_logits):
    loss = 0.0
    for df in fake_logits:  # G wants Df->1; summed (matches so-vits/RVC)
        loss = loss + F.mse_loss(df, torch.ones_like(df))
    return loss


def feature_match_loss(real_fmaps, fake_fmaps, weight=2.0):
    # L1 between intermediate D features, sum over discriminators x layers (so-vits/RVC use weight 2.0)
    loss = 0.0
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + F.l1_loss(f, r.detach())
    return weight * loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask=None):
    # so-vits simplified KL: logs_p - logs_q - 0.5 + 0.5*(z_p-m_p)²*exp(-2·logs_p)
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2 * logs_p)
    if z_mask is not None:
        # normalize by ACTIVE FRAMES (z_mask.sum()), not elements — [B,1,T] mask broadcasts over C in the
        # numerator, so this is per-frame channel-summed KL (~1/C vs a per-element mean, breaks c_kl balance)
        return (kl * z_mask).sum() / (z_mask.sum() + 1e-8)
    return kl.mean()


def slice_wave_real(wave, ids_slice, segment, hop):
    # [B,L]|[B,1,L] -> [B,1,segment] at ids_slice*hop (vectorized, CUDA-graph safe)
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)
    idx = (ids_slice * hop).unsqueeze(1) + torch.arange(segment, device=wave.device)
    return wave[:, 0, :].gather(1, idx).unsqueeze(1)


def slice_mel(mel_full, ids_slice, seg_frames):
    # [B, n_mels, T] -> [B, n_mels, seg] at the same speech segment
    idx = ids_slice.unsqueeze(1) + torch.arange(seg_frames, device=mel_full.device)
    return mel_full.gather(2, idx.unsqueeze(1).expand(mel_full.shape[0], mel_full.shape[1], seg_frames))


def gan_g_loss(discriminator, out, batch, mel_args, c_mel, c_kl, hop, use_lf0=False, mel_clip_val=None, mel_eps=None):
    # VITS generator loss: lsgan + feature-match + mel-L1 + kl (+ optional lf0).
    # mel_clip_val/eps default to so-vits values; RVC passes 2e-6/2e-7 explicitly.
    wave_gen = out["wave_gen"]; segment = wave_gen.shape[-1]
    wave_real_sliced = slice_wave_real(batch.wav, out["ids_slice"], segment, hop)
    real_logits, fake_logits, real_fmaps, fake_fmaps = discriminator(wave_real_sliced.detach(), wave_gen)
    loss_gen = lsgan_gen_loss(fake_logits)
    loss_fm = feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)
    loss_kl = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])

    n_fft, n_mels, sr, _, win, fmin, fmax = mel_args
    spec_kw = {} if mel_clip_val is None else {"clip_val": mel_clip_val}
    mel_kw = spec_kw.copy()
    if mel_eps is not None: mel_kw["eps"] = mel_eps
    y_mel = slice_mel(spec_to_mel(batch.mel, n_fft, n_mels, sr, fmin, fmax, **spec_kw), out["ids_slice"], segment // hop)
    y_hat_mel = mel_spectrogram(wave_gen.squeeze(1), n_fft, n_mels, sr, hop, win, fmin, fmax, **mel_kw)
    loss_mel = F.l1_loss(y_mel, y_hat_mel)

    sub = {"g/gan": loss_gen.item(), "g/fm": loss_fm.item(), "g/mel": loss_mel.item(), "g/kl": loss_kl.item()}
    if use_lf0 and isinstance(out.get("pred_lf0"), torch.Tensor):
        loss_lf0 = F.mse_loss(out["pred_lf0"], out["lf0"]); sub["g/lf0"] = loss_lf0.item()
    else:
        loss_lf0 = torch.tensor(0.0, device=wave_gen.device)
        if use_lf0: sub["g/lf0"] = loss_lf0.item()
    return loss_gen + loss_fm + c_mel * loss_mel + c_kl * loss_kl + loss_lf0, sub


def gan_d_loss(discriminator, out, batch, hop):
    wave_gen = out["wave_gen"].detach(); segment = wave_gen.shape[-1]
    wave_real_sliced = slice_wave_real(batch.wav, out["ids_slice"], segment, hop)
    real_logits, fake_logits, _, _ = discriminator(wave_real_sliced, wave_gen)
    loss_d = lsgan_disc_loss(real_logits, fake_logits)
    return loss_d, {"d/gan": loss_d.item()}