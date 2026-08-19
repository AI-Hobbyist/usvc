"""MLX GAN loss/slice helpers shared by sovits and rvc training modules (E3).

Verbatim move of the six helpers that were duplicated byte-for-byte in
usvc/train/mlx/modules/{rvc,sovits}.py (mirror of the torch side in
usvc/train/losses.py, where lsgan_gen_loss etc. live once).
"""
from __future__ import annotations

import mlx.core as mx

__all__ = [
    "slice_wave_real",
    "slice_mel",
    "lsgan_gen_loss",
    "lsgan_disc_loss",
    "feature_match_loss",
    "kl_loss",
]


def slice_wave_real(wave_real, ids_slice, segment, hop):
    """[B, T, 1] → slice segments。镜像 torch gather。"""
    B = wave_real.shape[0]
    starts = ids_slice * hop  # [B]
    idx = starts[:, None] + mx.arange(segment)[None, :]  # [B, seg]
    # wave_real [B, L, 1] → [B, seg]
    sliced = mx.take_along_axis(wave_real[..., 0], idx, axis=1)
    return sliced[..., None]  # [B, seg, 1]


def slice_mel(mel_full, ids_slice, seg_frames):
    """[B, n_mels, T] mel → [B, n_mels, seg]。镜像 torch gather。"""
    B, M, T = mel_full.shape
    idx = ids_slice[:, None] + mx.arange(seg_frames)[None, :]  # [B, seg]
    idx = mx.broadcast_to(idx[:, None, :], (B, M, seg_frames))  # [B, M, seg]
    return mx.take_along_axis(mel_full, idx, axis=2)


def lsgan_gen_loss(fake_logits):
    loss = mx.array(0.0)
    for df in fake_logits:
        loss = loss + ((df - 1.0) ** 2).mean()
    return loss


def lsgan_disc_loss(real_logits, fake_logits):
    loss = mx.array(0.0)
    for dr, df in zip(real_logits, fake_logits):
        loss = loss + ((dr - 1.0) ** 2).mean()
        loss = loss + (df ** 2).mean()
    return loss


def feature_match_loss(real_fmaps, fake_fmaps, weight=2.0):
    loss = mx.array(0.0)
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + mx.abs(f - mx.stop_gradient(r)).mean()
    return weight * loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """so-vits/RVC simplified KL(sum/mask)。z_p: [B, T, C] NLC。"""
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * mx.exp(-2 * logs_p)
    if z_mask is not None:
        kl = kl * z_mask
        return kl.sum() / (z_mask.sum() + 1e-8)
    return kl.mean()