"""MLX GAN loss/slice helpers shared by sovits+rvc training modules (E3;torch
侧单份在 usvc/train/losses.py)。z_p 等输入 NLC [B,T,C]。"""
from __future__ import annotations
import mlx.core as mx
__all__ = ["slice_wave_real", "slice_mel", "lsgan_gen_loss", "lsgan_disc_loss",
           "feature_match_loss", "kl_loss"]
def slice_wave_real(wave_real, ids_slice, segment, hop):  # [B,L,1]→[B,seg,1],镜像 torch gather
    idx = ids_slice * hop
    idx = idx[:, None] + mx.arange(segment)[None, :]  # [B, seg]
    return mx.take_along_axis(wave_real[..., 0], idx, axis=1)[..., None]
def slice_mel(mel_full, ids_slice, seg_frames):  # [B,M,T]→[B,M,seg]
    B, M, _ = mel_full.shape
    idx = ids_slice[:, None] + mx.arange(seg_frames)[None, :]
    return mx.take_along_axis(mel_full, mx.broadcast_to(idx[:, None, :], (B, M, seg_frames)), axis=2)
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
    for rl, fl in zip(real_fmaps, fake_fmaps):
        for r, f in zip(rl, fl):
            loss = loss + mx.abs(f - mx.stop_gradient(r)).mean()
    return weight * loss
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * mx.exp(-2 * logs_p)
    if z_mask is not None:
        kl = kl * z_mask
        return kl.sum() / (z_mask.sum() + 1e-8)
    return kl.mean()
