"""E2 anchors: shared gan_g_loss/gan_d_loss/slice helpers == the old inline
per-backend formulas (pre-merge rvc/sovits module.py bodies, incl. RVC's
explicit mel_clip_val/eps=2e-6/2e-7 and so-vits' lf0 branch).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from usvc.backends.sovits.model import mel_spectrogram, spec_to_mel
from usvc.train.dataset import SVCBatch
from usvc.train.losses import (
    feature_match_loss,
    gan_d_loss,
    gan_g_loss,
    kl_loss,
    lsgan_disc_loss,
    lsgan_gen_loss,
    slice_mel,
    slice_wave_real,
)


def _fake_batch(B=2, T=64, spec_ch=1025, mel_bins=128):
    torch.manual_seed(3)
    ids_slice = torch.tensor([2, 5])
    seg_frames = 8
    hop = 480
    wav = torch.randn(B, T * hop + 100)
    spec = torch.rand(B, spec_ch, T) * 3 - 1  # [B, C, T] (module convention)
    mel_full = spec_to_mel(spec, 2048, mel_bins, 48000, 0.0, None, clip_val=2e-6)
    batch = SVCBatch(
        units=torch.randn(B, T, 768), f0=torch.rand(B, T) * 300,
        mel=spec, wav=wav,
    )
    segment = (ids_slice.max().item() + seg_frames) * hop
    out = {
        "wave_gen": torch.randn(B, 1, segment),
        "ids_slice": ids_slice,
        "z_p": torch.randn(B, 192, T), "logs_q": torch.randn(B, 192, T) - 1,
        "m_p": torch.randn(B, 192, T), "logs_p": torch.randn(B, 192, T) - 1,
        "z_mask": (torch.rand(B, 1, T) > 0.3),
    }
    return batch, out, mel_full, seg_frames, ids_slice


class _StubDisc:
    """Fixed logits/fmaps; records what it was called with."""

    def __init__(self, B):
        torch.manual_seed(7)
        self.logits = [torch.randn(B, 512), torch.randn(B, 256)]
        self.fmaps = [[torch.randn(B, 16, 80) for _ in range(3)] for _ in range(4)]
        self.calls = []

    def __call__(self, real, fake):
        self.calls.append((real, fake))
        return self.logits, self.logits, self.fmaps, self.fmaps


def _ref_gan_g(disc, out, batch, mel_args, c_mel, c_kl, hop, *,
               mel_clip_val=None, mel_eps=None, use_lf0=False):
    """Reference replicating the old inline rvc/sovits generator_loss body."""
    wave_gen = out["wave_gen"]
    spec = batch.mel
    wave = batch.wav
    ids_slice = out["ids_slice"]
    segment = wave_gen.shape[-1]
    if wave.dim() == 2:
        wave_real = wave.unsqueeze(1)
    else:
        wave_real = wave
    starts = ids_slice * hop
    idx = starts.unsqueeze(1) + torch.arange(segment, device=wave_real.device)
    wave_real_sliced = wave_real[:, 0, :].gather(1, idx).unsqueeze(1)

    real_logits, fake_logits, real_fmaps, fake_fmaps = disc(
        wave_real_sliced.detach(), wave_gen)
    loss_gen = lsgan_gen_loss(fake_logits)
    loss_fm = feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)
    loss_kl = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])

    n_fft, n_mels, sr, _, win, fmin, fmax = mel_args
    seg_frames = segment // hop
    if mel_clip_val is not None:
        mel_full = spec_to_mel(spec, n_fft, n_mels, sr, fmin, fmax, clip_val=mel_clip_val)
    else:
        mel_full = spec_to_mel(spec, n_fft, n_mels, sr, fmin, fmax)
    idx_mel = ids_slice.unsqueeze(1) + torch.arange(seg_frames, device=mel_full.device)
    y_mel = mel_full.gather(
        2, idx_mel.unsqueeze(1).expand(mel_full.shape[0], mel_full.shape[1], seg_frames))
    if mel_eps is not None:
        y_hat_mel = mel_spectrogram(wave_gen.squeeze(1), n_fft, n_mels, sr, hop, win,
                                    fmin, fmax, clip_val=mel_clip_val, eps=mel_eps)
    else:
        y_hat_mel = mel_spectrogram(wave_gen.squeeze(1), n_fft, n_mels, sr, hop, win,
                                    fmin, fmax)
    loss_mel = F.l1_loss(y_mel, y_hat_mel)

    if use_lf0:
        pred_lf0 = out.get("pred_lf0")
        if isinstance(pred_lf0, torch.Tensor):
            loss_lf0 = F.mse_loss(pred_lf0, out["lf0"])
        else:
            loss_lf0 = torch.tensor(0.0, device=wave_gen.device)
    else:
        loss_lf0 = torch.tensor(0.0, device=wave_gen.device)
    total = loss_gen + loss_fm + c_mel * loss_mel + c_kl * loss_kl + loss_lf0
    sub = {"g/gan": loss_gen.item(), "g/fm": loss_fm.item(),
           "g/mel": loss_mel.item(), "g/kl": loss_kl.item()}
    if use_lf0:
        sub["g/lf0"] = loss_lf0.item()
    return total, sub


def _ref_gan_d(disc, out, batch, hop):
    wave_gen = out["wave_gen"].detach()
    wave = batch.wav
    ids_slice = out["ids_slice"]
    segment = wave_gen.shape[-1]
    if wave.dim() == 2:
        wave_real = wave.unsqueeze(1)
    else:
        wave_real = wave
    starts = ids_slice * hop
    idx = starts.unsqueeze(1) + torch.arange(segment, device=wave_real.device)
    wave_real_sliced = wave_real[:, 0, :].gather(1, idx).unsqueeze(1)
    real_logits, fake_logits, _, _ = disc(wave_real_sliced, wave_gen)
    loss_d = lsgan_disc_loss(real_logits, fake_logits)
    return loss_d, {"d/gan": loss_d.item()}


def test_slice_wave_real_matches_manual_gather():
    wave = torch.randn(2, 40000)
    ids = torch.tensor([10, 20])
    seg, hop = 2048, 480
    got = slice_wave_real(wave, ids, seg, hop)
    want = torch.stack([
        wave[i, ids[i] * hop: ids[i] * hop + seg] for i in range(2)
    ]).unsqueeze(1)
    assert torch.equal(got, want)
    got3 = slice_wave_real(wave.unsqueeze(1), ids, seg, hop)  # 3D input path
    assert torch.equal(got3, want)


def test_slice_mel_matches_manual_gather():
    mel = torch.randn(2, 128, 100)
    ids = torch.tensor([3, 9])
    got = slice_mel(mel, ids, 12)
    want = torch.stack([mel[i, :, ids[i]: ids[i] + 12] for i in range(2)])
    assert torch.equal(got, want)


def test_gan_g_loss_rvc_variant():
    """RVC: explicit clip 2e-6 / eps 2e-7, no lf0."""
    batch, out, _m, _s, _i = _fake_batch()
    disc = _StubDisc(2)
    mel_args = (2048, 128, 48000, 480, 2048, 0.0, None)
    got, sub = gan_g_loss(disc, out, batch, mel_args, 45.0, 1.0, 480,
                          mel_clip_val=2e-6, mel_eps=2e-7)
    ref, ref_sub = _ref_gan_g(
        disc, out, batch, mel_args, 45.0, 1.0, 480,
        mel_clip_val=2e-6, mel_eps=2e-7)
    assert torch.equal(got, ref)
    assert sub == ref_sub


def test_gan_g_loss_sovits_variant():
    """so-vits: default mel params, lf0 branch active."""
    batch, out, _m, _s, _i = _fake_batch(spec_ch=1025, mel_bins=80)
    out["pred_lf0"] = torch.randn(2, 8)
    out["lf0"] = torch.randn(2, 8)
    disc = _StubDisc(2)
    mel_args = (2048, 80, 44100, 512, 2048, 0.0, 22050.0)
    got, sub = gan_g_loss(disc, out, batch, mel_args, 45.0, 1.0, 512, use_lf0=True)
    ref, ref_sub = _ref_gan_g(
        disc, out, batch, mel_args, 45.0, 1.0, 512, use_lf0=True)
    assert torch.equal(got, ref)
    assert sub == ref_sub


def test_gan_g_loss_sovits_no_lf0_output():
    """so-vits without f0 decoder: pred_lf0 absent -> zero lf0, sub still lists it."""
    batch, out, _m, _s, _i = _fake_batch(spec_ch=1025, mel_bins=80)
    disc = _StubDisc(2)
    mel_args = (2048, 80, 44100, 512, 2048, 0.0, 22050.0)
    got, sub = gan_g_loss(disc, out, batch, mel_args, 45.0, 1.0, 512, use_lf0=True)
    ref, ref_sub = _ref_gan_g(
        disc, out, batch, mel_args, 45.0, 1.0, 512, use_lf0=True)
    assert torch.equal(got, ref)
    assert sub == ref_sub
    assert sub["g/lf0"] == 0.0


def test_gan_d_loss_matches_old():
    batch, out, _m, _s, _i = _fake_batch()
    disc = _StubDisc(2)
    got, sub = gan_d_loss(disc, out, batch, 480)
    ref, ref_sub = _ref_gan_d(disc, out, batch, 480)
    assert torch.equal(got, ref)
    assert sub == ref_sub