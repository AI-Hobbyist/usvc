"""GAN and VITS loss primitives.

Shared by so-vits and RVC backends (both use VITS-style GAN training).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..backends.sovits.model import mel_spectrogram, spec_to_mel

__all__ = [
    "lsgan_disc_loss",
    "lsgan_gen_loss",
    "feature_match_loss",
    "kl_loss",
    "slice_wave_real",
    "slice_mel",
    "gan_g_loss",
    "gan_d_loss",
]


def lsgan_disc_loss(
    real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]
) -> torch.Tensor:
    """LSGAN discriminator loss.

    LSGAN uses least-squares: D wants Dr→1, Df→0.
        loss = mean((Dr - 1)^2) + mean(Df^2)

    Args:
        real_logits: list of [B, ...] discriminator outputs on real samples.
        fake_logits: list of [B, ...] discriminator outputs on fake samples.
    """
    loss = 0.0
    for dr, df in zip(real_logits, fake_logits):
        loss = loss + F.mse_loss(dr, torch.ones_like(dr))
        loss = loss + F.mse_loss(df, torch.zeros_like(df))
    return loss  # sum over discriminators (matches so-vits/RVC, NOT /N)


def lsgan_gen_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator loss. G wants Df→1.

        loss = mean((Df - 1)^2)
    """
    loss = 0.0
    for df in fake_logits:
        loss = loss + F.mse_loss(df, torch.ones_like(df))
    return loss  # sum over discriminators (matches so-vits/RVC, NOT /N)


def feature_match_loss(
    real_fmaps: list[list[torch.Tensor]],
    fake_fmaps: list[list[torch.Tensor]],
    weight: float = 2.0,
) -> torch.Tensor:
    """Feature matching loss (L1 between intermediate D features).

    Args:
        real_fmaps: list (per discriminator) of list (per layer) of [B, ...].
        fake_fmaps: same structure.
        weight: scalar multiplier (sovits/RVC use 2.0).
    """
    loss = 0.0
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + F.l1_loss(f, r.detach())
    return weight * loss


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL divergence loss for VITS posterior → prior.

    Uses the so-vits-svc / VITS simplified KL formula:
        kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p)^2 * exp(-2*logs_p)
        loss = sum(kl * mask) / sum(mask)   [normalized by frame count]

    This differs slightly from the full VITS KL in the treatment of the
    posterior variance term (so-vits drops the (logs_p - logs_q)^2 component
    and normalizes by mask sum instead of mean).

    Args:
        z_p: [B, C, T] flow-transformed posterior sample.
        logs_q: [B, C, T] posterior log-std.
        m_p: [B, C, T] prior mean.
        logs_p: [B, C, T] prior log-std.
        z_mask: [B, 1, T] optional mask (broadcast over C).
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2 * logs_p)
    if z_mask is not None:
        kl = kl * z_mask
        # so-vits-svc / RVC convention: sum over (B, C, T), normalize by the
        # number of active frames (z_mask.sum()). The [B,1,T] mask broadcasts
        # over the channel dim C in the numerator, so this is the per-frame
        # channel-summed KL — NOT a per-element mean (which would be ~1/C of
        # this and break the c_kl balance of the original projects).
        return kl.sum() / (z_mask.sum() + 1e-8)
    return kl.mean()


def slice_wave_real(
    wave: torch.Tensor, ids_slice: torch.Tensor, segment: int, hop: int
) -> torch.Tensor:
    """Real-wave slice at random speech segment (vectorized, CUDA-graph safe).

    wave [B, L] or [B, 1, L] → [B, 1, segment] starting at ids_slice*hop.
    Training guarantees ids_slice*hop + segment <= wave length.
    """
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)
    idx = (ids_slice * hop).unsqueeze(1) + torch.arange(segment, device=wave.device)
    return wave[:, 0, :].gather(1, idx).unsqueeze(1)


def slice_mel(
    mel_full: torch.Tensor, ids_slice: torch.Tensor, seg_frames: int
) -> torch.Tensor:
    """Mel frame slice at the same speech segment: [B, n_mels, T] → [B, n_mels, seg]."""
    idx = ids_slice.unsqueeze(1) + torch.arange(seg_frames, device=mel_full.device)
    return mel_full.gather(
        2, idx.unsqueeze(1).expand(mel_full.shape[0], mel_full.shape[1], seg_frames)
    )


def gan_g_loss(
    discriminator: torch.nn.Module,
    out: dict,
    batch: "SVCBatch",
    mel_args: tuple,
    c_mel: float,
    c_kl: float,
    hop: int,
    use_lf0: bool = False,
    mel_clip_val: float | None = None,
    mel_eps: float | None = None,
):
    """VITS generator loss: lsgan + feature-match + mel-L1 + kl (+ optional lf0).

    Shared by sovits and rvc (BackendModules keep their own forward_generator
    and the strict G/D alternating order lives in BaseGANModule.training_step).
    mel_clip_val/mel_eps default to the so-vits values (1e-5/1e-6); RVC passes
    2e-6/2e-7 explicitly.
    """
    wave_gen = out["wave_gen"]  # [B, 1, segment*hop]
    segment = wave_gen.shape[-1]
    wave_real_sliced = slice_wave_real(batch.wav, out["ids_slice"], segment, hop)

    real_logits, fake_logits, real_fmaps, fake_fmaps = discriminator(
        wave_real_sliced.detach(), wave_gen
    )
    loss_gen = lsgan_gen_loss(fake_logits)
    loss_fm = feature_match_loss(real_fmaps, fake_fmaps, weight=2.0)
    loss_kl = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])

    # Mel loss: y_mel = slice(spec_to_mel(spec)), y_hat_mel = mel(wave_gen)
    n_fft, n_mels, sr, _, win, fmin, fmax = mel_args
    seg_frames = segment // hop
    spec_kw = {"clip_val": mel_clip_val} if mel_clip_val is not None else {}
    mel_kw = {}
    if mel_clip_val is not None:
        mel_kw["clip_val"] = mel_clip_val
    if mel_eps is not None:
        mel_kw["eps"] = mel_eps
    y_mel = slice_mel(
        spec_to_mel(batch.mel, n_fft, n_mels, sr, fmin, fmax, **spec_kw),
        out["ids_slice"],
        seg_frames,
    )
    y_hat_mel = mel_spectrogram(
        wave_gen.squeeze(1), n_fft, n_mels, sr, hop, win, fmin, fmax, **mel_kw
    )
    loss_mel = F.l1_loss(y_mel, y_hat_mel)

    sub = {"g/gan": loss_gen.item(), "g/fm": loss_fm.item(),
           "g/mel": loss_mel.item(), "g/kl": loss_kl.item()}
    if use_lf0:
        pred_lf0 = out.get("pred_lf0")
        if isinstance(pred_lf0, torch.Tensor):
            loss_lf0 = F.mse_loss(pred_lf0, out["lf0"])
        else:
            loss_lf0 = torch.tensor(0.0, device=wave_gen.device)
        sub["g/lf0"] = loss_lf0.item()
    else:
        loss_lf0 = torch.tensor(0.0, device=wave_gen.device)

    return loss_gen + loss_fm + c_mel * loss_mel + c_kl * loss_kl + loss_lf0, sub


def gan_d_loss(
    discriminator: torch.nn.Module,
    out: dict,
    batch: "SVCBatch",
    hop: int,
):
    """VITS discriminator loss: LSGAN(real→1, fake→0) on the detached wave_gen."""
    wave_gen = out["wave_gen"].detach()
    segment = wave_gen.shape[-1]
    wave_real_sliced = slice_wave_real(batch.wav, out["ids_slice"], segment, hop)
    real_logits, fake_logits, _, _ = discriminator(wave_real_sliced, wave_gen)
    loss_d = lsgan_disc_loss(real_logits, fake_logits)
    return loss_d, {"d/gan": loss_d.item()}
