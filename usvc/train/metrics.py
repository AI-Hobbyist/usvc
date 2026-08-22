"""Shared SVC validation metrics on mel spectrograms [T,M] or [B,T,M] (RIFT/DDSP/Diffusion-SVC)."""
import torch
import torch.nn.functional as F


def mcd(pred, target):
    # MCD in dB: 6.1411 = 10·sqrt(2)/ln10; all mel bins (matches RIFT-SVC's mcd)
    if pred.dim() == 2:
        return (mcd(pred.unsqueeze(0), target.unsqueeze(0))).squeeze(0)
    return 6.1411 * torch.sqrt(((pred - target) ** 2).sum(dim=-1).mean(dim=-1)).mean()


def si_snr(pred, target):
    if pred.dim() == 2:
        return si_snr(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
    eps = 1e-8
    p = pred.reshape(pred.shape[0], -1); t = target.reshape(target.shape[0], -1)
    p = p - p.mean(dim=-1, keepdim=True); t = t - t.mean(dim=-1, keepdim=True)  # zero-mean
    s_target = (p * t).sum(-1, keepdim=True) / ((t ** 2).sum(-1, keepdim=True) + eps) * t  # optimal scale
    return 10 * torch.log10((s_target ** 2).sum(-1) / (((p - s_target) ** 2).sum(-1) + eps) + eps).mean()


def psnr(pred, target):
    if pred.dim() == 2:
        return psnr(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
    eps = 1e-8
    mse_val = ((pred - target) ** 2).mean(dim=(-2, -1))
    return 10 * torch.log10((target.amax(dim=(-2, -1)) + eps) ** 2 / (mse_val + eps)).mean()


def mse(pred, target):
    return F.mse_loss(pred, target)


def compute_all_metrics(pred, target):
    return {k: f(pred, target) for k, f in (("mcd", mcd), ("si_snr", si_snr), ("psnr", psnr), ("mse", mse))}