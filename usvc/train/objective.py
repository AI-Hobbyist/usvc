"""Shared training objectives: FlowMatching (rift_dit/ddsp6/reflow) and DDPM (diffusion/sovits)."""
import math

import torch
import torch.nn.functional as F

from ..nn.schedule import linear_beta_schedule, compute_alphas


def lens_to_mask(t, length=None):
    # [b] lengths -> [b, n] boolean mask (True = valid frame); RIFT rf.py ``loss[mask]`` convention
    if length is None:
        length = int(t.amax().item()) if t.numel() else 0
    return torch.arange(length, device=t.device) < t[..., None]


def sample_time_lognorm(size, device):
    # stratified-logistic t ~ sigmoid(N(0,1)) (RIFT rf.py:20-27); concentrates near 0.5 where flow is hardest
    q = torch.linspace(0, 1, size + 1, device=device)
    z = q[:-1] + torch.rand((size,), device=device) / size
    return torch.sigmoid(torch.erfinv(2 * z - 1) * math.sqrt(2))


def sample_time_uniform(size, device):
    return torch.rand((size,), device=device)


class FlowMatchingObjective:
    # rectified-flow velocity MSE: x1=norm(mel), x0=randn, xt=(1-t)*x0+t*x1, target=x1-x0 (RIFT rf.py == DDSP reflow.py)
    def __init__(self, normalizer, time_schedule="lognorm", loss_type="l2"):
        self.norm, self.time_schedule, self.loss_type = normalizer, time_schedule, loss_type

    def sample_time(self, size, device):
        return sample_time_lognorm(size, device) if self.time_schedule == "lognorm" else sample_time_uniform(size, device)

    def compute_loss(self, x1_mel, velocity_fn, cond=None, mask=None):
        x1 = self.norm.norm(x1_mel)
        x0 = torch.randn_like(x1)  # RNG order parity-critical: randn_like BEFORE time draw
        t = self.sample_time(x1.shape[0], x1.device)
        while t.ndim < x1.ndim:
            t = t.unsqueeze(-1)
        pred = velocity_fn((1 - t) * x0 + t * x1, t.squeeze(), cond)
        mse = F.mse_loss(pred, x1 - x0, reduction="none")
        if self.loss_type == "l2_lognorm":
            # w = 0.398942/t/(1-t)·exp(-½·log(t/(1-t))²); must use the SAME t that built xt and fed the
            # net, else E[w]=1 collapses the lognorm emphasis into plain L2 (DDSP-SVC-6.3 verbatim port)
            t1 = t.flatten()
            w = 0.398942 / t1 / (1 - t1) * torch.exp(-0.5 * torch.log(t1 / (1 - t1)) ** 2)
            while w.ndim < mse.ndim:
                w = w.unsqueeze(-1)
            loss = w * mse
        else:
            loss = mse
        if mask is not None and loss.ndim >= 2 and mask.shape[1] == loss.shape[1]:  # [B,T,M] frame mask (rift_dit)
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)
        return loss.mean()


class DDPMObjective:
    # DDPM eps-prediction MSE, linear betas 1e-4..0.02 over timesteps (Diffusion-SVC diffusion.py:194-210)
    def __init__(self, normalizer, timesteps=1000, max_beta=0.02, loss_type="l2"):
        self.norm, self.timesteps, self.loss_type = normalizer, timesteps, loss_type
        a = compute_alphas(linear_beta_schedule(timesteps, max_beta=max_beta))
        self.sqrt_alphas_cumprod = a["sqrt_alphas_cumprod"]
        self.sqrt_one_minus_alphas_cumprod = a["sqrt_one_minus_alphas_cumprod"]

    def register_buffers(self, module):
        # non-persistent so they move with .to(device) but never enter checkpoints
        module.register_buffer("sqrt_alphas_cumprod", self.sqrt_alphas_cumprod, persistent=False)
        module.register_buffer("sqrt_one_minus_alphas_cumprod", self.sqrt_one_minus_alphas_cumprod, persistent=False)

    def compute_loss(self, x0_mel, denoise_fn, cond=None, t_max=None, mask=None):
        x0 = self.norm.norm(x0_mel)
        # RNG order parity-critical: randint BEFORE randn_like (t_max=None -> timesteps; sovits shallow passes k_step)
        t = torch.randint(0, t_max if t_max is not None else self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        acp = self.sqrt_alphas_cumprod.to(x0.device)[t]
        omacp = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t]
        while acp.ndim < x0.ndim:
            acp, omacp = acp.unsqueeze(-1), omacp.unsqueeze(-1)
        pred = denoise_fn(acp * x0 + omacp * noise, t, cond)
        if mask is not None and mask.shape[1] == x0.shape[1]:
            # padding mel=0 -> norm(+0.714) is a loud non-silence target that pollutes gradients; average real frames only
            elem = F.mse_loss(pred, noise, reduction="none") * mask.unsqueeze(-1)
            return elem.sum() / (mask.sum() * elem.shape[-1] + 1e-8)
        return F.l1_loss(pred, noise) if self.loss_type == "l1" else F.mse_loss(pred, noise)