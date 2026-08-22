import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

from .model_conformer_naive import ConformerNaiveEncoder


class CFNaiveMelPE(nn.Module):
    """Conformer-based Mel Prediction Encoder (FCPE): mel -> f0 latent (B, T, out_dims)."""
    def __init__(self, input_channels, out_dims, hidden_dims=512, n_layers=6, n_heads=8, f0_max=1975.5, f0_min=32.70,
                 use_fa_norm=False, conv_only=False, conv_dropout=0., atten_dropout=0., use_harmonic_emb=False):
        super().__init__()
        self.input_channels, self.out_dims, self.hidden_dims, self.n_layers, self.n_heads = input_channels, out_dims, hidden_dims, n_layers, n_heads
        self.f0_max, self.f0_min, self.use_fa_norm = f0_max, f0_min, use_fa_norm
        self.residual_dropout, self.attention_dropout = 0.1, 0.1  # 废弃代码,仅做兼容性保留
        self.harmonic_emb = nn.Embedding(9, hidden_dims) if use_harmonic_emb else None
        self.input_stack = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dims, 3, 1, 1), nn.GroupNorm(4, hidden_dims), nn.LeakyReLU(),
            nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1),
        )
        self.net = ConformerNaiveEncoder(num_layers=n_layers, num_heads=n_heads, dim_model=hidden_dims,
                                         use_norm=use_fa_norm, conv_only=conv_only, conv_dropout=conv_dropout, atten_dropout=atten_dropout)
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(nn.Linear(hidden_dims, out_dims))
        # torch vs numpy cent table differs ~1e-4..1e-3 — torch acceptable, keep single source
        self.register_buffer("cent_table", torch.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0], out_dims).detach())
        self.register_buffer("gaussian_blurred_cent_mask", (1200. * torch.log2(torch.Tensor([f0_max / 10.])))[0].detach())

    def forward(self, x, _h_emb=None):
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        if self.harmonic_emb is not None:  # training-only harmonic shift index
            x = x + self.harmonic_emb(torch.LongTensor([0 if _h_emb is None else int(_h_emb)]).to(x.device))
        return torch.sigmoid(self.output_proj(self.norm(self.net(x))))

    @torch.no_grad()
    def latent2cents_decoder(self, y, threshold=0.05, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            rtn = rtn * torch.where(confident <= threshold, float("-INF"), torch.ones_like(confident))
        return rtn

    @torch.no_grad()
    def latent2cents_local_decoder(self, y, threshold=0.05, mask=True):  # local argmax over 9 bins around peak
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = (torch.arange(0, 9).to(max_index.device) + (max_index - 4)).clamp(0, self.out_dims - 1)
        c_l, y_l = torch.gather(ci, -1, local_argmax_index), torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(c_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)
        if mask:
            rtn = rtn * torch.where(confident <= threshold, float("-INF"), torch.ones_like(confident))
        return rtn

    @torch.no_grad()
    def gaussian_blurred_cent2latent(self, cents):  # [B,N,1] -> latent gaussian bump, out of range -> 0
        mask = (cents > 0.1) & (cents < self.gaussian_blurred_cent_mask)
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()

    @torch.no_grad()
    def infer(self, mel, decoder="local_argmax", threshold=0.05):  # "argmax" | "local_argmax"
        latent = self.forward(mel)
        if decoder == "argmax":
            cents = self.latent2cents_decoder(latent, threshold=threshold)
        elif decoder == "local_argmax":
            cents = self.latent2cents_local_decoder(latent, threshold=threshold)
        else:
            raise ValueError(f"  [x] Unknown decoder type {decoder}.")
        return self.cent_to_f0(cents)

    def train_and_loss(self, mel, gt_f0, loss_scale=10):
        if mel.shape[-2] != gt_f0.shape[-2]:
            _len = min(mel.shape[-2], gt_f0.shape[-2])
            mel, gt_f0 = mel[:, :_len, :], gt_f0[:, :_len, :]
        x_gt = self.gaussian_blurred_cent2latent(self.f0_to_cent(gt_f0))
        if self.harmonic_emb is not None:
            x, x_half, x_double = self.forward(mel, _h_emb=0), self.forward(mel, _h_emb=1), self.forward(mel, _h_emb=2)
            loss = F.binary_cross_entropy(x, x_gt)
            loss += (F.binary_cross_entropy(x_half, self.gaussian_blurred_cent2latent(self.f0_to_cent(gt_f0) / 2)) + F.binary_cross_entropy(x_double, self.gaussian_blurred_cent2latent(self.f0_to_cent(gt_f0) * 2))) / 2
            return loss * loss_scale
        return F.binary_cross_entropy(self.forward(mel), x_gt) * loss_scale

    @torch.no_grad()
    def cent_to_f0(self, cent): return 10. * 2 ** (cent / 1200.)

    @torch.no_grad()
    def f0_to_cent(self, f0): return 1200. * torch.log2(f0 / 10.)