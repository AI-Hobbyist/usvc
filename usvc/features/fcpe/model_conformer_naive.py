import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from local_attention import LocalAttention

# From https://github.com/CNChTu/Diffusion-SVC/ by CNChTu
# License: MIT


class ConformerNaiveEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, use_norm=False, conv_only=False, conv_dropout=0., atten_dropout=0.):
        super().__init__()
        self.num_layers, self.num_heads, self.dim_model, self.use_norm = num_layers, num_heads, dim_model, use_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.encoder_layers = nn.ModuleList([CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.encoder_layers: x = layer(x, mask)
        return x


class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads=8, use_norm=False, conv_only=False, conv_dropout=0., atten_dropout=0.):
        super().__init__()
        self.conformer = nn.Sequential(ConformerConvModule(dim_model), nn.Dropout(conv_dropout)) if conv_dropout > 0. else ConformerConvModule(dim_model)
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留
        self.attn = SelfAttention(dim=dim_model, heads=num_heads, causal=False, use_norm=use_norm, dropout=atten_dropout) if not conv_only else None

    def forward(self, x, mask=None):  # pre-attn LN (self-attn on norm(x)), post-conv residual
        if self.attn is not None:
            x = x + self.attn(self.norm(x), mask=mask)
        return x + self.conformer(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), nn.GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
            nn.SiLU(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout),
        )

    def forward(self, x): out = self.net(x); return out


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding, groups):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size=kernel_size, padding=padding, groups=groups)

    def forward(self, x): out = self.conv(x); return out


def calc_same_padding(kernel_size):  # convs in the paper: 31 -> (15, 15); offset pads right by 1 on even kernels
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x): return x.transpose(*self.dims)
    # ^ data flow: (B,T,C) -> (B,C,T) then back after convs


class SelfAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8, dim_head=64, local_heads=0, local_window_size=256, nb_features=None,
                 feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False,
                 dropout=0., no_projection=False, use_norm=False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, generalized_attention=generalized_attention,
                                            kernel_fn=kernel_fn, qr_uniform_q=qr_uniform_q, no_projection=no_projection, use_norm=use_norm)
        self.heads, self.global_heads = heads, heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout,
                                         look_forward=int(not causal), rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None
        self.to_q, self.to_k, self.to_v = [nn.Linear(dim, inner_dim) for _ in range(3)]
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(self, x, context=None, mask=None, context_mask=None, name=None, inference=False, **kwargs):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        q, k, v = [t.reshape(t.shape[0], t.shape[1], h, -1).transpose(1, 2) for t in (self.to_q(x), self.to_k(context), self.to_v(context))]  # b n (h d) -> b h n d
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
        attn_outs = []
        if not empty(q):
            if exists(context_mask):
                v.masked_fill_(~context_mask[:, None, :, None], 0.)
            if not cross_attend:
                attn_outs.append(self.fast_attention(q, k, v))
        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            attn_outs.append(self.local_attn(lq, lk, lv, input_mask=mask))
        out = torch.cat(attn_outs, dim=1)
        out = self.dropout(self.to_out(out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)))  # b h n d -> b n (h d)
        return out


class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False,
                 kernel_fn=nn.ReLU(), qr_uniform_q=False, no_projection=False, use_norm=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads, self.nb_features, self.ortho_scaling = dim_heads, nb_features, ortho_scaling
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=nb_features, nb_columns=dim_heads,
                                         scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        self.register_buffer('projection_matrix', self.create_projection())
        self.generalized_attention, self.kernel_fn, self.no_projection, self.causal, self.use_norm = generalized_attention, kernel_fn, no_projection, causal, use_norm
        if self.causal or self.generalized_attention:
            raise NotImplementedError('Causal and generalized attention not implemented yet')

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.projection_matrix.copy_(self.create_projection())

    def forward(self, q, k, v):
        if self.use_norm:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        if self.no_projection:  # plain softmax-kernel path (efficient attention paper)
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        elif self.generalized_attention:
            raise NotImplementedError('generalized attention not implemented yet')
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=q.device)
            q, k = create_kernel(q, is_query=True), create_kernel(k, is_query=False)
        return linear_attention(q, k, None) if v is None else linear_attention(q, k, v)


def linear_attention(q, k, v):
    if v is None:
        return torch.einsum('...ed,...nd->...ne', k, q)
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-8)
    context = torch.einsum('...nd,...ne->...de', k, v)
    return torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    ratio = projection_matrix.shape[0] ** -0.5
    projection = projection_matrix[None, None].expand(b, h, *projection_matrix.shape).type_as(data)  # 'j d -> b h j d'
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    diag_data = (((data ** 2).sum(dim=-1) / 2.0) * (data_normalizer ** 2)).unsqueeze(dim=-1)
    if is_query:  # eps OUTSIDE exp for query, INSIDE for key (original arithmetic, keep bitwise)
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data + eps))
    return data_dash.type_as(data)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = [orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device) for _ in range(nb_full_blocks)]
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        block_list.append(orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)[:remaining_rows])
    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return torch.diag(multiplier) @ final_matrix


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    if qr_uniform_q:  # @Parskatt: q uniform https://arxiv.org/pdf/math-ph/0609050.pdf
        q *= torch.diag(r, 0).sign()
    return q.t()


def default(val, d): return val if exists(val) else d
def exists(val): return val is not None
def empty(tensor): return tensor.numel() == 0