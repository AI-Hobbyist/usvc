"""Pure-torch ContentVec / HuBERT-Base (HF wav2vec2 layout, zero key remap —
pytorch_model.bin loads strict): feature_extractor.conv_layers.[0..6].{conv,
layer_norm(0 only)}; feature_projection.{layer_norm,projection,dropout};
encoder.pos_conv_embed.conv (weight_norm dim=2) + encoder.{layer_norm,dropout,
layers.[0..11].{attention.{k,v,q,out}_proj, layer_norm, feed_forward.
{intermediate_dense,output_dense}, final_layer_norm}}; final_proj,
masked_spec_embed. conv cols [10,3,3,3,3,2,2] strides [5,2,2,2,2,2,2],
pos-conv k128/g16 same-pad (even kernel -> drop 1 right sample)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, has_norm):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=False)
        self.layer_norm = nn.GroupNorm(out_ch, out_ch) if has_norm else None

    def forward(self, x):
        return F.gelu(self.layer_norm(self.conv(x)) if self.layer_norm is not None else self.conv(x))

class HubertFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [(1, 512, 10, 5, True)] + [(512, 512, 3, 2, False)] * 4 + [(512, 512, 2, 2, False)] * 2
        self.conv_layers = nn.ModuleList([_ConvLayer(*c) for c in cfg])

    def forward(self, x):  # [B,1,T] -> [B,512,T']
        for layer in self.conv_layers: x = layer(x)
        return x

class HubertFeatureProjection(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.projection(self.layer_norm(x)))

class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, dim=768, kernel=128, groups=16):
        super().__init__()
        # HF layout: sym pad kernel//2, then drop 1 right sample when kernel even
        self.conv = nn.Conv1d(dim, dim, kernel, stride=1, padding=kernel // 2, groups=groups)
        nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.num_pad_remove = 1 if kernel % 2 == 0 else 0

    def forward(self, x):
        # slice BEFORE gelu: torch gelu differs ~1 ulp on non-contiguous views (SIMD path switch)
        y = self.conv(x.transpose(1, 2))
        if self.num_pad_remove: y = y[:, :, : -self.num_pad_remove]
        return F.gelu(y).transpose(1, 2)

class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dim, self.head_dim = heads, dim, dim // heads
        self.scale = self.head_dim ** -0.5  # HF scale = head_dim^-0.5, NOT dim^-0.5
        self.k_proj, self.v_proj, self.q_proj, self.out_proj = [nn.Linear(dim, dim) for _ in range(4)]

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        q, k, v = [p(x).view(B, T, self.h, -1).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj)]
        # Fused flash-attention (Triton): no [B,h,T,T] materialization (6GB/layer @12 layers),
        # numerically equiv to (q@k.T*scale).softmax()@v; fallback on any failure
        from ..._triton import fused_attention as _fa_mod
        try:
            out = _fa_mod.attention(q, k, v, self.scale, attn_mask=attn_mask)
        except Exception:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            out = attn.softmax(dim=-1) @ v
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))

class _FeedForward(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.intermediate_dense, self.output_dense = nn.Linear(dim, inter), nn.Linear(inter, dim)

    def forward(self, x): return self.output_dense(F.gelu(self.intermediate_dense(x)))

class HubertEncoderLayer(nn.Module):
    def __init__(self, dim, heads, inter):
        super().__init__()
        self.attention = _Attention(dim, heads)
        self.layer_norm = nn.LayerNorm(dim)
        self.feed_forward = _FeedForward(dim, inter)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attn_mask=None):  # post-LN (HF): residual = raw input, LN after merge
        x = self.layer_norm(x + self.dropout(self.attention(x, attn_mask=attn_mask)))
        return self.final_layer_norm(x + self.feed_forward(x))

class HubertEncoder(nn.Module):
    def __init__(self, dim=768, heads=12, inter=3072, layers=12):
        super().__init__()
        self.pos_conv_embed = HubertPositionalConvEmbedding(dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([HubertEncoderLayer(dim, heads, inter) for _ in range(layers)])

    def forward(self, x, output_layer: Optional[int] = None, attn_mask=None):
        x = self.dropout(self.layer_norm(x + self.pos_conv_embed(x)))
        for layer in self.layers[:output_layer if output_layer else len(self.layers)]:
            x = layer(x, attn_mask=attn_mask)
        return x

class ContentVec(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = HubertFeatureExtractor()
        self.feature_projection = HubertFeatureProjection()
        self.encoder = HubertEncoder()
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.final_proj = nn.Linear(768, 256)  # ContentVec head 768->256 (RVC 3_feature256)
        self._mask = False  # no SpecAugment masking at inference

    def encode(self, x, layer: Optional[int] = None, audio_lengths=None):
        """x: [B,1,T] 16k wav -> [B, T', 768]; audio_lengths (16k samples)
        builds an attention mask over padded frames for batched variable-length."""
        feats = self.feature_projection(self.feature_extractor(x).transpose(1, 2))
        attn_mask = None
        if audio_lengths is not None:
            frame_lens = (audio_lengths // 320 - 1).clamp(min=1)  # conv stride 320, no padding
            attn_mask = torch.arange(feats.shape[1], device=feats.device)[None, :] < frame_lens[:, None]
        return self.encoder(feats, output_layer=layer, attn_mask=attn_mask)

    @torch.inference_mode()
    def last_hidden_state(self, x): return self.encode(x)

    @torch.inference_mode()
    def projected(self, x): return self.final_proj(self.encode(x))

    def _mask_features(self, x):  # SpecAugment only matters at pretraining time
        return x

def contentvec_from_hf_ckpt(ckpt_path, *, map_location="cpu") -> "ContentVec":
    """Load HF pytorch_model.bin; no key remap (module names mirror HF layout)."""
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    m = ContentVec()
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    consume_prefix_in_state_dict_if_present(raw, "module.")
    missing, unexpected = m.load_state_dict(raw, strict=False)
    m.eval()
    return m, missing, unexpected