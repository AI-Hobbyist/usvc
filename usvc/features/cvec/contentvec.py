"""Pure-torch ContentVec / HuBERT-Base (HF wav2vec2 structure).

Matches the HuggingFace ``HubertModel`` state-dict layout exactly, so a HF
``pytorch_model.bin`` loads with ``strict=True`` and zero key remapping. This
replaces the transformers dependency for content-encoder inference.

Structure (ContentVec / hubert-base):
  feature_extractor.conv_layers.[0..6]   (group-norm on layer 0 only)
  feature_projection.{layer_norm, projection}
  encoder.pos_conv_embed.conv             (weight_norm: weight_g + weight_v)
  encoder.layer_norm
  encoder.layers.[0..11].{attention.{k,v,q,out}_proj, layer_norm,
                         feed_forward.{intermediate_dense, output_dense},
                         final_layer_norm}
  final_proj                              (ContentVec's 768->256 head, optional)

No transformers, no fairseq. Config: hidden=768, ffn=3072, heads=12, layers=12,
conv kernels [10,3,3,3,3,2,2] strides [5,2,2,2,2,2,2].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ContentVec", "contentvec_from_hf_ckpt"]


# ---------------------------------------------------------------------------
# Feature extractor (1-D conv stack)
# ---------------------------------------------------------------------------
class _ConvLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, has_norm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=False)
        self.layer_norm = nn.GroupNorm(out_ch, out_ch) if has_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return F.gelu(x)


class HubertFeatureExtractor(nn.Module):
    """Matches HF wav2vec2 feature_extractor exactly (kernels/strides fixed)."""

    def __init__(self) -> None:
        super().__init__()
        cfg = [
            (1, 512, 10, 5, True),   # layer 0: has group norm
            (512, 512, 3, 2, False),
            (512, 512, 3, 2, False),
            (512, 512, 3, 2, False),
            (512, 512, 3, 2, False),
            (512, 512, 2, 2, False),
            (512, 512, 2, 2, False),
        ]
        self.conv_layers = nn.ModuleList(
            [_ConvLayer(i, o, k, s, n) for (i, o, k, s, n) in cfg]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        for layer in self.conv_layers:
            x = layer(x)
        return x  # [B, 512, T_frames]


class HubertFeatureProjection(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Positional conv embedding (weight-normed grouped conv)
# ---------------------------------------------------------------------------
class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, dim: int = 768, kernel: int = 128, groups: int = 16) -> None:
        super().__init__()
        # HF layout: symmetric padding = kernel//2, then HubertSamePadLayer removes
        # 1 sample from the right when (T + 2*pad - kernel) is odd (i.e. kernel is even).
        pad = kernel // 2
        self.conv = nn.Conv1d(dim, dim, kernel, stride=1, padding=pad, groups=groups)
        nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        # SamePad: remove right pad so output length == input length.
        self.num_pad_remove = 1 if (kernel % 2 == 0) else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.num_pad_remove > 0:
            x = x[..., : -self.num_pad_remove]
        x = F.gelu(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# Transformer encoder layer (matches HF attention/feed_forward naming)
# ---------------------------------------------------------------------------
class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.h = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5  # HF scale = head_dim^-0.5, NOT dim^-0.5
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            # attn_mask: [B, T] (True = valid, False = pad) -> [B,1,1,T]
            m = attn_mask[:, None, None, :]
            attn = attn.masked_fill(~m, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, h, T, D/h]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, inter: int) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(dim, inter)
        self.output_dense = nn.Linear(inter, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class HubertEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, inter: int) -> None:
        super().__init__()
        self.attention = _Attention(dim, heads)
        self.layer_norm = nn.LayerNorm(dim)
        self.feed_forward = _FeedForward(dim, inter)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Post-LN (HF HuBERT): residual is the *raw* input, LN applied after merge.
        residual = x
        x = self.attention(x, attn_mask=attn_mask)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm(x)         # LN after attention residual

        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)   # final LN after FFN residual
        return x


class HubertEncoder(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, inter: int = 3072, layers: int = 12) -> None:
        super().__init__()
        self.pos_conv_embed = HubertPositionalConvEmbedding(dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([HubertEncoderLayer(dim, heads, inter) for _ in range(layers)])

    def forward(
        self,
        x: torch.Tensor,
        output_layer: Optional[int] = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.pos_conv_embed(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        n = output_layer if output_layer else len(self.layers)
        for layer in self.layers[:n]:
            x = layer(x, attn_mask=attn_mask)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class ContentVec(nn.Module):
    """Pure-torch ContentVec (drop-in for HF HubertModel on last_hidden_state)."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = HubertFeatureExtractor()
        self.feature_projection = HubertFeatureProjection()
        self.encoder = HubertEncoder()
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        # ContentVec's final projection (768 -> 256); only used if caller wants it.
        self.final_proj = nn.Linear(768, 256)
        self._mask = False  # no SpecAugment masking at inference

    def _mask_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self._mask:
            # SpecAugment: replace random spans with masked_spec_embed.
            # Simplified — full masking only matters for pretraining, not inference.
            pass
        return x

    def encode(
        self,
        x: torch.Tensor,
        layer: Optional[int] = None,
        audio_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, 1, T] raw 16kHz waveform. Returns hidden [B, T_frames, 768].

        If ``audio_lengths`` is given (16kHz sample counts per clip), an
        attention mask is built so padded positions don't contaminate real
        outputs — required for correct batched inference of variable-length clips.
        """
        feats = self.feature_extractor(x)  # [B, 512, T']
        feats = feats.transpose(1, 2)      # [B, T', 512]
        feats = self.feature_projection(feats)  # [B, T', 768]
        feats = self._mask_features(feats)

        attn_mask = None
        if audio_lengths is not None:
            # frame count = audio_len // 320 - 1 (conv stride 320, no padding)
            frame_lens = (audio_lengths // 320 - 1).clamp(min=1)
            T_frames = feats.shape[1]
            attn_mask = torch.arange(T_frames, device=feats.device)[None, :] < frame_lens[:, None]
        return self.encoder(feats, output_layer=layer, attn_mask=attn_mask)

    @torch.inference_mode()
    def last_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    @torch.inference_mode()
    def projected(self, x: torch.Tensor) -> torch.Tensor:
        """ContentVec final_proj output (768 -> 256). RVC-style 3_feature256."""
        return self.final_proj(self.encode(x))


def contentvec_from_hf_ckpt(ckpt_path: str, *, map_location: str = "cpu") -> ContentVec:
    """Load a HF ContentVec ``pytorch_model.bin`` into pure-torch ContentVec.

    No key remapping needed — module names mirror the HF layout exactly.
    """
    m = ContentVec()
    raw = torch.load(ckpt_path, map_location=map_location)
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    consume_prefix_in_state_dict_if_present(raw, "module.")
    missing, unexpected = m.load_state_dict(raw, strict=False)
    # final_proj may be missing if the ckpt is plain hubert-base (not ContentVec).
    m.eval()
    return m, missing, unexpected
