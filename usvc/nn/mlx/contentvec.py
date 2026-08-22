"""MLX 版 ContentVec/HuBERT-Base(镜像 features/cvec/contentvec.py;HF
wav2vec2 state-dict 键名完全一致,权重经 _convert 加载:pos_conv weight_norm
dim=2 合并、Conv 转 channels-last)。数值对齐:与 torch encode max diff < 1e-4。

torch 差异:GroupNorm 用 pytorch_compatible=True;GELU 用 erf 精确版
(mlx 默认=torch F.gelu 默认);attention 用 mx.fast.scaled_dot_product_attention
(scale=head_dim**-0.5);masked_fill 用 mx.where。
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

__all__ = ["ContentVec", "contentvec_from_hf_ckpt"]

# HF hubert-base 固定常量
_DIM, _INTER, _HEADS, _LAYERS = 768, 3072, 12, 12
_HEAD_DIM = _DIM // _HEADS  # 64
_CONV_CFG = [  # (in, out, kernel, stride, has_group_norm),总 stride=320
    (1, 512, 10, 5, True), (512, 512, 3, 2, False), (512, 512, 3, 2, False),
    (512, 512, 3, 2, False), (512, 512, 3, 2, False), (512, 512, 2, 2, False),
    (512, 512, 2, 2, False),
]


def _conv1d(in_ch, out_ch, kernel, stride=1, padding=0, groups=1, bias=True):
    """MLX Conv1d NLC,无 padding='same';reflect 需输入侧 _ops 预处理。"""
    return nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding,
                     groups=groups, bias=bias)


class _ConvLayer(nn.Module):
    """torch _ConvLayer:Conv1d(bias=False) + 可选 GroupNorm + GELU(erf)。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, has_norm: bool):
        super().__init__()
        self.conv = _conv1d(in_ch, out_ch, kernel, stride=stride, padding=0, bias=False)
        # GroupNorm(out,out) = InstanceNorm;pytorch_compatible 必须置真才对齐
        self.layer_norm = nn.GroupNorm(out_ch, out_ch, pytorch_compatible=True) if has_norm else None

    def __call__(self, x):
        x = self.conv(x)  # NLC
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return nn.gelu(x)


class HubertFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = [_ConvLayer(i, o, k, s, n) for (i, o, k, s, n) in _CONV_CFG]

    def __call__(self, x):  # [B,T,1] -> [B,T',512]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class HubertFeatureProjection(nn.Module):
    def __init__(self, dim: int = _DIM):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, dim)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x):  # [B,T,512] -> [B,T,768]
        return self.dropout(self.projection(self.layer_norm(x)))


class HubertPositionalConvEmbedding(nn.Module):
    """pos_conv:Conv1d(768,768,128,pad=64,groups=16) + weight_norm(dim=2,
    加载时合并)。偶数 kernel → 右侧切 1 样本保证输出长度==输入。"""

    def __init__(self, dim: int = _DIM, kernel: int = 128, groups: int = 16):
        super().__init__()
        self.conv = _conv1d(dim, dim, kernel, stride=1, padding=kernel // 2,
                            groups=groups, bias=True)
        self.num_pad_remove = 1 if (kernel % 2 == 0) else 0

    def __call__(self, x):
        x = self.conv(x)
        if self.num_pad_remove > 0:
            x = x[:, :-1, :]  # 切右侧 1 样本(channels-last)
        return nn.gelu(x)


class _Attention(nn.Module):
    """SDPA 替代 torch triton/native 双路径;scale=head_dim**-0.5=0.125。"""

    def __init__(self, dim: int = _DIM, heads: int = _HEADS):
        super().__init__()
        self.h, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x, attn_mask=None):
        B, T, D = x.shape
        h, hd = self.h, self.head_dim
        q, k, v = [i.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
                   for i in (self.q_proj(x), self.k_proj(x), self.v_proj(x))]
        sdpa_mask = attn_mask[:, None, None, :] if attn_mask is not None else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                   mask=sdpa_mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D))


class _FeedForward(nn.Module):
    def __init__(self, dim: int = _DIM, inter: int = _INTER):
        super().__init__()
        self.intermediate_dense = nn.Linear(dim, inter)
        self.output_dense = nn.Linear(inter, dim)

    def __call__(self, x):
        return self.output_dense(nn.gelu(self.intermediate_dense(x)))


class HubertEncoderLayer(nn.Module):
    """Post-LN(HF HuBERT 一致):residual 用原始输入,LN 在 attention/FFN 后。"""

    def __init__(self, dim: int = _DIM, heads: int = _HEADS, inter: int = _INTER):
        super().__init__()
        self.attention = _Attention(dim, heads)
        self.layer_norm = nn.LayerNorm(dim)
        self.feed_forward = _FeedForward(dim, inter)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x, attn_mask=None):
        residual = x
        x = self.layer_norm(residual + self.dropout(self.attention(x, attn_mask=attn_mask)))
        return self.final_layer_norm(x + self.feed_forward(x))


class HubertEncoder(nn.Module):
    def __init__(self, dim: int = _DIM, heads: int = _HEADS, inter: int = _INTER,
                 layers: int = _LAYERS):
        super().__init__()
        self.pos_conv_embed = HubertPositionalConvEmbedding(dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.layers = [HubertEncoderLayer(dim, heads, inter) for _ in range(layers)]

    def __call__(self, x, output_layer: Optional[int] = None, attn_mask=None):
        x = self.dropout(self.layer_norm(x + self.pos_conv_embed(x)))
        for layer in self.layers[:output_layer if output_layer else len(self.layers)]:
            x = layer(x, attn_mask=attn_mask)
        return x


class ContentVec(nn.Module):
    """用法:m=ContentVec();m.load_weights(list(load_torch_ckpt_to_mlx(
    ckpt, weight_norm_dims=CONTENTVEC_WEIGHT_NORM_DIMS).items()), strict=False)。"""

    def __init__(self):
        super().__init__()
        self.feature_extractor = HubertFeatureExtractor()
        self.feature_projection = HubertFeatureProjection()
        self.encoder = HubertEncoder()
        self.masked_spec_embed = mx.zeros(_DIM)  # 推理占位,state_dict 兼容
        self.final_proj = nn.Linear(_DIM, 256)

    def encode(self, x: mx.array, layer: Optional[int] = None,
               audio_lengths: Optional[mx.array] = None) -> mx.array:
        """raw 16k wave → hidden [B, T_frames, 768]。接受 [B,1,T] 或 [B,T,1](NLC)
        自动检测。audio_lengths → attn_mask:frame = (len//320-1).clamp(min=1)。"""
        if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] != 1:
            x = x.transpose(0, 2, 1)  # torch 布局 [B,1,T] -> [B,T,1]
        feats = self.feature_projection(self.feature_extractor(x))
        attn_mask = None
        if audio_lengths is not None:
            frame_lens = mx.maximum(audio_lengths // 320 - 1, 1)
            attn_mask = mx.arange(feats.shape[1])[None, :] < frame_lens[:, None]
        return self.encoder(feats, output_layer=layer, attn_mask=attn_mask)

    def projected(self, x: mx.array) -> mx.array:
            return self.final_proj(self.encode(x))


def contentvec_from_hf_ckpt(ckpt_path: str) -> ContentVec:
    """HF pytorch_model.bin → MLX ContentVec(weight_norm dim=2 合并 + 转置)。"""
    from ._convert import CONTENTVEC_WEIGHT_NORM_DIMS, load_torch_ckpt_to_mlx
    m = ContentVec()
    m.load_weights(list(load_torch_ckpt_to_mlx(
        ckpt_path, weight_norm_dims=CONTENTVEC_WEIGHT_NORM_DIMS).items()), strict=False)
    m.eval()
    return m