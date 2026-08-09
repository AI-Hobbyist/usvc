"""MLX 版 ContentVec / HuBERT-Base。

镜像 ``usvc/features/cvec/contentvec.py`` 的纯 torch 实现,逐类对应。
HF wav2vec2 state-dict 布局完全一致(键名 mirror),权重经
``_convert.load_torch_ckpt_to_mlx`` 加载(weight_norm pos_conv 在加载时合并)。

数值对齐目标:与 torch 版 ``ContentVec.encode`` 的 max abs diff < 1e-4。

与 torch 版的差异(均已在映射手册记录):
  - GroupNorm: MLX ``GroupNorm(C, C, pytorch_compatible=True)``
  - GELU: MLX ``nn.GELU(approx='none')``(精确 erf,与 torch 默认一致)
  - weight_norm(pos_conv, dim=2):加载时合并成普通 weight,运行时零开销
  - attention: 用 ``mx.fast.scaled_dot_product_attention(scale=head_dim**-0.5)``
    替代 torch 的 triton fused / native 双路径
  - masked_fill: ``mx.where(mask, x, -inf)``
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

__all__ = ["ContentVec", "contentvec_from_hf_ckpt"]

# ContentVec 配置(HF hubert-base,固定常量)
_DIM = 768
_INTER = 3072
_HEADS = 12
_LAYERS = 12
_HEAD_DIM = _DIM // _HEADS  # 64
# Conv feature extractor kernels/strides (固定,逐层)
_CONV_CFG = [
    (1, 512, 10, 5, True),    # layer 0: has group norm
    (512, 512, 3, 2, False),
    (512, 512, 3, 2, False),
    (512, 512, 3, 2, False),
    (512, 512, 3, 2, False),
    (512, 512, 2, 2, False),
    (512, 512, 2, 2, False),
]


# ---------------------------------------------------------------------------
# Conv1d with explicit padding(MLX Conv1d padding 仅支持 int 零填充)
# ---------------------------------------------------------------------------

def _conv1d(in_ch, out_ch, kernel, stride=1, padding=0, groups=1, bias=True):
    """便捷构造:Conv1d + 显式 pad(MLX 无 padding='same' 等字符串)。

    MLX Conv1d 输入 NLC,权重 (O, K, C/g)。padding 参数仍用 MLX 原生
    (int 零填充)。reflect/等需在输入侧用 _ops.pad_reflect1d 预处理。
    """
    return nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding,
                     groups=groups, bias=bias)


class _ConvLayer(nn.Module):
    """对应 torch _ConvLayer(Conv1d bias=False + 可选 GroupNorm + GELU)。

    torch: Conv1d 无 padding(T' = (T-k)//s+1)。MLX Conv1d padding=0 等价。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, has_norm: bool):
        super().__init__()
        # bias=False(与 torch 一致)
        self.conv = _conv1d(in_ch, out_ch, kernel, stride=stride, padding=0, bias=False)
        # GroupNorm(out_ch, out_ch) 当 groups=channels,等价 InstanceNorm
        # MLX GroupNorm 必须设 pytorch_compatible=True 才与 torch 数值一致
        self.layer_norm = nn.GroupNorm(out_ch, out_ch, pytorch_compatible=True) if has_norm else None

    def __call__(self, x):
        # x: [B, T, C] (MLX channels-last)
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return nn.gelu(x)  # mlx 默认 erf 精确版,等价 torch F.gelu 默认


class HubertFeatureExtractor(nn.Module):
    """7 层 _ConvLayer,总 stride = 5·2·2·2·2·2·2 = 320。"""

    def __init__(self):
        super().__init__()
        self.conv_layers = [_ConvLayer(i, o, k, s, n) for (i, o, k, s, n) in _CONV_CFG]

    def __call__(self, x):
        # x: [B, T, 1] -> [B, T', 512]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class HubertFeatureProjection(nn.Module):
    """LayerNorm(512) + Linear(512→768) + Dropout(0.1,推理恒等)。"""

    def __init__(self, dim: int = _DIM):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, dim)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x):
        # x: [B, T, 512] -> [B, T, 768]
        x = self.layer_norm(x)
        x = self.projection(x)
        # 推理(eval)下 Dropout 恒等;训练时按 p 缩放
        return self.dropout(x)


class HubertPositionalConvEmbedding(nn.Module):
    """位置卷积嵌入。torch 用 Conv1d(768,768,128,pad=64,groups=16) + weight_norm(dim=2)。

    MLX 适配:
      - weight_norm 在 ckpt 加载时合并(weight_g/weight_v → weight),运行时普通 conv
      - Conv1d groups=16,padding=64(对称零填充,偶数 kernel)
      - num_pad_remove=1(kernel 偶数):右侧切 1 样本,保证输出长度 == 输入
      - GELU
    """

    def __init__(self, dim: int = _DIM, kernel: int = 128, groups: int = 16):
        super().__init__()
        # padding=64 = kernel//2(对称)
        self.conv = _conv1d(dim, dim, kernel, stride=1, padding=kernel // 2,
                            groups=groups, bias=True)
        self.num_pad_remove = 1 if (kernel % 2 == 0) else 0

    def __call__(self, x):
        # x: [B, T, 768] -> conv 需要 [B, T, C](MLX NLC,已就位)
        # torch: transpose(1,2) -> [B,C,T];MLX 直接 NLC
        x = self.conv(x)
        if self.num_pad_remove > 0:
            # 切右侧 1 样本(MLX channels-last,时间维是 axis=1)
            x = x[:, :-1, :]
        return nn.gelu(x)  # erf 精确版


class _Attention(nn.Module):
    """多头自注意力。scale = head_dim ** -0.5 = 64^-0.5 ≈ 0.125。

    用 mx.fast.scaled_dot_product_attention(Metal flash-attention)替代 torch
    的 triton fused / native 双路径。无 mask 时直接 SDPA;有 mask 时构造
    additive mask 传入。
    """

    def __init__(self, dim: int = _DIM, heads: int = _HEADS):
        super().__init__()
        self.h = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5  # 0.125,与 torch 一致(head_dim 非 dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x, attn_mask=None):
        # x: [B, T, 768]
        B, T, D = x.shape
        h, hd = self.h, self.head_dim
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # [B, T, D] -> [B, T, h, hd] -> [B, h, T, hd]
        q = q.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, h, hd).transpose(0, 2, 1, 3)

        # 构造 SDPA mask:attn_mask 是 [B, T] bool(True=有效)
        sdpa_mask = None
        if attn_mask is not None:
            # mx.fast.scaled_dot_product_attention mask 支持:
            #   'causal' 字符串,或 bool/float array。
            # bool mask:True 参与计算;float:additive。
            # 我们的 attn_mask [B, T] 需扩展到 [B, 1, 1, T](query 全对 key 的有效位)
            sdpa_mask = attn_mask[:, None, None, :]  # [B,1,1,T]

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=sdpa_mask
        )
        # [B, h, T, hd] -> [B, T, D]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)


class _FeedForward(nn.Module):
    """Linear(768→3072) + GELU + Linear(3072→768)。"""

    def __init__(self, dim: int = _DIM, inter: int = _INTER):
        super().__init__()
        self.intermediate_dense = nn.Linear(dim, inter)
        self.output_dense = nn.Linear(inter, dim)

    def __call__(self, x):
        return self.output_dense(nn.gelu(self.intermediate_dense(x)))  # erf 精确版


class HubertEncoderLayer(nn.Module):
    """Post-LN 编码层(与 HF HuBERT 一致)。

    关键:residual 用**原始**输入(非 norm 后),LN 在 attention/FFN 之后。
    """

    def __init__(self, dim: int = _DIM, heads: int = _HEADS, inter: int = _INTER):
        super().__init__()
        self.attention = _Attention(dim, heads)
        self.layer_norm = nn.LayerNorm(dim)
        self.feed_forward = _FeedForward(dim, inter)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x, attn_mask=None):
        residual = x
        x = self.attention(x, attn_mask=attn_mask)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm(x)          # LN after attention residual
        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)    # final LN after FFN residual
        return x


class HubertEncoder(nn.Module):
    """位置卷积 + LN + dropout + N 层 EncoderLayer。"""

    def __init__(self, dim: int = _DIM, heads: int = _HEADS, inter: int = _INTER, layers: int = _LAYERS):
        super().__init__()
        self.pos_conv_embed = HubertPositionalConvEmbedding(dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.layers = [HubertEncoderLayer(dim, heads, inter) for _ in range(layers)]

    def __call__(self, x, output_layer: Optional[int] = None, attn_mask=None):
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
    """MLX 版 ContentVec(对应 torch ``ContentVec``)。

    用法::

        m = ContentVec()
        sd = load_torch_ckpt_to_mlx(ckpt, weight_norm_dims=CONTENTVEC_WEIGHT_NORM_DIMS)
        m.load_weights(list(sd.items()), strict=False)
        hidden = m.encode(mx_input)   # [B, T_frames, 768]
    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = HubertFeatureExtractor()
        self.feature_projection = HubertFeatureProjection()
        self.encoder = HubertEncoder()
        # masked_spec_embed 仅推理占位,不参与计算(保持 state_dict 兼容)
        self.masked_spec_embed = mx.zeros(_DIM)
        self.final_proj = nn.Linear(_DIM, 256)

    def encode(
        self,
        x: mx.array,
        layer: Optional[int] = None,
        audio_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """x: [B, 1, T] raw 16kHz wave(MLX NLC: [B, T, 1])。

        注意:torch 版输入 [B, 1, T](channel-first),MLX channels-last
        应为 [B, T, 1]。本方法接受两者(自动检测最后一维是否为 1)。
        返回 hidden [B, T_frames, 768]。
        """
        # 统一到 [B, T, 1](MLX NLC)
        if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] != 1:
            # torch 布局 [B, 1, T] -> [B, T, 1]
            x = x.transpose(0, 2, 1)

        feats = self.feature_extractor(x)          # [B, T', 512]
        feats = self.feature_projection(feats)     # [B, T', 768]

        attn_mask = None
        if audio_lengths is not None:
            # frame count = audio_len // 320 - 1(conv stride 320,无 padding)
            # 与 torch (audio_lengths // 320 - 1).clamp(min=1) 一致
            frame_lens = mx.maximum(audio_lengths // 320 - 1, 1)
            T_frames = feats.shape[1]
            # attn_mask[b, t] = t < frame_lens[b]
            attn_mask = mx.arange(T_frames)[None, :] < frame_lens[:, None]
        return self.encoder(feats, output_layer=layer, attn_mask=attn_mask)

    def projected(self, x: mx.array) -> mx.array:
        """ContentVec final_proj 输出(768 → 256)。RVC 3_feature256。"""
        return self.final_proj(self.encode(x))


def contentvec_from_hf_ckpt(ckpt_path: str) -> ContentVec:
    """加载 HF ContentVec pytorch_model.bin 到 MLX ContentVec。

    权重经 ``_convert.load_torch_ckpt_to_mlx`` 处理:
      - weight_norm(pos_conv dim=2)合并成普通 weight
      - Conv 权重转置 channels-last
      - 去 module. 前缀

    Returns:
        model(已 eval 的 ContentVec)
    """
    from ._convert import load_torch_ckpt_to_mlx, CONTENTVEC_WEIGHT_NORM_DIMS
    m = ContentVec()
    sd = load_torch_ckpt_to_mlx(
        ckpt_path,
        weight_norm_dims=CONTENTVEC_WEIGHT_NORM_DIMS,
    )
    # mlx.nn.Module.load_weights 接受 (key, array) 列表,返回 self(非 tuple)
    m.load_weights(list(sd.items()), strict=False)
    m.eval()
    return m
