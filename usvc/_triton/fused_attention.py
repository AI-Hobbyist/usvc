"""Triton fused kernels for contentvec attention.

(A) fused_attention: 替换 contentvec _Attention.forward 的
    q@k.T * scale -> softmax -> @v 三步为一个 flash-attention 风格的 fused kernel，
    不再 materialize [B,h,T,T] attention 矩阵（6GB/layer），消除 softmax 的内存墙。

数值等价于 PyTorch 原生路径（q@k.T*scale, optional masked_fill(-inf), softmax, @v），
用 tests/test_triton_fusion.py 守 bit 级等价（rtol=1e-5 atol=1e-6）。
"""
from __future__ import annotations

import torch

from ._compat import _HAS_TRITON

if _HAS_TRITON:
    import triton
    import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, H, T, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_EVEN_M: tl.constexpr, IS_EVEN_N: tl.constexpr,
):
    """Flash-attention forward (no mask). grid: (tl.cdiv(T, BLOCK_M), B*H)."""
    start_m = tl.program_id(0)
    off = tl.program_id(1)
    off_b = off // H
    off_h = off % H

    # offsets
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    Out += off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    k_ptrs = K + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # scale once
    qk_scale = sm_scale

    # load Q (keep for all N loops)
    if IS_EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)

    end_n = T
    for start_n in range(0, end_n, BLOCK_N):
        # load K, V blocks
        if IS_EVEN_N:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < T, other=0.0)
        # compute QK
        qk = tl.dot(q, tl.trans(k))   # [BLOCK_M, BLOCK_N]
        qk = qk * qk_scale

        # online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        # update
        l_new = alpha * l_i + l_ij
        acc = acc * alpha[:, None]
        # load V, acc += p @ V
        if IS_EVEN_N:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0)
        acc = acc + tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new
        # advance K, V pointers
        k_ptrs += BLOCK_N * stride_kt
        v_ptrs += BLOCK_N * stride_vt

    # finalize: acc / l_i
    acc = acc / l_i[:, None]

    # store Out
    o_ptrs = Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    if IS_EVEN_M:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty))
    else:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < T)


def fused_attention(q, k, v, scale):
    """q,k,v: [B, H, T, D_head] float32. 返回 [B, H, T, D_head].

    数值等价于: (q @ k.transpose(-2,-1) * scale).softmax(-1) @ v
    无 mask（contentvec 无 padding 的单 clip 路径）。
    """
    B, H, T, D = q.shape
    out = torch.empty_like(q)

    # block sizes (criterion d 实测最优：M=256 N=64 比 M=128 快 ~10%%，改参无效)
    BLOCK_M = 256 if T >= 256 else 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    is_even_m = (T % BLOCK_M) == 0
    is_even_n = (T % BLOCK_N) == 0

    grid = (triton.cdiv(T, BLOCK_M), B * H)
    _fwd_kernel[grid](
        q, k, v, scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, D,
        BLOCK_M, BLOCK_N, BLOCK_D,
        is_even_m, is_even_n,
        num_warps=8, num_stages=2,
    )
    return out


def fused_attention_masked(q, k, v, scale, attn_mask):
    """带 mask 的版本（batched padding 场景）。当前 contentvec 单 clip 无 mask，
    走 fused_attention。如需 mask，回退到 PyTorch 原生（保证数值正确）。"""
    attn = (q @ k.transpose(-2, -1)) * scale
    m = attn_mask[:, None, None, :]
    attn = attn.masked_fill(~m, float("-inf"))
    return attn.softmax(dim=-1) @ v


def attention(q, k, v, scale, attn_mask=None):
    """统一入口：有 triton 且无 mask 用 fused；否则 PyTorch 原生。"""
    if _HAS_TRITON and attn_mask is None and q.is_cuda:
        return fused_attention(q, k, v, scale)
    if attn_mask is not None:
        return fused_attention_masked(q, k, v, scale, attn_mask)
    return (q @ k.transpose(-2, -1) * scale).softmax(dim=-1) @ v
