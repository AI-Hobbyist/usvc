"""Triton fused contentvec attention：q@k.T*scale -> softmax -> @v 三步合成一个
flash-attention 风格 kernel，不 materialize [B,h,T,T] 矩阵，消除 softmax 内存墙。
数值等价 PyTorch 路径（绝对误差主导），tests/test_triton_fusion.py 守等价。"""
import torch

from ._compat import _HAS_TRITON

if _HAS_TRITON:
    import triton
    import triton.language as tl
else:
    # 无 triton 环境(macOS): no-op jit stub，kernel 定义不报错；attention() 走 torch fallback
    class _TritonStub:
        @staticmethod
        def jit(fn=None, **kw):
            return fn if fn is not None else (lambda f: f)
    triton = _TritonStub()
    tl = None


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, Out,
                stride_qb, stride_qh, stride_qt, stride_qd,
                stride_kb, stride_kh, stride_kt, stride_kd,
                stride_vb, stride_vh, stride_vt, stride_vd,
                stride_ob, stride_oh, stride_ot, stride_od,
                B, H, T, D: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                IS_EVEN_M: tl.constexpr, IS_EVEN_N: tl.constexpr):
    """Flash-attention forward (no mask). grid: (cdiv(T,BLOCK_M), B*H)。"""
    off_b, off_h = (tl.program_id(1) // H), (tl.program_id(1) % H)
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    Out += off_b * stride_ob + off_h * stride_oh
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n, offs_d = tl.arange(0, BLOCK_N), tl.arange(0, BLOCK_D)
    q_ptrs = Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    k_ptrs = K + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf"); l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    q = tl.load(q_ptrs) if IS_EVEN_M else tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)

    for start_n in range(0, T, BLOCK_N):
        abs_n = start_n + offs_n  # 绝对 N 索引（offs_n 是块内偏移）
        k = tl.load(k_ptrs) if IS_EVEN_N else tl.load(k_ptrs, mask=abs_n[:, None] < T, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        # 越界 N 列置 -inf；越界 M 行用 -1e4 避免 NaN（这些行不会被 store）
        if not IS_EVEN_N:
            qk = tl.where(abs_n[None, :] < T, qk, float("-inf"))
        if not IS_EVEN_M:
            qk = tl.where(offs_m[:, None] < T, qk, -1e4)
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs) if IS_EVEN_N else tl.load(v_ptrs, mask=abs_n[:, None] < T, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
        k_ptrs += BLOCK_N * stride_kt; v_ptrs += BLOCK_N * stride_vt

    o_ptrs = Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    if IS_EVEN_M:
        tl.store(o_ptrs, (acc / l_i[:, None]).to(Out.dtype.element_ty))
    else:
        tl.store(o_ptrs, (acc / l_i[:, None]).to(Out.dtype.element_ty), mask=offs_m[:, None] < T)


def fused_attention(q, k, v, scale):
    """q,k,v: [B,H,T,D_head] fp32 -> [B,H,T,D_head]，无 mask。"""
    B, H, T, D = q.shape
    out = torch.empty_like(q)
    BLOCK_M = 256 if T >= 256 else 128 if T >= 128 else 64  # 实测 M=256/N=64 最优
    BLOCK_N, BLOCK_D = 64, triton.next_power_of_2(D)
    _fwd_kernel[(triton.cdiv(T, BLOCK_M), B * H)](
        q, k, v, scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, D, BLOCK_M, BLOCK_N, BLOCK_D,
        T % BLOCK_M == 0, T % BLOCK_N == 0,
        num_warps=8, num_stages=2,
    )
    return out


def fused_attention_masked(q, k, v, scale, attn_mask):
    """带 mask 版本（batched padding）：当前无 fused 实现，回退 PyTorch 保证数值正确。"""
    return ((q @ k.transpose(-2, -1)) * scale) \
        .masked_fill(~attn_mask[:, None, None, :], float("-inf")).softmax(dim=-1) @ v


def attention(q, k, v, scale, attn_mask=None):
    """统一入口：有 triton 且无 mask 用 fused；否则 PyTorch 原生。"""
    if _HAS_TRITON and attn_mask is None and q.is_cuda:
        return fused_attention(q, k, v, scale)
    if attn_mask is not None:
        return fused_attention_masked(q, k, v, scale, attn_mask)
    return (q @ k.transpose(-2, -1) * scale).softmax(dim=-1) @ v
