"""Triton fused residual_add + layer_norm。

contentvec encoder layer:
    x = residual + x        # residual add
    x = layer_norm(x)       # LayerNorm
合并成一个 kernel：读 residual + x 一次，算 mean/var + 归一化，写一次。
消除 layernorm 单独读 x 的第二次内存遍历（768-dim × 11184 × 12layers）。

数值等价于: LayerNorm(residual + x)。
"""
from __future__ import annotations

import torch

from ._compat import _HAS_TRITON

if _HAS_TRITON:
    import triton
    import triton.language as tl


@triton.jit
def _residual_layernorm_kernel(
    X, Res, Y, W, B,
    Mean, Rstd,
    stride_xb, stride_xr, stride_xd,
    N,
    eps,
    BLOCK_D: tl.constexpr,
):
    """fused: Y = LayerNorm(Res + X). grid: (B*T_rows,). one program per row."""
    row = tl.program_id(0)
    X += row * stride_xr
    Res += row * stride_xr
    Y += row * stride_xr
    W += 0  # weight is [D], broadcast
    B += 0

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < N

    x = tl.load(X + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)
    r = tl.load(Res + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)
    a = x + r

    mean = tl.sum(a, axis=0) / N
    var = tl.sum((a - mean) * (a - mean), axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    y = (a - mean) * rstd
    w = tl.load(W + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    b = tl.load(B + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    y = y * w + b

    tl.store(Y + offs_d * stride_xd, y.to(Y.dtype.element_ty), mask=mask_d)
    if Mean is not None:
        tl.store(Mean + row, mean)
    if Rstd is not None:
        tl.store(Rstd + row, rstd)


def fused_residual_layernorm(x, residual, weight, bias, eps=1e-5):
    """Y = LayerNorm(x + residual). x/residual: [..., D]. weight/bias: [D]."""
    orig_shape = x.shape
    D = orig_shape[-1]
    x2d = x.reshape(-1, D).contiguous()
    r2d = residual.reshape(-1, D).contiguous()
    M = x2d.shape[0]
    y = torch.empty_like(x2d)
    BLOCK_D = triton.next_power_of_2(D)
    _residual_layernorm_kernel[(M,)](
        x2d, r2d, y, weight, bias,
        None, None,  # no need to store mean/rstd for inference
        x2d.stride(0), x2d.stride(0), x2d.stride(1),
        D, eps, BLOCK_D,
        num_warps=4,
    )
    return y.reshape(orig_shape)


def residual_layernorm(x, residual, weight, bias, eps=1e-5):
    """统一入口。triton 不可用或 CPU 回退 F.layer_norm(x+residual)。"""
    if _HAS_TRITON and x.is_cuda:
        try:
            return fused_residual_layernorm(x, residual, weight, bias, eps)
        except Exception:
            pass
    return torch.nn.functional.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps)
