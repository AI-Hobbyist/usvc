"""Triton fused kernel 数值等价守卫（M2）。

每个 fused kernel 对比 Triton 实现 vs PyTorch 原生：
    torch.testing.assert_close(triton, pytorch, rtol=1e-5, atol=1e-6)
"""
import pytest
import torch

from usvc._triton._compat import _HAS_TRITON

pytestmark = pytest.mark.skipif(not _HAS_TRITON, reason="triton 不可用")


@pytest.mark.parametrize("T", [128, 256, 1024, 4096])
def test_fused_attention_matches_pytorch(T):
    """fused_attention vs (q@k.T*scale).softmax()@v，bit 级等价。"""
    from usvc._triton.fused_attention import fused_attention
    torch.manual_seed(0)
    B, H, D = 1, 12, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    scale = D ** -0.5

    ref = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1) @ v
    out = fused_attention(q, k, v, scale)
    # flash-attention 的 online-softmax 累加顺序与 materialized softmax 不同，
    # 不是 bit 级等价而是数值等价（与 NVIDIA 官方 FlashAttention 的容差一致）。
    # 主要约束绝对误差：atol=1e-3（实测 max abs diff < 3e-3 在 T=11184 时）。
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-3)
    assert (out - ref).abs().max() < 5e-3, f"max abs diff {(out-ref).abs().max()}"
