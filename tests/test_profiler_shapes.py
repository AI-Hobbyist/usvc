"""conv/gemm shape 捕获 + FLOPs 计算的单元测试。

测 _compute_flops 公式正确性(纯函数, 不需 GPU/真实推理)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usvc._profiler.shape_capture import _compute_flops


def test_conv2d_flops():
    # (1,256,100,80) x (256,256,3,3) -> 2*256*256*9*100*80
    flops = _compute_flops("aten::conv2d", [(1, 256, 100, 80), (256, 256, 3, 3)])
    expected = 2 * 256 * 256 * 9 * 100 * 80
    assert flops == expected, f"{flops} != {expected}"


def test_conv1d_flops():
    # (1,512,L) x (512,512,K)
    flops = _compute_flops("aten::conv1d", [(1, 512, 1000), (512, 512, 3)])
    expected = 2 * 512 * 512 * 3 * 1000
    assert flops == expected


def test_mm_flops():
    # (128,256) x (256,512)
    flops = _compute_flops("aten::mm", [(128, 256), (256, 512)])
    expected = 2 * 128 * 512 * 256
    assert flops == expected


def test_bmm_flops():
    # (1,80,256) x (1,256,80)
    flops = _compute_flops("aten::bmm", [(1, 80, 256), (1, 256, 80)])
    expected = 2 * 1 * 80 * 80 * 256
    assert flops == expected


def test_addmm_flops():
    # bias(M,) x (M,K) x (K,N) -> 找两个 2D
    flops = _compute_flops("aten::addmm", [(128,), (128, 256), (256, 512)])
    expected = 2 * 128 * 512 * 256
    assert flops == expected


def test_matmul_2d():
    flops = _compute_flops("aten::matmul", [(64, 128), (128, 256)])
    assert flops == 2 * 64 * 256 * 128


def test_matmul_3d():
    flops = _compute_flops("aten::matmul", [(2, 64, 128), (2, 128, 256)])
    assert flops == 2 * 2 * 64 * 256 * 128


def test_linear_flops():
    # (*, in) x (out, in)
    flops = _compute_flops("aten::linear", [(8, 768), (3072, 768)])
    expected = 2 * 8 * 3072 * 768
    assert flops == expected


def test_linear_batched():
    # (1, 29999, 768) x (768, 768) -> batch dim 29999
    flops = _compute_flops("aten::linear", [(1, 29999, 768), (768, 768)])
    expected = 2 * 29999 * 768 * 768
    assert flops == expected


def test_unknown_op_returns_none():
    assert _compute_flops("aten::relu", [(1, 100)]) is None


def test_empty_shapes_returns_none():
    assert _compute_flops("aten::mm", []) is None
    assert _compute_flops("aten::conv2d", [()]) is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
