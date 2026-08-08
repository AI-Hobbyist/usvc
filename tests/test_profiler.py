"""M1 profiler 正确性守卫。

守护：
  1. peak 加载：值与 docs/peak_a10g.json 一致，不被硬编码污染
  2. kernel 分类：已知 kernel 名归到正确类别
  3. FLOPs 估计：matmul/conv/elementwise 的估算公式正确
  4. sanity 判据函数：ratio 计算 + pass 逻辑正确
（数值等价守卫 tests/test_triton_fusion.py 在 M2 落地 triton kernel 时再加。）
"""
import pytest
import torch

from usvc._profiler.peak import load_peak, _dtype_key
from usvc._profiler._kernel_flops import classify_kernel, estimate_op_flops, CUDNN_CUBLAS, \
    ELEMENTWISE_REDUCE, MEMCPY_TRANSFER, TRITON_FUSED
from usvc._profiler import roofline as rl


def test_peak_load_matches_json():
    p = load_peak()
    # 实测值范围检查（A10G 合理区间，非宣传值 125T）
    assert 30 < p.gemm_tflops["float32"] < 36, "fp32 峰值应在 ~34T"
    assert 60 < p.gemm_tflops["float16"] < 75, "fp16 峰值应在 ~68T"
    assert 60 < p.gemm_tflops["bfloat16"] < 75
    assert 550 < p.bw_read_gbs < 700, "读带宽应在 ~608 GB/s"
    assert p.sm_count == 80


def test_dtype_key_mapping():
    assert _dtype_key(torch.float16) == "float16"
    assert _dtype_key(torch.bfloat16) == "bfloat16"
    assert _dtype_key(torch.float32) == "float32"


def test_classify_kernel():
    assert classify_kernel("void cutlass::Kernel2<cutlass_80_simt_sgemm>") == CUDNN_CUBLAS
    assert classify_kernel("cudnnConvolutionForward") == CUDNN_CUBLAS
    assert classify_kernel("vectorized_elementwise_kernel<add>") == ELEMENTWISE_REDUCE
    assert classify_kernel("triton__0d1d2d3d_kernel") == TRITON_FUSED
    assert classify_kernel("void at::native::copy_kernel_cuda") == MEMCPY_TRANSFER


def test_estimate_op_flops_matmul():
    # aten::mm [M,K],[K,N] -> 2*M*K*N
    flops = estimate_op_flops("aten::mm", ([1024, 512], [512, 2048]))
    assert flops == 2 * 1024 * 512 * 2048


def test_estimate_op_flops_bmm():
    flops = estimate_op_flops("aten::bmm", ([8, 64, 128], [8, 128, 256]))
    assert flops == 2 * 8 * 64 * 128 * 256


def test_estimate_op_flops_conv():
    # conv2d [B,Cin,H,W], weight[Out,Cin,kH,kW]
    flops = estimate_op_flops("aten::conv2d", ([1, 64, 32, 32], [128, 64, 3, 3]))
    expected = 2 * 128 * 64 * 3 * 3 * (64 // 64) * 1 * 32 * 32
    assert flops == expected


def test_estimate_op_flops_elementwise():
    flops = estimate_op_flops("aten::add", ([1000, 1000], [1000, 1000]))
    assert flops == 1000 * 1000


def test_sanity_check_matmul_logic():
    p = load_peak()
    # 一次 34T 的 matmul: 8192^3, 假设耗时 = 2*8192^3 / 34e12 s
    dur_us = (2 * 8192 ** 3 / (p.gemm_tflops["float32"] * 1e12)) * 1e6
    sc = rl.sanity_check_matmul(8192, 8192, 8192, dur_us, p, torch.float32)
    assert sc["pass"] is True
    assert 99 <= sc["ratio_pct"] <= 101  # 刚好打满


def test_sanity_check_matmul_slow_fails():
    p = load_peak()
    # 耗时翻倍 = 一半性能，应不达标
    dur_us = (2 * 8192 ** 3 / (p.gemm_tflops["float32"] * 1e12)) * 1e6 * 2
    sc = rl.sanity_check_matmul(8192, 8192, 8192, dur_us, p, torch.float32)
    assert sc["pass"] is False
    assert sc["ratio_pct"] < 95
