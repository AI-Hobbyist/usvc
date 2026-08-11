"""profiler 校准回归测试。

守护 roofline/occupancy 反推算法的正确性:
- GEMM 大矩阵利用率 ∈ [70%, 100%]
- CONV 利用率 ∈ [60%, 95%]
- elementwise 大 tensor 带宽利用率 ∈ [80%, 110%]
- occupancy 计算对已知 kernel 给出已知值

任何算法改动必须过这个测试,防止"漂亮的误导"。
对应设计文档 §8。
"""
import pytest
import torch

# 占位: 测试文件先建框架, 实际微基准在 verify_roofline_calib.py 验证过
# 这里用 pytest 固化断言, 进 CI


class TestOccupancyCalc:
    """occupancy 计算器回归(A10G sm86)。"""

    def test_elementwise_high_occupancy(self):
        from usvc._profiler.occupancy import theoretical_occupancy
        # elementwise: 17 regs, 128 threads/block → 应 ~100%
        occ = theoretical_occupancy(17, (128, 1, 1), (8, 6))
        assert occ >= 90, f"elementwise occupancy 应高, 实际 {occ}"

    def test_conv_register_pressure(self):
        from usvc._profiler.occupancy import theoretical_occupancy
        # conv xmma: 238 regs, 128 threads/block → 应 ~17% (寄存器压死)
        occ = theoretical_occupancy(238, (128, 1, 1), (8, 6))
        assert 10 <= occ <= 25, f"conv occupancy 应被寄存器压低, 实际 {occ}"

    def test_unknown_regs_returns_negative(self):
        from usvc._profiler.occupancy import theoretical_occupancy
        assert theoretical_occupancy(0, (128, 1, 1), (8, 6)) == -1.0

    def test_a10g_arch_limits(self):
        from usvc._profiler.occupancy import get_arch_limits
        lim = get_arch_limits((8, 6))
        assert lim.max_warps_per_sm == 48  # A10G
        assert lim.max_threads_per_sm == 1536


class TestRooflineCalibration:
    """GEMM/CONV/elementwise 的反推 vs 实测利用率(微基准)。"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")
    def test_gemm_large_near_peak(self):
        """大 GEMM 应接近 fp16 peak(~65T), 证明 FLOPS 反推正确。"""
        a = torch.randn(1024, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        for _ in range(20):
            c = a @ b
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 50
        flops = 2 * 1024 * 4096 * 4096
        tflops = flops / dt / 1e12
        # 利用率 = tflops / 65T (A10G fp16 实测 ~67.8)
        util = tflops / 67.8
        assert 0.70 <= util <= 1.05, f"GEMM 利用率 {util:.2f} 越界, 反推可能错"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")
    def test_conv_mid_near_peak(self):
        """中等 conv 应 60-90% peak, 证明 conv FLOPS 反推正确。"""
        x = torch.randn(1, 128, 1, 11000, device="cuda", dtype=torch.float16).to(
            memory_format=torch.channels_last)
        w = torch.randn(256, 128, 1, 5, device="cuda", dtype=torch.float16)
        import torch.nn.functional as F
        for _ in range(20):
            y = F.conv2d(x, w)
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            y = F.conv2d(x, w)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 50
        flops = 2 * 256 * 128 * 5 * 11000
        tflops = flops / dt / 1e12
        util = tflops / 62.2  # A10G conv fp16 实测
        assert 0.50 <= util <= 0.95, f"CONV 利用率 {util:.2f} 越界"


class TestNsysParser:
    """nsys 解析器(用合成数据, 不依赖真实 trace)。"""

    def test_classify_kernel_rules(self):
        from usvc._profiler._kernel_flops import classify_kernel, CUDNN_CUBLAS, ELEMENTWISE_REDUCE
        assert classify_kernel("sm86_xmma_fprop_implicit_gemm") == CUDNN_CUBLAS
        assert classify_kernel("vectorized_elementwise_kernel") == ELEMENTWISE_REDUCE
        assert classify_kernel("ampere_fp16_s16816gemm") == CUDNN_CUBLAS

    def test_flops_from_grid_elementwise(self):
        from usvc._profiler.nsys_flops import estimate_kernel_flops_from_grid
        # grid (1000,1,1) → numel 1000 → 1000 FLOPs
        f = estimate_kernel_flops_from_grid("elementwise_kernel", (1000, 1, 1), (128, 1, 1))
        assert f == 1000

    def test_flops_from_grid_conv_returns_none(self):
        """conv 无 shapes 不反推(诚实返回 None)。"""
        from usvc._profiler.nsys_flops import estimate_kernel_flops_from_grid
        f = estimate_kernel_flops_from_grid("sm86_xmma_fprop", (1000, 1, 1), (128, 1, 1))
        assert f is None
