"""理论 occupancy 计算器。

nsys 给 registersPerThread + block 尺寸,用 CUDA Occupancy Calculator 公式
算理论 occupancy。这是 ncu 真实测量的近似(ncu 被容器锁,ERR_NVGPUCTRPERM),
但足以判瓶颈(如 conv 17% vs elementwise 100%,差异显著)。

公式(CUDA Programming Guide / Occupancy Calculator):
  warps_per_block = ceil(block_threads / 32)
  regs_per_warp   = ceil_256(regs_per_thread × 32)   # register allocation granularity
  warps_by_regs   = max_regs_per_sm / regs_per_warp
  warps_by_threads = max_threads_per_sm / 32
  warps_by_blocks  = max_blocks_per_sm × warps_per_block
  active_warps = min(above, aligned down to warps_per_block multiple)
  occupancy = active_warps / max_warps_per_sm

参考: docs/PROFILER_DESIGN.md §5.2
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuArchLimits:
    """每 SM 的硬件上限(按 GPU 架构)。"""
    compute_capability: tuple[int, int]
    max_threads_per_sm: int
    max_warps_per_sm: int        # = max_threads_per_sm / 32
    max_blocks_per_sm: int
    max_registers_per_sm: int
    register_alloc_granularity: int  # per-warp, 上取整倍数


# 常见架构常数表(NVIDIA 官方 Occupancy Calculator 数据)
# 来源: CUDA C Programming Guide, Table "Compute Capabilities"
_ARCH_LIMITS: dict[tuple[int, int], GpuArchLimits] = {
    # Ampere
    (8, 0): GpuArchLimits((8, 0), 2048, 64, 32, 65536, 256),
    (8, 6): GpuArchLimits((8, 6), 1536, 48, 16, 65536, 256),  # A10G = GA102 = sm86
    (8, 7): GpuArchLimits((8, 7), 1536, 48, 16, 65536, 256),
    (8, 9): GpuArchLimits((8, 9), 2048, 64, 24, 65536, 256),  # Hopper
    # Ada / Hopper
    (8, 8): GpuArchLimits((8, 8), 1536, 48, 16, 65536, 256),
    (8, 5): GpuArchLimits((8, 5), 2048, 64, 32, 65536, 256),
    # Turing
    (7, 5): GpuArchLimits((7, 5), 1024, 32, 16, 65536, 256),
    # Volta
    (7, 0): GpuArchLimits((7, 0), 2048, 64, 32, 65536, 256),
}


def get_arch_limits(compute_capability: tuple[int, int]) -> GpuArchLimits:
    """按 compute capability 取架构常数。未知架构回退到 sm86(A10G)。"""
    if compute_capability in _ARCH_LIMITS:
        return _ARCH_LIMITS[compute_capability]
    # 找最近的(同 major 的最高 minor)
    same_major = [k for k in _ARCH_LIMITS if k[0] == compute_capability[0]]
    if same_major:
        return _ARCH_LIMITS[max(same_major)]
    return _ARCH_LIMITS[(8, 6)]  # 默认 A10G


def theoretical_occupancy(
    registers_per_thread: int,
    block: tuple[int, int, int],
    arch: tuple[int, int] = (8, 6),
) -> float:
    """计算理论 occupancy(0-100)。

    Args:
        registers_per_thread: nsys 的 registersPerThread
        block: (blockX, blockY, blockZ) 线程块尺寸
        arch: compute capability, 默认 sm86 (A10G)
    Returns:
        occupancy 百分比(0-100)。registers=0(无信息)返回 -1 表示未知。
    """
    if registers_per_thread <= 0:
        return -1.0
    limits = get_arch_limits(arch)
    block_threads = block[0] * block[1] * block[2]
    if block_threads <= 0:
        return -1.0

    warps_per_block = (block_threads + 31) // 32
    # register allocation: per-warp, 上取整到 granularity(通常 256)
    regs_per_warp = ((registers_per_thread * 32) + limits.register_alloc_granularity - 1)
    regs_per_warp = (regs_per_warp // limits.register_alloc_granularity) * limits.register_alloc_granularity
    if regs_per_warp <= 0:
        return -1.0

    warps_by_regs = limits.max_registers_per_sm // regs_per_warp
    warps_by_threads = limits.max_warps_per_sm  # 等价于 max_threads/32
    warps_by_blocks = limits.max_blocks_per_sm * warps_per_block

    active_warps = min(warps_by_regs, warps_by_threads, warps_by_blocks)
    # 整块调度:active_warps 必须是 warps_per_block 的倍数
    active_warps = (active_warps // warps_per_block) * warps_per_block
    if active_warps <= 0:
        return 0.0
    return active_warps / limits.max_warps_per_sm * 100.0
