"""从 nsys kernel 名 + grid/block 反推 FLOPs/Bytes。

nsys 的 KERNEL 表没有 tensor shapes(torch.profiler 的 aten op 有,但 nsys 走 CUDA 层),
所以只能靠 kernel 名模式 + grid/block 做有限反推。

策略(诚实分级):
- GEMM/CONV kernel 名含 tensor core 尺寸信息(如 s16816 = 16x8x16),
  但不含具体 M/N/K。无法精确反推 FLOPs —— 标 unknown,不画 roofline 精确点。
- elementwise/reduce: 按 grid 维度(=输出元素数)反推 numel,算 Bytes。
  这是可靠的(grid ≈ 输出 size)。FLOPs 按 op 类型估。
- memcpy: nsys 给真实 bytes,直接用。

关键原则:算不出的绝不编造坐标。roofline 上这类 kernel 按耗时大小显示,不定位。
"""
from __future__ import annotations

import re
from typing import Optional

# 复用现有分类器
from ._kernel_flops import classify_kernel, ELEMENTWISE_REDUCE, MEMCPY_TRANSFER, CUDNN_CUBLAS


def estimate_kernel_bytes_from_grid(
    kernel_name: str,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
) -> Optional[int]:
    """从 grid 反推 elementwise/reduce 的 Bytes。

    grid 总线程数 ≈ 输出元素数(逐元素 kernel 每线程一个元素)。
    Bytes = 2 × numel × dtype_size (读输入 + 写输出)。
    dtype_size 无法从 kernel 名确定,假设 fp16/fp32 混合取 2(SVC 推理多为 fp16 autocast)。
    """
    cat = classify_kernel(kernel_name)
    if cat != ELEMENTWISE_REDUCE:
        return None
    gx, gy, gz = grid
    numel = gx * gy * gz
    if numel <= 0:
        return None
    # 推断 dtype: kernel 名含 fp16/half 标志则 2,否则保守取 4(fp32)
    dtype_size = 2 if re.search(r"f16|fp16|half|bf16|bfloat16|_half", kernel_name, re.I) else 4
    # elementwise: 读1份 + 写1份 (binary op 读2份, 保守用2读1写=3)
    if re.search(r"binary|CUDAFunctor_add|CUDAFunctor_mul|add|mul", kernel_name, re.I):
        return numel * dtype_size * 3  # 2读1写
    return numel * dtype_size * 2      # 1读1写


def estimate_kernel_flops_from_grid(
    kernel_name: str,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
) -> Optional[int]:
    """elementwise/reduce 的 FLOPs(粗估,1 FLOP/element)。

    GEMM/CONV 无法从 kernel 名反推(无 M/N/K),返回 None。
    """
    cat = classify_kernel(kernel_name)
    if cat != ELEMENTWISE_REDUCE:
        return None
    numel = grid[0] * grid[1] * grid[2]
    if numel <= 0:
        return None
    return numel  # 1 FLOP/element 近似(leaky_relu/add 等)


def is_roofline_plottable(kernel_name: str) -> bool:
    """该 kernel 能否画进 roofline 的精确散点(有可信 FLOPs+Bytes)。

    只有 elementwise/reduce 类能从 grid 可靠反推。GEMM/CONV 无 shapes 不画精确点。
    """
    cat = classify_kernel(kernel_name)
    return cat == ELEMENTWISE_REDUCE
