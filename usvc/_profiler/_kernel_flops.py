"""Kernel 分类 + FLOPs/Bytes 估计。

torch.profiler 的 CUDA kernel 事件名字是底层 kernel 名（如 cutlass/vectorized_elementwise），
不带 tensor 形状。我们：
  1. 按 kernel 名模式分类到 6 类（cudnn/cublas, triton-fused, elementwise/reduce, memcpy, sync/idle, other）
  2. FLOPs/Bytes 用「同一次 profile 的 aten op 事件（带 input_shapes）」来估算，
     再按 kernel 时间比例分摊。无法确定的真实 kernel 记为 unknown（不瞎猜），在 roofline 上单独标注。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import torch

# ---- 分类 ----
CUDNN_CUBLAS = "cudnn/cublas"
TRITON_FUSED = "triton-fused"
ELEMENTWISE_REDUCE = "elementwise/reduce"
MEMCPY_TRANSFER = "memcpy/transfer"
SYNC_IDLE = "sync/idle"
OTHER = "other"

CATEGORIES = [CUDNN_CUBLAS, TRITON_FUSED, ELEMENTWISE_REDUCE, MEMCPY_TRANSFER, SYNC_IDLE, OTHER]

# kernel 名匹配规则（按优先级，先命中先用）
_KERNEL_PATTERNS = [
    # cuBLAS / cutlass GEMM (含 aten::mm/bmm/addmm/linear —— 它们最终走 cublas)
    (CUDNN_CUBLAS, re.compile(r"cutlass|cublas|gemm|sgemm|gemv|getrf|getri|trsm|"
                              r"aten::mm|aten::bmm|aten::addmm|aten::linear|aten::matmul", re.I)),
    # cuDNN conv / winograd
    (CUDNN_CUBLAS, re.compile(r"cudnn|winograd|convolv|spatial", re.I)),
    # cuFFT / FFT (STFT 主导算子，已优化不可压)
    (CUDNN_CUBLAS, re.compile(r"fft|_fft|vector_fft", re.I)),
    # Triton fused kernel（我们自己写的 + torch._inductor）。严格匹配 triton 命名，
    # 避免 _kernel$ 误吃 at::native::elementwise_kernel（后者由 elementwise 规则处理）。
    (TRITON_FUSED, re.compile(r"triton_+_?\d|inductor|program_id|_triton_|triton::", re.I)),
    # elementwise / reduce（胶水）
    (ELEMENTWISE_REDUCE, re.compile(
        r"elementwise|reduce|unary|binary|tensor_iterator|_kernel_vectorized|"
        r"aten::add|aten::sub|aten::mul|aten::div|aten::abs|aten::neg|aten::exp|aten::log|aten::sqrt|aten::clamp|aten::relu|aten::gelu|"
        r"aten::_softmax|softmax|add|sub|mul|div|relu|clamp|sigmoid|tanh|exp|log|sqrt|gelu|"
        r"_sum|_max|_min|_mean|argmax|argmin|norm", re.I)),
    # memcpy / transfer
    (MEMCPY_TRANSFER, re.compile(r"memcpy|cudaMemcpy|copy_kernel|tensor_copy|set_|fill_|zero_|"
                                 r"cudaMemset|to_copy|_to_copy|aten::copy_|aten::to", re.I)),
    # sync / idle（这些通常 self_device_time=0，是 CPU 侧）
    (SYNC_IDLE, re.compile(r"cudaDeviceSynchronize|cudaStreamSynchronize|cudaEvent|"
                           r"StreamSync|EventSync|cudaStreamWait", re.I)),
]


def classify_kernel(name: str) -> str:
    """按 kernel 名分类。"""
    for cat, pat in _KERNEL_PATTERNS:
        if pat.search(name):
            return cat
    return OTHER


# ---- aten op → FLOPs/Bytes 估计 ----
def estimate_op_flops(op_name: str, shapes) -> Optional[int]:
    """从 aten op 名 + input_shapes 估算 FLOPs。返回 None 表示无法确定。"""
    if not shapes:
        return None
    name = op_name.lower()
    # 去掉 aten:: / autograd:: 前缀，只看 op 本体
    short = name.split("::")[-1]

    # matmul: shapes=[[M,K],[K,N]]
    if "matmul" in name or "bmm" in name or "linear" in name or "addmm" in name or short in ("mm", "bmm"):
        try:
            # 找两个 2D+ 矩阵
            mats = [s for s in shapes if s and len(s) >= 2]
            if len(mats) >= 2:
                a, b = mats[0], mats[1]
                # batched bmm
                if len(a) == 3 and len(b) == 3:
                    batch = a[0]
                    return 2 * batch * a[1] * a[2] * b[2]
                m, k = a[-2], a[-1]
                n = b[-1]
                return 2 * m * k * n
        except Exception:
            return None

    # conv (1d/2d/3d): shapes=[input[B,C,...], weight[out,in,k...], ...]
    if "conv" in name:
        try:
            inp = shapes[0]
            w = shapes[1]
            if inp and w and len(inp) >= 2 and len(w) >= 2:
                # 输出 spatial = 输入 spatial (same padding 近似)
                spatial = 1
                for d in inp[2:]:
                    spatial *= d
                flops_per = 2 * w[0] * w[1]
                for k in w[2:]:
                    flops_per *= k
                # groups
                return flops_per * (inp[1] // w[1]) * inp[0] * spatial
        except Exception:
            return None

    # reduce: sum/mean/max/min over dim — FLOPs≈ element 数（不算浮点乘，但计入算力负载近似）
    if any(x in name for x in ("_sum", "sum", "_mean", "mean", "argmax", "argmin", "_max", "_min")):
        try:
            out = shapes[-1] if shapes else None
            if out:
                n = 1
                for d in out:
                    n *= d
                return n
        except Exception:
            return None

    # elementwise: add/mul/relu/... — 按 element 数算（1 FLOP/element 近似）
    if any(x in name for x in ("add", "sub", "mul", "div", "relu", "clamp", "sigmoid",
                               "tanh", "exp", "log", "sqrt", "neg", "abs")):
        try:
            out = shapes[-1] if shapes else (shapes[0] if shapes else None)
            if out:
                n = 1
                for d in out:
                    n *= d
                return n
        except Exception:
            return None

    return None


def estimate_op_bytes(op_name: str, shapes, dtypes) -> Optional[int]:
    """估算 op 的字节移动量（读输入 + 写输出）。近似：每个参与张量读/写一遍。"""
    if not shapes:
        return None
    total = 0
    for sh, dt in zip(shapes, dtypes or []):
        if not sh:
            continue
        n = 1
        for d in sh:
            n *= d
        bytes_per = _dtype_bytes(dt)
        total += n * bytes_per
    # 至少算一次写
    return total if total > 0 else None


def _dtype_bytes(dt) -> int:
    if dt is None:
        return 4
    s = str(dt)
    if "float64" in s or "double" in s:
        return 8
    if "float16" in s or "half" in s or "bfloat16" in s:
        return 2
    if "float32" in s or "float" in s:
        return 4
    if "int8" in s or "uint8" in s or "bool" in s:
        return 1
    return 4


@dataclass
class KernelStat:
    """单个 kernel（聚合后）的统计。"""
    name: str
    category: str
    total_us: float          # 累计 GPU 时间 (μs)
    count: int
    # roofline 估算（可能为 None = 无法确定，不画进 roofline 的精确点）
    flops: Optional[int] = None
    bytes: Optional[int] = None

    @property
    def arithmetic_intensity(self) -> Optional[float]:
        if self.flops and self.bytes:
            return self.flops / self.bytes
        return None

    @property
    def achieved_flops_per_s(self) -> Optional[float]:
        if self.flops and self.total_us > 0:
            return self.flops / (self.total_us * 1e-6)
        return None
