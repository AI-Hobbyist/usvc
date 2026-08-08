"""加载本机实测峰值 (docs/peak_a10g.json)，提供按 dtype 分桶的峰值查询。

所有 roofline / 利用率计算的除数必须经过这里，不得硬编码宣传值。
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Optional

import torch

# 默认路径：仓库根 docs/peak_a10g.json
_DEFAULT_PEAK_PATH = Path(__file__).resolve().parents[2] / "docs" / "peak_a10g.json"


class PeakData:
    """本机实测峰值的只读视图。"""

    def __init__(self, data: dict):
        self._data = data
        peaks = data.get("peaks", {})
        # 计算峰值 (TFLOPS)，按精度桶
        self.gemm_tflops = {
            "float32": peaks.get("gemm_FP32_tflops"),   # 34.1
            "tf32": peaks.get("gemm_TF32_tflops"),      # 34.1 (≈FP32)
            "float16": peaks.get("gemm_FP16_tflops"),   # 68.4
            "bfloat16": peaks.get("gemm_BF16_tflops"),  # 69.0
        }
        # conv 峰值 (cuDNN, TFLOPS)
        self.conv_tflops = {
            "float32": peaks.get("conv_FP32_tflops"),
            "float16": peaks.get("conv_FP16_tflops"),
            "bfloat16": peaks.get("conv_BF16_tflops"),
        }
        # HBM 带宽 (GB/s)
        self.bw_read_gbs = peaks.get("bw_read_gbs")    # 608
        self.bw_write_gbs = peaks.get("bw_write_gbs")  # 666
        self.bw_copy_gbs = peaks.get("bw_copy_gbs")
        # 读写综合带宽（roofline 斜线用：单 kernel 通常读输入 + 写输出，取读/写均值较保守）
        # 取 min(read, write) 更保守，取 read 更常见（多数 kernel 读多于写）。这里用 read 作为 roofline 斜线分母
        # （memory-bound kernel 通常被读带宽限制）。
        self.bw_gbs = self.bw_read_gbs
        # SM 信息
        gi = data.get("gpu_info", {})
        self.sm_count = gi.get("multiprocessor_count")
        self.max_threads_per_sm = gi.get("max_threads_per_multi_processor")
        self.gpu_name = gi.get("name")
        self.cuda_capability = gi.get("cuda_capability")

    # ---- 计算峰值查询 ----
    def compute_peak_tflops(self, dtype: torch.dtype, *, is_conv: bool = False) -> Optional[float]:
        """按 dtype 返回实测计算峰值 (TFLOPS)。找不到返回 None。"""
        key = _dtype_key(dtype)
        tbl = self.conv_tflops if is_conv else self.gemm_tflops
        v = tbl.get(key)
        if v is None and key in ("float16", "bfloat16"):
            # conv 表缺时退回 gemm 同精度
            v = self.gemm_tflops.get(key)
        return v

    def memory_peak_gbs(self) -> Optional[float]:
        return self.bw_gbs

    def as_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "sm_count": self.sm_count,
            "max_threads_per_sm": self.max_threads_per_sm,
            "gemm_tflops": self.gemm_tflops,
            "conv_tflops": self.conv_tflops,
            "bw_read_gbs": self.bw_read_gbs,
            "bw_write_gbs": self.bw_write_gbs,
        }


def _dtype_key(dtype) -> str:
    """torch.dtype -> 峰值表里的 key。"""
    if dtype in (torch.float16,):
        return "float16"
    if dtype in (torch.bfloat16,):
        return "bfloat16"
    if dtype in (torch.float32,):
        # 区分 tf32 由调用方处理；默认 float32 当作 fp32 峰值（更保守，因训练开 TF32 后实际更高）
        return "float32"
    if dtype in (torch.float64,):
        return "float32"  # 退化，没有 fp64 峰值，借用 fp32（仅用于估算，不影响判据）
    if dtype in (torch.int8, torch.uint8):
        return "float16"  # int8 大致按 fp16 tensor core 计
    return "float32"


def load_peak(path: Optional[Path] = None) -> PeakData:
    """加载实测峰值。path 默认 docs/peak_a10g.json。"""
    p = Path(path) if path else _DEFAULT_PEAK_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"peak json 不存在: {p}。请先跑 docs/bench_peak.py 生成本机实测峰值。"
        )
    with open(p) as f:
        return PeakData(json.load(f))


@functools.lru_cache(maxsize=1)
def _cached_peak() -> PeakData:
    return load_peak()
