"""(A) Roofline 图。

策略：
  torch.profiler 的 CUDA kernel 事件不带 shape，无法直接算 FLOPs。
  我们用「aten op 事件（带 input_shapes + input_types）」估算每个 op 的 FLOPs/Bytes，
  再把 op 时间（cpu/device）作为该 op 的实测时间 → 一个 roofline 点。
  对纯底层 kernel（无对应 op）的，按 kernel 分类近似：cublas→用相邻 matmul op 的强度，elementwise→按带宽侧。

  实操上：我们把「带 shape 的 aten op」作为 roofline 的点（最可信），这些点已能覆盖
  matmul/conv/softmax/reduce/elementwise 等主导算子，足够判断 memory/compute bound。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .peak import PeakData
from ._kernel_flops import estimate_op_flops, estimate_op_bytes


@dataclass
class RooflinePoint:
    name: str
    flops: int
    bytes: int
    duration_us: float
    dtype: str
    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes if self.bytes > 0 else float("inf")
    @property
    def achieved_flops_per_s(self) -> float:
        return self.flops / (self.duration_us * 1e-6) if self.duration_us > 0 else 0.0


@dataclass
class RooflineReport:
    points: list = field(default_factory=list)
    peak: Optional[PeakData] = None
    # 用于判据 d 的「主导点」
    def dominant_point(self) -> Optional[RooflinePoint]:
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.flops * 1e-9 * max(p.duration_us, 1e-6))


def collect_roofline_points(op_events, kernels_for_timing=None) -> list:
    """从 aten op 事件（带 shapes）收集 roofline 点。

    op_events: torch.profiler events (aten 级，带 input_shapes/input_types)
    kernels_for_timing: 可选，从 CUDA kernel 时间回填（目前用 op 自带 device_time）
    """
    pts = []
    for ev in op_events:
        key = ev.key
        # 只要 aten 级 op
        if not (key.startswith("aten::") or "::" in key):
            continue
        shapes = getattr(ev, "input_shapes", ()) or ()
        # input_types 在新版 torch.profiler 里可能要 ev.stack 或不可得；尝试
        dtypes = []
        try:
            # key_averages 不直接给 types，退化为 float32 估算
            dtypes = ["CPUFloatType"] * len(shapes)
        except Exception:
            pass
        flops = estimate_op_flops(key, shapes)
        bytes_ = estimate_op_bytes(key, shapes, dtypes)
        dur_us = (getattr(ev, "self_device_time_total", 0) or 0) / 1e3  # ns->μs
        if flops is None or bytes_ is None or dur_us <= 0:
            continue
        pts.append(RooflinePoint(
            name=key, flops=flops, bytes=bytes_, duration_us=dur_us,
            dtype="float32",
        ))
    return pts


def roofline_boundaries(peak: PeakData):
    """返回 roofline 的计算水平线 + 带宽斜线参数（用于绘图）。

    返回 dict：
      compute_tflops_by_dtype: {dtype_key: tflops}
      bw_gbs: 带宽峰值
      ridge_ai: 各 dtype 的脊点 AI = compute_peak / bw_peak (FLOP/Byte)
    """
    out = {"compute_tflops": {}, "bw_gbs": peak.memory_peak_gbs(), "ridge_ai": {}}
    for dk, tf in peak.gemm_tflops.items():
        if tf and peak.bw_gbs:
            out["compute_tflops"][dk] = tf
            # ridge AI (FLOP/Byte) = TFLOPS*1e12 / (GB/s*1e9) = TFLOPS/(GB/s)*1e3
            out["ridge_ai"][dk] = (tf * 1e12) / (peak.bw_gbs * 1e9)
    return out


def point_compute_util(pt: RooflinePoint, peak: PeakData) -> Optional[float]:
    """点距计算 roofline 的距离 %。"""
    peak_tf = peak.compute_peak_tflops(torch.float32)
    if peak_tf is None or pt.achieved_flops_per_s <= 0:
        return None
    return 100.0 * (pt.achieved_flops_per_s / 1e12) / peak_tf


def point_is_memory_bound(pt: RooflinePoint, peak: PeakData) -> Optional[bool]:
    """点的 AI < ridge AI => memory bound。"""
    b = roofline_boundaries(peak)
    ridge = b["ridge_ai"].get("float32")
    if ridge is None:
        return None
    return pt.arithmetic_intensity < ridge


def sanity_check_matmul(M: int, K: int, N: int, duration_us: float,
                         peak: PeakData, dtype=torch.float32) -> dict:
    """sanity check：纯 matmul 应落在实测 roofline 95% 以内。

    返回 {flops, achieved_tflops, peak_tflops, ratio_pct, pass}
    """
    flops = 2 * M * K * N
    achieved = flops / (duration_us * 1e-6)
    achieved_tf = achieved / 1e12
    peak_tf = peak.compute_peak_tflops(dtype)
    ratio = (100.0 * achieved_tf / peak_tf) if peak_tf else float("nan")
    return {
        "shape": f"{M}x{K}x{N}",
        "flops": flops,
        "duration_us": duration_us,
        "achieved_tflops": achieved_tf,
        "peak_tflops": peak_tf,
        "ratio_pct": ratio,
        "pass": bool(ratio >= 95.0) if peak_tf else False,
    }
