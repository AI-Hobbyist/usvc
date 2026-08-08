"""(B) Kernel 时间分类：把 torch.profiler 的 kernel 事件按 6 类归类，输出占比 + 饼图数据。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ._kernel_flops import KernelStat, classify_kernel, CATEGORIES


@dataclass
class BreakdownReport:
    category_us: dict = field(default_factory=dict)   # category -> 累计 μs
    total_gpu_us: float = 0.0
    total_cpu_us: float = 0.0
    sync_idle_us: float = 0.0
    transfer_us: float = 0.0
    kernels: list = field(default_factory=list)        # 所有 KernelStat

    def category_pct(self) -> dict:
        if self.total_gpu_us <= 0:
            return {c: 0.0 for c in CATEGORIES}
        return {c: 100.0 * self.category_us.get(c, 0.0) / self.total_gpu_us for c in CATEGORIES}

    # §0 判据 b：胶水占比
    @property
    def glue_pct(self) -> float:
        """胶水 = elementwise/reduce 类（待融合的）。返回 % of GPU time。"""
        return self.category_pct().get("elementwise/reduce", 0.0)

    # §0 判据 c：传输/闲置占比
    @property
    def transfer_idle_pct(self) -> float:
        p = self.category_pct()
        return p.get("memcpy/transfer", 0.0) + p.get("sync/idle", 0.0)

    def to_dict(self) -> dict:
        return {
            "total_gpu_us": self.total_gpu_us,
            "total_cpu_us": self.total_cpu_us,
            "category_us": dict(self.category_us),
            "category_pct": self.category_pct(),
            "glue_pct": self.glue_pct,
            "transfer_idle_pct": self.transfer_idle_pct,
            "top_kernels": [
                {"name": k.name, "category": k.category, "us": k.total_us,
                 "count": k.count, "flops": k.flops, "bytes": k.bytes}
                for k in sorted(self.kernels, key=lambda x: -x.total_us)[:15]
            ],
        }


def build_breakdown(kernel_events: Iterable, *, cpu_sync_us: float = 0.0) -> BreakdownReport:
    """
    kernel_events: torch.profiler 的 key_averages()（已 group by key）
    cpu_sync_us: 额外的 CPU 等待时间（用于 §0.c 判据的 sync 部分）
    """
    rep = BreakdownReport()
    for cat in CATEGORIES:
        rep.category_us[cat] = 0.0

    # 设备类型常量（避免硬依赖 torch import）
    try:
        from torch.profiler import DeviceType
        CUDA_DEV = DeviceType.CUDA
    except Exception:
        CUDA_DEV = 2  # DeviceType.CUDA == 2

    for ev in kernel_events:
        key = ev.key
        dev_type = getattr(ev, "device_type", None)
        # 只统计 CUDA 设备侧事件（避免 aten:: CPU 包装事件重复计入 device 时间）。
        # aten::_softmax(CPU) 与 softmax_warp(CUDA) 都报 self_device_time，会重复。
        if dev_type is not None and dev_type != CUDA_DEV:
            # CPU 侧 aten 事件：只累计它的 CPU 时间（用于 §0.c 的 sync/idle 判据参考）
            cpu_us = getattr(ev, "self_cpu_time_total", 0) or 0
            rep.total_cpu_us += cpu_us / 1e3
            continue
        dev_us = getattr(ev, "self_device_time_total", 0) or 0
        dev_us = dev_us  # torch.profiler self_device_time_total 已是 μs，不要再除
        if dev_us <= 0:
            continue

        category = classify_kernel(key)
        rep.category_us[category] = rep.category_us.get(category, 0.0) + dev_us
        rep.total_gpu_us += dev_us
        rep.kernels.append(KernelStat(
            name=key, category=category, total_us=dev_us, count=ev.count
        ))

    # sync/idle 的 CPU 等待也算进 transfer/idle 判据
    rep.sync_idle_us += cpu_sync_us
    return rep
