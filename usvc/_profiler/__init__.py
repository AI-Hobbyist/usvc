"""usvc profiler 三件套（M1）。

统一入口 USVCProfiler 包裹一个 callable，跑 N 次，产出：
  (A) Roofline 图  — roofline.py
  (B) Kernel 时间分类饼图 — kernel_breakdown.py
  (C) 硬件利用率三件套 — utilization.py

所有 roofline / 利用率除数读 docs/peak_a10g.json 实测峰值，不硬编码宣传值。
"""
from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.profiler import profile, ProfilerActivity

from .peak import PeakData, load_peak
from . import roofline as _rl
from . import kernel_breakdown as _kb
from . import utilization as _ut
from .report import ProfileReport

__all__ = [
    "USVCProfiler", "ProfileReport", "PeakData", "load_peak",
    "profile_roofline", "profile_kernels", "sample_utilization",
]


class USVCProfiler:
    """统一 profile 入口。

    用法:
        prof = USVCProfiler("mel_baseline", out_dir=Path("docs/prof/mel"))
        report = prof.run(fn, flops_per_iter=..., bytes_per_iter=...)
        report.save_json(); report.plot_all(); print(report.verdict())
    """

    def __init__(self, name: str, out_dir: Path, *,
                 warmup: int = 3, iters: int = 10,
                 peak: Optional[PeakData] = None,
                 sample_interval_s: float = 0.005,
                 allow_tf32: bool = True):
        """allow_tf32: profile 期间是否开 TF32（默认 True，与 docs/bench_peak.py 记录峰值时一致）。
        peak json 的 gemm_FP32_tflops 实测值是在 TF32=on 下测得的，故 fp32 的 roofline
        除数与判据都基于此。若 workload 强制纯 fp32，应同时换用纯 fp32 实测峰值。"""
        self.name = name
        self.out_dir = Path(out_dir)
        self.warmup = warmup
        self.iters = iters
        self.peak = peak or load_peak()
        self.sample_interval_s = sample_interval_s
        self._allow_tf32 = allow_tf32

    def run(self, fn: Callable, *,
            flops_per_iter: Optional[int] = None,
            bytes_per_iter: Optional[int] = None,
            ramp_gpu_s: float = 3.0) -> ProfileReport:
        # 设置 TF32，与 peak 记录条件一致（docs/bench_peak.py 顶层 allow_tf32=True）
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self._allow_tf32
        try:
            # GPU clock ramp：持续负载把频率拉到峰值状态（bench_peak 全扫描自然 warm，单点需人工 ramp）
            if ramp_gpu_s > 0:
                self._ramp_gpu(ramp_gpu_s)
            return self._run_impl(fn, flops_per_iter, bytes_per_iter)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    def _ramp_gpu(self, seconds: float):
        """用大方阵 matmul 持续打满 GPU seconds 秒，把 clock 拉到稳态峰值。"""
        n = 4096
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            torch.matmul(a, b)
        torch.cuda.synchronize()
        del a, b
        gc.collect(); torch.cuda.empty_cache()

    def _run_impl(self, fn, flops_per_iter, bytes_per_iter) -> ProfileReport:
        # 1. 计时（不带 profiler，最干净）
        iter_us = self._time_iters(fn)

        # 2. occupancy 后台采样 + profiler 跑一次拿 kernel/op 统计
        sampler = _ut.OccupancySampler(interval_s=self.sample_interval_s)
        sampler.start()
        try:
            prof = self._profile_once(fn)
        finally:
            occ = sampler.stop()

        # 3. 构建报告
        br = _kb.build_breakdown(prof.key_averages())
        rl_pts = _rl.collect_roofline_points(list(prof.key_averages()))
        rl = _rl.RooflineReport(points=rl_pts, peak=self.peak)
        rl._name = self.name  # type: ignore[attr-defined]

        # 利用率：用总 FLOPs/Bytes 估
        measured_flops_per_s = None
        measured_bytes_per_s = None
        if flops_per_iter and iter_us:
            measured_flops_per_s = flops_per_iter / (statistics.median(iter_us) * 1e-6)
        if bytes_per_iter and iter_us:
            measured_bytes_per_s = bytes_per_iter / (statistics.median(iter_us) * 1e-6)
        # 计算/带宽利用率用主导 dtype（flops_per_iter 一般是 fp32/tf32 训练场景）
        util_dtype = torch.float32
        util = _ut.UtilizationReport(
            sm_occupancy=occ,
            measured_flops_per_s=measured_flops_per_s,
            measured_bytes_per_s=measured_bytes_per_s,
            peak_bw_gbs=self.peak.memory_peak_gbs(),
        )
        util.compute_util_pct = _ut.compute_utilization(
            measured_flops_per_s or 0, self.peak, util_dtype) if measured_flops_per_s else None
        util.bandwidth_util_pct = _ut.bandwidth_utilization(
            measured_bytes_per_s or 0, self.peak) if measured_bytes_per_s else None
        if not sampler.available:
            util.notes.append(f"pynvml 不可用 ({sampler._err})，SM occupancy 缺失")

        rep = ProfileReport(
            name=self.name, out_dir=self.out_dir, peak=self.peak,
            breakdown=br, roofline=rl, utilization=util,
            iter_us_median=statistics.median(iter_us) if iter_us else None,
            flops_per_iter=flops_per_iter, bytes_per_iter=bytes_per_iter,
            iters=self.iters, warmup=self.warmup,
        )
        return rep

    def _time_iters(self, fn) -> list:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(self.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)  # μs
        return ts

    def _profile_once(self, fn):
        # acc_events=True: 保留所有事件跨 cycle
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True, with_stack=False, with_modules=False,
        ) as prof:
            for _ in range(max(self.warmup, 1)):
                fn()
            torch.cuda.synchronize()
            for _ in range(self.iters):
                fn()
            torch.cuda.synchronize()
        return prof


# ---- 函数式接口（对应 §1.2 的导出名）----
def profile_roofline(fn, name="op", out_dir=Path("docs/prof"), **kw) -> ProfileReport:
    """产 (A) roofline。"""
    return USVCProfiler(name, Path(out_dir), **kw).run(fn)


def profile_kernels(fn, name="op", out_dir=Path("docs/prof"), **kw) -> ProfileReport:
    """产 (B) kernel 分类。"""
    return USVCProfiler(name, Path(out_dir), **kw).run(fn)


def sample_utilization(fn, name="op", out_dir=Path("docs/prof"), **kw) -> ProfileReport:
    """产 (C) 利用率。"""
    return USVCProfiler(name, Path(out_dir), **kw).run(fn)
