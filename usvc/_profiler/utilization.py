"""(C) 硬件利用率三件套：
  - SM occupancy：后台线程稳态采样
  - 内存带宽利用率：实测 Bytes/s ÷ 实测带宽峰值
  - 计算利用率：实测 FLOP/s ÷ 实测计算峰值（按 kernel 精度分桶）

SM occupancy 分两级：
  - 近似级（pynvml GPU utilization / nvidia-smi SM%）：稳态采样，0-100，反映 SM 是否有活跃 warp，
    但不是 §0.a 严格意义的 "warp occupancy"。标注 source="sm_activity_approx"。
  - 真值级（ncu）：sm__warps_active.avg.pct_of_peak_sustained_active，标注 source="warp_occupancy_ncu"。
判据 §0.a 用真值；若只能拿到近似级，报告里明确注明，不混为一谈。
"""
from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .peak import PeakData


@dataclass
class OccupancySample:
    value: float       # 0-100
    source: str        # "sm_activity_approx" | "warp_occupancy_ncu"


@dataclass
class UtilizationReport:
    sm_occupancy: Optional[OccupancySample] = None
    bandwidth_util_pct: Optional[float] = None   # 实测 Bytes/s ÷ 峰值
    compute_util_pct: Optional[float] = None     # 实测 FLOP/s ÷ 峰值
    measured_bytes_per_s: Optional[float] = None
    measured_flops_per_s: Optional[float] = None
    peak_bw_gbs: Optional[float] = None
    peak_compute_tflops: Optional[float] = None
    notes: list = field(default_factory=list)


class OccupancySampler:
    """后台线程稳态采样 SM activity（pynvml 近似级）。"""

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._nvml_ok = False
        self._err = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
        except Exception as e:
            self._err = str(e)

    @property
    def available(self) -> bool:
        return self._nvml_ok

    def _sample_once(self) -> Optional[float]:
        if not self._nvml_ok:
            return None
        try:
            # nvml "gpu" utilization = SM activity % (0-100)
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu)
        except Exception:
            return None

    def start(self):
        self._stop.clear()
        self._samples.clear()

        def loop():
            while not self._stop.is_set():
                v = self._sample_once()
                if v is not None:
                    self._samples.append(v)
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> OccupancySample:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return OccupancySample(value=float("nan"), source="sm_activity_approx")
        # 稳态：去掉前后各 10%（剔除 warmup/cooldown 尾巴），取中位数
        s = sorted(self._samples)
        n = len(s)
        trim = max(1, n // 10)
        core = s[trim: n - trim] if n > 2 * trim else s
        med = statistics.median(core)
        return OccupancySample(value=med, source="sm_activity_approx")

    def raw_samples(self) -> list[float]:
        return list(self._samples)


def ncu_warp_occupancy(command: list[str], *, kernel_filter: Optional[str] = None,
                       ncu_bin: str = "ncu") -> Optional[float]:
    """用 ncu 跑一次命令，解析 sm__warps_active 拿真 warp occupancy (%)。

    需要把目标程序包装成可执行命令。返回所有采样 kernel 的 warp occupancy 中位数。
    需要 root / CAP_SYS_ADMIN。失败返回 None。
    """
    try:
        metrics = "sm__warps_active.avg.pct_of_peak_sustained_active"
        cmd = [ncu_bin, "--metrics", metrics, "--csv", "--units", "base"]
        if kernel_filter:
            cmd += ["-k", kernel_filter]
        cmd += command
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        text = out.stdout
    except Exception:
        return None
    # 解析 CSV，找 metric 列
    vals = []
    for line in text.splitlines():
        if "sm__warps_active" in line or line.startswith('"Kernel"'):
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        for p in parts:
            p = p.rstrip("%")
            try:
                vals.append(float(p))
            except ValueError:
                continue
    if not vals:
        return None
    return statistics.median(vals)


def compute_utilization(measured_flops_per_s: float, peak: PeakData,
                         dtype, *, is_conv: bool = False) -> Optional[float]:
    """计算利用率 = 实测 FLOP/s ÷ 实测峰值。返回 %。"""
    peak_tflops = peak.compute_peak_tflops(dtype, is_conv=is_conv)
    if peak_tflops is None or measured_flops_per_s <= 0:
        return None
    return 100.0 * (measured_flops_per_s / 1e12) / peak_tflops


def bandwidth_utilization(measured_bytes_per_s: float, peak: PeakData) -> Optional[float]:
    """带宽利用率 = 实测 Bytes/s ÷ 实测带宽峰值。返回 %。"""
    peak_gbs = peak.memory_peak_gbs()
    if peak_gbs is None or measured_bytes_per_s <= 0:
        return None
    return 100.0 * (measured_bytes_per_s / 1e9) / peak_gbs
