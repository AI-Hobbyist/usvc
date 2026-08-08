"""报告数据结构 + 绘图（roofline / 饼图 / 利用率柱图）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .peak import PeakData
from .kernel_breakdown import BreakdownReport
from .roofline import RooflineReport, RooflinePoint, roofline_boundaries, point_compute_util
from .utilization import UtilizationReport, OccupancySample


@dataclass
class ProfileReport:
    name: str
    out_dir: Path
    peak: PeakData
    breakdown: BreakdownReport
    roofline: RooflineReport
    utilization: UtilizationReport
    iter_us_median: Optional[float] = None  # 单次 iter 纯耗时 (μs)
    flops_per_iter: Optional[int] = None
    bytes_per_iter: Optional[int] = None
    iters: int = 0
    warmup: int = 0

    def save_json(self):
        d = {
            "name": self.name,
            "iters": self.iters,
            "warmup": self.warmup,
            "iter_us_median": self.iter_us_median,
            "flops_per_iter": self.flops_per_iter,
            "bytes_per_iter": self.bytes_per_iter,
            "peak": self.peak.as_dict(),
            "breakdown": self.breakdown.to_dict(),
            "utilization": _util_to_dict(self.utilization),
            "roofline_points": [
                {"name": p.name, "flops": p.flops, "bytes": p.bytes,
                 "duration_us": p.duration_us, "AI": p.arithmetic_intensity,
                 "achieved_tflops": p.achieved_flops_per_s / 1e12}
                for p in self.roofline.points
            ],
        }
        out = Path(self.out_dir) / f"{self.name}_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(d, f, indent=2, default=str)
        return out

    def plot_all(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _plot_roofline(self.roofline, self.peak, self.out_dir / f"{self.name}_roofline.png")
        _plot_pie(self.breakdown, self.out_dir / f"{self.name}_kernel_pie.png")
        _plot_util(self.utilization, self.out_dir / f"{self.name}_utilization.png")

    def verdict(self) -> dict:
        """对 §0 的 a/b/c 判据给出当前是否达标（d 改参无效需人工再跑一次）。"""
        u = self.utilization
        b = self.breakdown
        sm_ok = (u.sm_occupancy and not math.isnan(u.sm_occupancy.value)
                 and u.sm_occupancy.value >= 85.0) if u.sm_occupancy else False
        glue_ok = b.glue_pct <= 5.0
        transfer_ok = b.transfer_idle_pct < 3.0
        return {
            "a_sm_occupancy_85": {"pass": sm_ok, "value": u.sm_occupancy.value if u.sm_occupancy else None,
                                   "source": u.sm_occupancy.source if u.sm_occupancy else None,
                                   "note": "近似=SM activity(pynvml), 真值=warp occupancy(ncu)"},
            "b_glue_le_5pct": {"pass": glue_ok, "value": b.glue_pct},
            "c_transfer_idle_lt_3pct": {"pass": transfer_ok, "value": b.transfer_idle_pct},
            "d_param_invariant": {"pass": None, "note": "需人工改参再跑一次验证"},
            "all_pass": sm_ok and glue_ok and transfer_ok,
        }


def _util_to_dict(u: UtilizationReport) -> dict:
    return {
        "sm_occupancy": {"value": u.sm_occupancy.value, "source": u.sm_occupancy.source} if u.sm_occupancy else None,
        "bandwidth_util_pct": u.bandwidth_util_pct,
        "compute_util_pct": u.compute_util_pct,
        "measured_bytes_per_s": u.measured_bytes_per_s,
        "measured_flops_per_s": u.measured_flops_per_s,
        "notes": u.notes,
    }


import math


def _plot_roofline(roofline: RooflineReport, peak: PeakData, out: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    b = roofline_boundaries(peak)
    # AI 范围 0.1 ~ 1e4
    ai = np.logspace(-1, 4, 400)
    bw_gbs = b["bw_gbs"]
    # 带宽斜线: FLOP/s = AI * bytes/s
    bw_line = ai * (bw_gbs * 1e9) / 1e12  # TFLOPS
    ax.plot(ai, bw_line, "k--", lw=1.2, label=f"HBM BW {bw_gbs:.0f} GB/s (实测)")

    colors = {"float32": "tab:blue", "float16": "tab:orange", "bfloat16": "tab:green"}
    for dk, tf in b["compute_tflops"].items():
        ax.axhline(tf, color=colors.get(dk, "gray"), ls=":", lw=1.0,
                   label=f"{dk} GEMM peak {tf:.1f} TFLOPS (实测)")

    # 画点
    for p in roofline.points:
        if p.bytes <= 0 or p.duration_us <= 0:
            continue
        ai_p = p.arithmetic_intensity
        tflops_p = p.achieved_flops_per_s / 1e12
        ax.scatter([ai_p], [tflops_p], s=30, alpha=0.7, zorder=3)
    # 标注最重的点
    dom = roofline.dominant_point()
    if dom and dom.bytes > 0 and dom.duration_us > 0:
        ax.scatter([dom.arithmetic_intensity], [dom.achieved_flops_per_s / 1e12],
                   s=120, marker="*", color="red", zorder=4,
                   label=f"dominant: {dom.name[:40]}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1e-2, 200)
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title(f"Roofline - {getattr(roofline,'_name','')}\nA10G measured peaks (not marketing)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_pie(breakdown: BreakdownReport, out: Path):
    cats = [c for c in breakdown.category_us if breakdown.category_us[c] > 0]
    vals = [breakdown.category_us[c] for c in cats]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    color_map = {
        "cudnn/cublas": "#4C72B0", "triton-fused": "#55A868",
        "elementwise/reduce": "#C44E52", "memcpy/transfer": "#8172B2",
        "sync/idle": "#CCB974", "other": "#999999",
    }
    colors = [color_map.get(c, "#999") for c in cats]
    wedges, texts, autotexts = ax.pie(
        vals, labels=cats, autopct="%1.1f%%", colors=colors, startangle=90,
        pctdistance=0.8, textprops={"fontsize": 9})
    ax.set_title(f"Kernel time breakdown\n"
                 f"glue={breakdown.glue_pct:.1f}% (crit-b <=5%) | "
                 f"transfer+idle={breakdown.transfer_idle_pct:.1f}% (crit-c <3%)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_util(u: UtilizationReport, out: Path):
    labels = ["SM occupancy", "带宽利用率", "计算利用率"]
    vals = []
    for v in [u.sm_occupancy.value if u.sm_occupancy else None,
              u.bandwidth_util_pct, u.compute_util_pct]:
        vals.append(v if (v is not None and not math.isnan(v)) else 0.0)
    src = u.sm_occupancy.source if u.sm_occupancy else "n/a"
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.axhline(85, color="red", ls="--", lw=1, label="判据a 阈值 85%")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Utilization %")
    ax.set_title(f"HW utilization triple\nSM occ source={src}")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
