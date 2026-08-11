"""nsys NVTX 阶段解析。

通过 `nsys stats` 命令从 .nsys-rep 提取 NVTX range 数据,绕过 nsys 2024.2
sqlite 导出的 textId 关联 bug(字符串在 sqlite 里读不出,但 nsys stats 能正确显示)。

两个 report:
- nvtx_kern_sum: 每个 NVTX range 内的 kernel 汇总(按 stage 拆 kernel)
- nvtx_pushpop_trace: 每个 range 的 start/end/duration(时间轴瀑布)
"""
from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageKernel:
    """某个 stage 内的单个 kernel 聚合。"""
    name: str
    total_ns: int
    count: int
    avg_ns: float


@dataclass
class StageInfo:
    """一个推理阶段(extract/run_model/stitch)的统计。"""
    name: str
    instances: int          # 该 range 被触发次数(warmup + profile)
    total_ns: int           # 累计耗时
    kernels: list[StageKernel] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1e6


def parse_nvtx_from_rep(nsys_rep_path: str | Path) -> list[StageInfo]:
    """从 .nsys-rep 解析 NVTX 阶段。返回按 total_ns 降序的 stage 列表。

    需要 nsys 在 PATH。若 nsys 不可用或无 NVTX 数据,返回空列表(不报错,
    让上层决定是否降级)。
    """
    rep = Path(nsys_rep_path)
    if not rep.exists():
        return []

    nsys = _find_nsys()
    if nsys is None:
        return []

    # 1. pushpop_trace: 拿每个 stage 的总耗时和实例数
    trace_csv = _run_stats(nsys, rep, "nvtx_pushpop_trace")
    ranges = _parse_ranges(trace_csv)
    if not ranges:
        return []

    # 2. kern_sum: 拿每个 stage 内的 kernel 分解
    kern_csv = _run_stats(nsys, rep, "nvtx_kern_sum")
    stage_kernels = _parse_kern_sum(kern_csv)

    stages: list[StageInfo] = []
    for name, (instances, total_ns) in ranges.items():
        stages.append(StageInfo(
            name=name, instances=instances, total_ns=total_ns,
            kernels=stage_kernels.get(name, []),
        ))
    stages.sort(key=lambda s: s.total_ns, reverse=True)
    return stages


def _run_stats(nsys: str, rep: Path, report: str) -> str:
    """调 nsys stats 拿 CSV 输出。"""
    cmd = [nsys, "stats", "--report", report, "--format", "csv,column",
           "--force-export=true", str(rep)]
    # nsys stats 会先输出 "Generating SQLite..." / "Processing..." 到 stderr/stdout,
    # CSV 数据在后面。用 capture_output 这里是安全的(stats 命令不采集 GPU)。
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.stdout


def _parse_ranges(csv_text: str) -> dict[str, tuple[int, int]]:
    """从 pushpop_trace CSV 提取 {stage_name: (instances, total_ns)}。

    CSV 列: Start,End,Duration,DurChild,DurNonChild,Name,PID,TID,...
    """
    out: dict[str, list[int]] = {}  # name -> [count, total_ns]
    reader = _csv_rows(csv_text)
    for row in reader:
        if row is None:
            continue
        name = (row.get("Name") or "").strip()
        dur = _to_int(row.get("Duration (ns)"))
        if not name or dur is None:
            continue
        if name not in out:
            out[name] = [0, 0]
        out[name][0] += 1
        out[name][1] += dur
    return {k: (v[0], v[1]) for k, v in out.items()}


def _parse_kern_sum(csv_text: str) -> dict[str, list[StageKernel]]:
    """从 kern_sum CSV 提取 {stage_name: [StageKernel, ...]}。

    CSV 列: NVTX Range,Style,PID,TID,NVTX Inst,Kern Inst,Total Time (ns),
            Avg (ns),Med (ns),Min,Max,StdDev,Kernel Name
    """
    out: dict[str, list[StageKernel]] = {}
    reader = _csv_rows(csv_text)
    for row in reader:
        if row is None:
            continue
        stage = (row.get("NVTX Range") or "").strip()
        kname = (row.get("Kernel Name") or "").strip()
        total = _to_int(row.get("Total Time (ns)"))
        cnt = _to_int(row.get("Kern Inst"))
        avg = _to_float(row.get("Avg (ns)"))
        if not stage or not kname or total is None or cnt is None:
            continue
        out.setdefault(stage, []).append(StageKernel(
            name=kname, total_ns=total, count=cnt,
            avg_ns=avg or (total / cnt if cnt else 0),
        ))
    # 每个 stage 内按 total_ns 降序
    for v in out.values():
        v.sort(key=lambda k: k.total_ns, reverse=True)
    return out


def _csv_rows(text: str) -> csv.DictReader:
    """跳过 nsys stats 的非 CSV 前缀行(Generating/Processing),返回 DictReader。"""
    # nsys stats 把 log 打到 stdout 头部, CSV 在 "header,header\n" 开始。
    # 找到第一行(不含 nsys log 关键词)作为表头。
    lines = text.splitlines()
    start = 0
    log_kw = ("Generating", "Processing", "WARNING", "Executing", "Time (%)")
    for i, ln in enumerate(lines):
        # 表头行: 不含 log 关键词 且 含逗号
        if "," in ln and not any(ln.strip().startswith(k) for k in log_kw):
            start = i
            break
    body = "\n".join(lines[start:])
    return csv.DictReader(io.StringIO(body))


def _to_int(s) -> int | None:
    try:
        return int(float(s)) if s else None
    except (ValueError, TypeError):
        return None


def _to_float(s) -> float | None:
    try:
        return float(s) if s else None
    except (ValueError, TypeError):
        return None


def _find_nsys() -> str | None:
    for pat in ("/opt/nvidia/nsight-systems/2024.2.3/bin/nsys",):
        if Path(pat).exists():
            return pat
    from glob import glob
    for p in glob("/opt/nvidia/nsight-systems/*/bin/nsys"):
        return p
    from shutil import which
    return which("nsys")
