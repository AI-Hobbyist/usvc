"""nsys sqlite 解析器。

从 nsys profile 产出的 .sqlite 提取结构化数据:
- GPU kernel 列表(名/耗时/实例数/寄存器/grid/block)
- memcpy 传输(真实 bytes)
- 硬件信息(smCount/memoryBandwidth —— roofline 峰值除数来源)

nsys 采集: nsys profile --trace=cuda,nvtx --export=sqlite -o <prefix> <cmd>
产出: <prefix>.sqlite (本模块解析) + <prefix>.nsys-rep (Perfetto UI 用)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class KernelRow:
    """单个 kernel 聚合后的统计(按 shortName 分组)。"""
    name: str               # demangled name
    total_ns: int           # 累计 GPU 时间(ns)
    count: int
    registers_per_thread: float  # 中位(同类 kernel 可能微变)
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    # 聚合后填(分类/反推)
    category: str = "other"
    flops: Optional[int] = None
    bytes_io: Optional[int] = None


@dataclass
class MemcpyStat:
    total_bytes: int
    total_ns: int
    count: int


@dataclass
class HardwareInfo:
    name: str
    sm_count: int
    memory_bandwidth_gbs: float   # GB/s
    clock_rate_hz: int
    l2_cache_bytes: int
    total_memory_bytes: int


@dataclass
class NsysReport:
    kernels: list[KernelRow]
    memcpy: MemcpyStat
    hardware: HardwareInfo
    total_gpu_ns: int             # 所有 kernel + memcpy 的 GPU 时间
    stages: list = None           # list[StageInfo], NVTX 阶段(可选, 需 nsys-rep)
    compute_ops: list = None      # list[ComputeOp], conv/gemm shape+FLOPs(可选, 需 shapes.json)

    @property
    def total_kernel_ns(self) -> int:
        return sum(k.total_ns for k in self.kernels)


def parse_sqlite(sqlite_path: str | Path,
                 nsys_rep_path: str | Path | None = None,
                 shapes_path: str | Path | None = None) -> NsysReport:
    """解析 nsys 产出的 sqlite,返回结构化报告。

    - nsys_rep_path: 提供→调用 nsys stats 解析 NVTX 阶段(绕过 sqlite textId bug)
    - shapes_path: 提供→加载 conv/gemm shape+FLOPs(ShapeCapture 产出)
    """
    p = Path(sqlite_path)
    if not p.exists():
        raise FileNotFoundError(f"nsys sqlite 不存在: {p}")
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        hardware = _parse_hardware(conn)
        memcpy = _parse_memcpy(conn)
        kernels = _parse_kernels(conn)
    finally:
        conn.close()

    total_gpu_ns = sum(k.total_ns for k in kernels) + memcpy.total_ns

    stages = None
    if nsys_rep_path is not None:
        try:
            from .nsys_nvtx import parse_nvtx_from_rep
            stages = parse_nvtx_from_rep(nsys_rep_path)
        except Exception:
            stages = None  # NVTX 解析失败不阻塞主报告

    compute_ops = None
    if shapes_path is not None and Path(shapes_path).exists():
        try:
            compute_ops = _load_compute_ops(shapes_path)
        except Exception:
            compute_ops = None

    return NsysReport(
        kernels=kernels, memcpy=memcpy, hardware=hardware,
        total_gpu_ns=total_gpu_ns, stages=stages, compute_ops=compute_ops,
    )


def _parse_hardware(conn: sqlite3.Connection) -> HardwareInfo:
    row = conn.execute(
        "SELECT name, smCount, memoryBandwidth, clockRate, l2CacheSize, totalMemory "
        "FROM TARGET_INFO_GPU LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("sqlite 无 TARGET_INFO_GPU 表(硬件信息缺失)")
    return HardwareInfo(
        name=row["name"],
        sm_count=row["smCount"],
        # memoryBandwidth 单位是 bytes/s (696096000000 = 696 GB/s)
        memory_bandwidth_gbs=row["memoryBandwidth"] / 1e9,
        clock_rate_hz=row["clockRate"],
        l2_cache_bytes=row["l2CacheSize"],
        total_memory_bytes=row["totalMemory"],
    )


def _parse_memcpy(conn: sqlite3.Connection) -> MemcpyStat:
    try:
        row = conn.execute(
            "SELECT count(*) AS n, COALESCE(sum(bytes),0) AS b, sum(end-start) AS t "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchone()
    except sqlite3.OperationalError:
        return MemcpyStat(0, 0, 0)
    if row is None:
        return MemcpyStat(0, 0, 0)
    return MemcpyStat(
        total_bytes=row["b"] or 0,
        total_ns=row["t"] or 0,
        count=row["n"] or 0,
    )


def _parse_kernels(conn: sqlite3.Connection) -> list[KernelRow]:
    """聚合 kernel:按 demangledName 分组,取中位寄存器/grid/block。"""
    rows = conn.execute(
        """
        SELECT s.value AS name,
               sum(k.end - k.start) AS total_ns,
               count(*) AS cnt,
               k.registersPerThread AS regs,
               k.gridX, k.gridY, k.gridZ,
               k.blockX, k.blockY, k.blockZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY k.demangledName
        ORDER BY total_ns DESC
        """
    ).fetchall()

    out: list[KernelRow] = []
    # 同名 kernel 的 regs/grid/block 可能微变(如 dynamic shape),取中位
    from statistics import median
    grouped: dict[str, list] = {}
    # 上面 SQL 已按 demangledName GROUP BY, regs 取的就是该组任意一行;
    # 为稳健,重新查每个 name 的 regs 中位。
    name_to_regs: dict[str, float] = {}
    for name in {r["name"] for r in rows}:
        regs_rows = conn.execute(
            "SELECT DISTINCT registersPerThread AS r FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName=s.id WHERE s.value=? ORDER BY r", (name,)
        ).fetchall()
        regs_vals = [rr["r"] for rr in regs_rows if rr["r"]]
        name_to_regs[name] = median(regs_vals) if regs_vals else 0.0

    for r in rows:
        out.append(KernelRow(
            name=r["name"],
            total_ns=r["total_ns"],
            count=r["cnt"],
            registers_per_thread=name_to_regs.get(r["name"], r["regs"] or 0),
            grid=(r["gridX"], r["gridY"], r["gridZ"]),
            block=(r["blockX"], r["blockY"], r["blockZ"]),
        ))
    return out


def _load_compute_ops(path: str | Path) -> list:
    """加载 shapes.json 为 ComputeOp 列表。"""
    import json
    from .shape_capture import ComputeOp
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for d in data:
        out.append(ComputeOp(
            op=d["op"],
            shapes=tuple(tuple(s) for s in d["shapes"]),
            count=d["count"],
            flops_per_call=d["flops_per_call"],
        ))
    out.sort(key=lambda o: o.total_flops, reverse=True)
    return out
