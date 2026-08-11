"""tfevents 标量读取：训练曲线数据源。

Lightning 训练把标量写进 ``exp_dir/lightning_logs/version_N/events.out.tfevents.*``，
本模块用 tensorboard 的 EventAccumulator 增量解析（Reload 只读新增事件），
供前端轮询绘制训练曲线（docs/WEBUI_DESIGN.md 决策点 4：不嵌 tensorboard 进程）。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

__all__ = ["TfEventsStore"]

#: 每个 tag 输出的点数上限（前端 canvas 足够；超出等距降采样）
MAX_POINTS_PER_TAG = 2000
#: 结果缓存秒数（避免频繁全量序列化）
CACHE_SECONDS = 5.0


def _downsample(pts: list[list], limit: int) -> list[list]:
    """等距降采样（保首尾）。"""
    if len(pts) <= limit:
        return pts
    idx = [round(i * (len(pts) - 1) / (limit - 1)) for i in range(limit)]
    return [pts[i] for i in idx]


class TfEventsStore:
    """按 exp_dir 缓存 EventAccumulator，增量 Reload + 降采样输出。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._acc: dict[str, EventAccumulator] = {}
        self._cache: dict[str, tuple[float, dict]] = {}

    def scalars(self, exp_dir: str) -> dict:
        """返回 {tags: {tag: [[step, value], ...]}, version, updated_at}。"""
        with self._lock:
            now = time.time()
            cached = self._cache.get(exp_dir)
            if cached is not None and now - cached[0] < CACHE_SECONDS:
                return cached[1]
            data = self._read(exp_dir)
            self._cache[exp_dir] = (now, data)
            return data

    def _read(self, exp_dir: str) -> dict:
        logs = Path(exp_dir) / "lightning_logs"
        if not logs.is_dir():
            return {"tags": {}, "version": None, "updated_at": None}
        versions = sorted(
            (p for p in logs.iterdir() if p.name.startswith("version_") and p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if not versions:
            return {"tags": {}, "version": None, "updated_at": None}
        version = versions[-1]  # 最新 version 目录
        key = str(version)
        acc = self._acc.get(key)
        if acc is None:
            acc = EventAccumulator(key, size_guidance={"scalars": 0})
            self._acc[key] = acc
        acc.Reload()

        tags: dict[str, list[list]] = {}
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            pts = [[int(e.step), float(e.value)] for e in events]
            tags[tag] = _downsample(pts, MAX_POINTS_PER_TAG)
        updated_at = max((f.stat().st_mtime for f in version.iterdir()), default=None)
        return {"tags": tags, "version": version.name, "updated_at": updated_at}

    def invalidate(self, exp_dir: str) -> None:
        """训练 job 结束后清掉该 exp 的缓存（可选）。"""
        with self._lock:
            self._cache.pop(exp_dir, None)
