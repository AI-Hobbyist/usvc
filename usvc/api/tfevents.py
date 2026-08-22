"""tfevents scalar reader (training curves): Lightning writes
``exp_dir/lightning_logs/version_N/events.out.tfevents.*``; incremental Reload + downsample for
the frontend poller (WEBUI_DESIGN decision 4: no embedded tensorboard process)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MAX_POINTS_PER_TAG = 2000  # canvas budget; beyond this, equidistant downsample (keep ends)
CACHE_SECONDS = 5.0


def _downsample(pts, limit):
    if len(pts) <= limit: return pts
    idx = [round(i * (len(pts) - 1) / (limit - 1)) for i in range(limit)]
    return [pts[i] for i in idx]


class TfEventsStore:  # EventAccumulator per exp_dir, incremental Reload + cached output
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._acc: dict[str, EventAccumulator] = {}
        self._cache: dict[str, tuple[float, dict]] = {}

    def scalars(self, exp_dir) -> dict:  # {tags: {tag: [[step, value], ...]}, version, updated_at}
        with self._lock:
            now = time.time()
            cached = self._cache.get(exp_dir)
            if cached is not None and now - cached[0] < CACHE_SECONDS: return cached[1]
            data = self._read(exp_dir)
            self._cache[exp_dir] = (now, data)
            return data

    def _read(self, exp_dir) -> dict:
        empty = {"tags": {}, "version": None, "updated_at": None}
        logs = Path(exp_dir) / "lightning_logs"
        if not logs.is_dir(): return empty
        versions = sorted((p for p in logs.iterdir() if p.name.startswith("version_") and p.is_dir()),
                          key=lambda p: p.stat().st_mtime)
        if not versions: return empty
        version = str(versions[-1])  # newest run
        acc = self._acc.setdefault(version, EventAccumulator(version, size_guidance={"scalars": 0}))
        acc.Reload()
        tags = {}
        for tag in acc.Tags().get("scalars", []):
            tags[tag] = _downsample([[int(e.step), float(e.value)] for e in acc.Scalars(tag)],
                                    MAX_POINTS_PER_TAG)
        return {"tags": tags, "version": Path(version).name,
                "updated_at": max((f.stat().st_mtime for f in Path(version).iterdir()), default=None)}

    def invalidate(self, exp_dir) -> None:  # optional post-job cache drop
        with self._lock:
            self._cache.pop(exp_dir, None)
