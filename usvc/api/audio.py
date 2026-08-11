"""音频产物注册表：wav 流式播放 + 波形峰值预算（前端不读原始音频文件）。

设计（docs/WEBUI_DESIGN.md §0.10）：大文件只传路径字符串，波形峰值由后端算好下发。
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

__all__ = ["AudioError", "AudioStore"]

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


class AudioError(Exception):
    """音频注册错误（文件不存在 / 类型不支持）。"""


class AudioStore:
    """音频文件注册表：id → 路径 + 元信息；峰值计算结果带缓存。"""

    def __init__(self) -> None:
        self._items: dict[str, dict] = {}
        self._peaks_cache: dict[str, dict] = {}

    def register(self, path: str | Path, *, job_id: str | None = None, kind: str = "generic") -> dict:
        p = Path(path).resolve()
        if not p.exists():
            raise AudioError(f"file not found: {p}")
        if p.suffix.lower() not in AUDIO_EXTS:
            raise AudioError(f"unsupported audio type: {p.suffix} (allowed: {sorted(AUDIO_EXTS)})")
        # 同一路径复用同一 id（幂等注册）
        for item in self._items.values():
            if item["path"] == str(p):
                return item
        item = {
            "id": uuid.uuid4().hex[:12],
            "path": str(p),
            "name": p.name,
            "size": p.stat().st_size,
            "job_id": job_id,
            "kind": kind,
            "created_at": time.time(),
        }
        self._items[item["id"]] = item
        return item

    def list(self) -> list[dict]:
        return sorted(self._items.values(), key=lambda i: i["created_at"], reverse=True)

    def get(self, audio_id: str) -> dict | None:
        return self._items.get(audio_id)

    def peaks(self, audio_id: str, n: int = 800) -> dict | None:
        item = self.get(audio_id)
        if item is None:
            return None
        cached = self._peaks_cache.get(audio_id)
        if cached is not None and cached["n"] == n:
            return cached
        result = _compute_peaks(item["path"], n)
        self._peaks_cache[audio_id] = result
        return result


def _compute_peaks(path: str, n: int) -> dict:
    """把波形压缩为 n 组 [min, max]，供前端 canvas 直接绘制。"""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    channels = data.shape[1]
    mono = data.mean(axis=1)
    total = len(mono)
    n = max(1, min(n, total))
    idx = np.linspace(0, total, n + 1).astype(int)
    peaks: list[list[float]] = []
    for i in range(n):
        seg = mono[idx[i]: idx[i + 1]]
        if len(seg) == 0:
            peaks.append([0.0, 0.0])
        else:
            peaks.append([float(seg.min()), float(seg.max())])
    return {
        "n": n,
        "peaks": peaks,
        "duration": total / sr,
        "sample_rate": sr,
        "channels": channels,
    }
