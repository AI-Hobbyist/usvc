"""Audio artifact registry (WEBUI_DESIGN §0.10): big files travel as path strings; the backend
precomputes waveform peaks so the frontend never reads raw audio."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


class AudioError(Exception):
    """Audio registration failure (missing file / unsupported type)."""


class AudioStore:  # id -> path + meta; peaks results cached per id
    def __init__(self) -> None:
        self._items: dict[str, dict] = {}
        self._peaks_cache: dict[str, dict] = {}

    def register(self, path, *, job_id=None, kind="generic") -> dict:
        p = Path(path).resolve()
        if not p.exists(): raise AudioError(f"file not found: {p}")
        if p.suffix.lower() not in AUDIO_EXTS:
            raise AudioError(f"unsupported audio type: {p.suffix} (allowed: {sorted(AUDIO_EXTS)})")
        for item in self._items.values():
            if item["path"] == str(p): return item  # idempotent per resolved path
        item = {"id": uuid.uuid4().hex[:12], "path": str(p), "name": p.name,
                "size": p.stat().st_size, "job_id": job_id, "kind": kind, "created_at": time.time()}
        self._items[item["id"]] = item
        return item

    def list(self) -> list[dict]: return sorted(self._items.values(), key=lambda i: i["created_at"], reverse=True)

    def get(self, audio_id) -> dict | None: return self._items.get(audio_id)

    def peaks(self, audio_id, n=800) -> dict | None:
        item = self.get(audio_id)
        if item is None: return None
        cached = self._peaks_cache.get(audio_id)
        if cached is not None and cached["n"] == n: return cached
        result = _compute_peaks(item["path"], n)
        self._peaks_cache[audio_id] = result
        return result


def _compute_peaks(path, n) -> dict:  # n groups of [min, max] for direct canvas drawing
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    total = len(mono)
    n = max(1, min(n, total))
    idx = np.linspace(0, total, n + 1).astype(int)
    segs = (mono[idx[i]:idx[i + 1]] for i in range(n))
    peaks = [[float(s.min()), float(s.max())] if len(s) else [0.0, 0.0] for s in segs]
    return {"n": n, "peaks": peaks, "duration": total / sr,
            "sample_rate": sr, "channels": data.shape[1]}
