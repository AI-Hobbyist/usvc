"""meta.json registry (DSP-001 §6): which feature hashes are computed and how many files.
Engine consults it to skip done work; backends validate required hashes before training."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class AudioEntry:
    speaker: str
    file_name: str
    frame_len: int
    duration: float
    audio_samples: int


@dataclass
class DatasetMeta:
    path: Path
    sr: int
    hop: int
    f0_min: float
    f0_max: float
    speakers: list[str] = field(default_factory=list)
    audios: list[AudioEntry] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)
    # kind -> [{hash, config, count}]; units nested: features["units"][encoder_kind] -> [...]
    features: dict[str, Any] = field(
        default_factory=lambda: {"mel": [], "f0": [], "rms": [], "spec": [], "units": {}}
    )

    @property
    def spk2idx(self): return {s: i for i, s in enumerate(self.speakers)}

    def _bucket(self, fc) -> list:
        if fc.kind == "units":
            return self.features.setdefault("units", {}).setdefault(fc.params.get("kind", "unknown"), [])
        return self.features.setdefault(fc.kind, [])

    def is_hash_registered(self, fc, *, count_complete=True) -> bool:
        h = fc.hash()
        return any(e["hash"] == h and (not count_complete or e.get("count", 0) >= len(self.audios))
                   for e in self._bucket(fc))

    def register(self, fc, count) -> None:  # append-only; re-run/force updates count in place
        h = fc.hash()
        for e in self._bucket(fc):
            if e["hash"] == h:
                e["count"] = count; return
        self._bucket(fc).append({"hash": h, "config": {"kind": fc.kind, **fc.params}, "count": count})

    def save(self) -> None:
        d = {"schema_version": SCHEMA_VERSION, "sr": self.sr, "hop": self.hop,
             "f0_min": self.f0_min, "f0_max": self.f0_max, "speakers": self.speakers,
             "audios": [asdict(a) for a in self.audios], "splits": self.splits, "features": self.features}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meta(dataset_dir) -> DatasetMeta:
    p = Path(dataset_dir) / "meta.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"meta.json schema_version mismatch: got {raw.get('schema_version')}, "
                         f"expected {SCHEMA_VERSION}. Re-preprocess the dataset.")
    m = DatasetMeta(path=p, sr=raw["sr"], hop=raw["hop"], f0_min=raw["f0_min"], f0_max=raw["f0_max"],
                    speakers=list(raw.get("speakers", [])),
                    audios=[AudioEntry(**a) for a in raw.get("audios", [])],
                    splits=dict(raw.get("splits", {})), features=dict(raw.get("features", {})))
    for k in ("mel", "f0", "rms", "spec"): m.features.setdefault(k, [])  # forward-compat backfill
    m.features.setdefault("units", {})
    return m


def init_dataset_meta(dataset_dir, *, sr, hop, f0_min, f0_max) -> DatasetMeta:
    return DatasetMeta(path=Path(dataset_dir) / "meta.json", sr=sr, hop=hop, f0_min=f0_min, f0_max=f0_max)


def save_meta(meta: DatasetMeta) -> None:
    meta.save()
