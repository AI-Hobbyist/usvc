"""Dataset metadata: meta.json read/write + integrity checks (DSP-001 §6).

meta.json is the single registry of which feature hashes have been computed and
how many files. PreprocessEngine consults it to skip already-computed features;
backends consult it to validate required hashes exist before training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .feature_config import FeatureConfig

__all__ = ["DatasetMeta", "AudioEntry", "init_dataset_meta", "load_meta", "save_meta"]

SCHEMA_VERSION = 1


@dataclass
class AudioEntry:
    speaker: str
    file_name: str
    frame_len: int
    duration: float
    audio_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "file_name": self.file_name,
            "frame_len": self.frame_len,
            "duration": self.duration,
            "audio_samples": self.audio_samples,
        }


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
    # features: kind -> list of {hash, config, count}
    # units is nested: features["units"][encoder_kind] -> list[{hash,...}]
    features: dict[str, Any] = field(
        default_factory=lambda: {
            "mel": [],
            "f0": [],
            "rms": [],
            "spec": [],
            "units": {},
        }
    )

    @property
    def spk2idx(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.speakers)}

    # ---- hash registry -------------------------------------------------
    def is_hash_registered(self, fc: FeatureConfig, *, count_complete: bool = True) -> bool:
        bucket = self._bucket(fc)
        for entry in bucket:
            if entry["hash"] == fc.hash():
                if count_complete:
                    return entry.get("count", 0) >= len(self.audios)
                return True
        return False

    def register(self, fc: FeatureConfig, count: int) -> None:
        """Append-only registration of a freshly computed feature hash."""
        bucket = self._bucket(fc)
        h = fc.hash()
        # Update count if already present (re-run / force), else append.
        for entry in bucket:
            if entry["hash"] == h:
                entry["count"] = count
                return
        bucket.append({"hash": h, "config": {"kind": fc.kind, **fc.params}, "count": count})

    def _bucket(self, fc: FeatureConfig) -> list[dict[str, Any]]:
        if fc.kind == "units":
            enc = fc.params.get("kind", "unknown")
            nested = self.features.setdefault("units", {})
            return nested.setdefault(enc, [])
        return self.features.setdefault(fc.kind, [])

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sr": self.sr,
            "hop": self.hop,
            "f0_min": self.f0_min,
            "f0_max": self.f0_max,
            "speakers": self.speakers,
            "audios": [a.to_dict() for a in self.audios],
            "splits": self.splits,
            "features": self.features,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_meta(dataset_dir: str | Path) -> DatasetMeta:
    p = Path(dataset_dir) / "meta.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"meta.json schema_version mismatch: got {raw.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}. Re-preprocess the dataset."
        )
    m = DatasetMeta(
        path=p,
        sr=raw["sr"],
        hop=raw["hop"],
        f0_min=raw["f0_min"],
        f0_max=raw["f0_max"],
        speakers=list(raw.get("speakers", [])),
        audios=[AudioEntry(**a) for a in raw.get("audios", [])],
        splits=dict(raw.get("splits", {})),
        features=dict(raw.get("features", {})),
    )
    # backfill missing buckets for forward-compat
    for k in ("mel", "f0", "rms", "spec"):
        m.features.setdefault(k, [])
    m.features.setdefault("units", {})
    return m


def init_dataset_meta(
    dataset_dir: str | Path, *, sr: int, hop: int, f0_min: float, f0_max: float
) -> DatasetMeta:
    p = Path(dataset_dir) / "meta.json"
    return DatasetMeta(path=p, sr=sr, hop=hop, f0_min=f0_min, f0_max=f0_max)


def save_meta(meta: DatasetMeta) -> None:
    meta.save()
