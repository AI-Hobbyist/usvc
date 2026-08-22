"""Feature config & artifact-hash (DSP-001 §5, DSP-002 §1): identical configs -> identical
hash -> one shared artifact across backends. Only params that change tensor values belong here;
runtime knobs (device/workers/paths) must stay out or the hash turns machine-specific.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

_SLOTS = ("mel", "f0", "rms", "spec", "units", "aug_shift", "aug_mel")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class FeatureConfig:
    kind: str  # mel | f0 | rms | spec | units | aug_shift | aug_mel
    params: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        blob = {"kind": self.kind, **self.params}
        # nvSTFT center=False grid version tag: bumping invalidates stale .pt artifacts
        # from the old symmetric n_fft//2 padding (mel/spec/rms/aug_mel share the grid).
        if self.kind in ("mel", "spec", "rms", "aug_mel"): blob["_grid"] = "nvstft_v2"
        return hashlib.sha256(_stable_json(blob).encode()).hexdigest()[:16]

    @property
    def path_tag(self) -> str:  # units bucketed per encoder kind so different models never share a dir
        return f"{self.kind}/{self.hash()}" if self.kind != "units" \
            else f"units/{self.params.get('kind', 'unknown')}/{self.hash()}"

    def __repr__(self) -> str:
        return f"FeatureConfig({self.kind}@{self.hash()} {self.params})"


@dataclass(frozen=True)
class FeatureSpec:
    """A backend's feature requirements (any subset of _SLOTS)."""

    mel: FeatureConfig | None = None
    f0: FeatureConfig | None = None
    rms: FeatureConfig | None = None
    spec: FeatureConfig | None = None
    units: FeatureConfig | None = None
    aug_shift: FeatureConfig | None = None
    aug_mel: FeatureConfig | None = None

    def _named(self):
        return [(n, getattr(self, n)) for n in _SLOTS]

    def all_required(self) -> list[FeatureConfig]:
        # slot order matters: run() computes aug_shift before aug_mel reads it
        return [c for _, c in self._named() if c is not None]

    def as_hash_map(self) -> dict[str, str]:  # kind -> hash, for dataset construction
        return {n: c.hash() for n, c in self._named() if c is not None}
