"""Feature configuration & config-hash (DSP-001 §5, DSP-002 §1).

Every feature artifact is located by a stable config hash derived from its
FeatureConfig. Identical configs -> identical hash -> single artifact, computed
once and shared across backends.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["FeatureConfig", "FeatureSpec"]


def _stable_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no ensure_ascii surprises."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class FeatureConfig:
    """One feature requirement + the params that affect its numeric output.

    Only fields that change the resulting tensor values belong in `params`.
    Runtime-only knobs (device, num_workers, verbose, force, absolute paths)
    must NOT be here, or the hash becomes machine-specific.
    """

    kind: str  # "mel" | "f0" | "rms" | "spec" | "units"
    params: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        blob = {"kind": self.kind, **self.params}
        return hashlib.sha256(_stable_json(blob).encode()).hexdigest()[:16]

    @property
    def path_tag(self) -> str:
        """Sub-path segment for this feature under the dataset dir.

        For ``units`` the bucket is ``units/{encoder_kind}/{hash}`` so that
        physically different models never share a directory. Other kinds use
        ``{kind}/{hash}``.
        """
        if self.kind == "units":
            kind = self.params.get("kind", "unknown")
            return f"units/{kind}/{self.hash()}"
        return f"{self.kind}/{self.hash()}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FeatureConfig({self.kind}@{self.hash()} {self.params})"


@dataclass(frozen=True)
class FeatureSpec:
    """A backend's full feature requirements (any subset of the slots)."""

    mel: FeatureConfig | None = None
    f0: FeatureConfig | None = None
    rms: FeatureConfig | None = None
    spec: FeatureConfig | None = None
    units: FeatureConfig | None = None

    def all_required(self) -> list[FeatureConfig]:
        return [c for c in (self.mel, self.f0, self.rms, self.spec, self.units) if c is not None]

    def as_hash_map(self) -> dict[str, str]:
        """Map feature kind -> hash, for dataset construction."""
        out: dict[str, str] = {}
        for name, cfg in (
            ("mel", self.mel),
            ("f0", self.f0),
            ("rms", self.rms),
            ("spec", self.spec),
            ("units", self.units),
        ):
            if cfg is not None:
                out[name] = cfg.hash()
        return out
