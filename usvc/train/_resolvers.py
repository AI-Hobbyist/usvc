"""Shared train-CLI helpers (W1 dedup).

Feature-hash resolution + val-interval mapping were duplicated verbatim in
usvc/train/cli.py and usvc/train/mlx/cli.py; both now import from here.
"""
from __future__ import annotations

from ..data.feature_config import FeatureConfig, FeatureSpec

__all__ = ["resolve_feature_hashes", "resolve_val_interval"]


def resolve_val_interval(val_interval: float | int, n_batches: int) -> int | float:
    """Map cfg.val_interval to Lightning's val_check_interval.

    Step-based training uses check_val_every_n_epoch=None, which REQUIRES an
    integer val_check_interval (Lightning raises otherwise). So for >1
    (step-count mode) we always return an int:
      - val_interval <= n_batches: validate every val_interval steps (int).
      - val_interval >  n_batches: clamp to n_batches (validate once per epoch),
        avoiding "interval > num_batches" on small datasets.
    <=1 is epoch-fraction mode (epoch-based training), returned as float.
    """
    if val_interval <= 1:
        return max(float(val_interval), 1e-6)
    # Step-count mode: Lightning needs an int here (check_val_every_n_epoch=None).
    # val_check_interval may exceed batches/epoch — Lightning accumulates across
    # epochs, so 2000 validates every 2000 steps even if an epoch is only 32 steps.
    return int(val_interval)


def _match_hash(kind: str, fc: FeatureConfig | None, entries: list) -> str:
    """Return the meta entry hash whose config hash matches fc, falling
    back to entries[0] if no match (e.g. config-stripped meta)."""
    if fc is None or not entries:
        return entries[0]["hash"] if entries else ""
    want = fc.hash()
    for e in entries:
        if e.get("hash") == want:
            return e["hash"]
    # Fallback: compare config dicts for best-effort match
    for e in entries:
        if e.get("config", {}).get("n_mels") == fc.params.get("n_mels"):
            return e["hash"]
    return entries[0]["hash"]


def resolve_feature_hashes(
    meta, want_spec: FeatureSpec, units_kind: str | None
) -> dict[str, str]:
    """Build flat {kind: hash} from meta.features for dataset construction.

    meta.features structure: {"mel":[{hash,...}], "f0":[...], "units":{enc:[{hash,...}]}}.
    Each FeatureConfig's stable hash picks the correct entry — handles the
    case where one dataset dir holds multiple mel hashes (e.g. sovits 80-bin
    mel AND reflow_shallow 128-bin mel coexist).
    """
    feature_hashes: dict[str, str] = {}
    for kind, entries in meta.features.items():
        if kind == "units":
            # nested: {encoder_kind: [{hash,...}]}
            enc_map = entries or {}
            if units_kind is not None and units_kind in enc_map:
                feature_hashes["units"] = _match_hash(
                    kind, want_spec.units, enc_map[units_kind]
                )
            else:
                for enc_kind, enc_entries in enc_map.items():
                    if enc_entries:
                        feature_hashes["units"] = enc_entries[0]["hash"]
                        break
        elif isinstance(entries, list) and entries:
            fc = getattr(want_spec, kind, None)  # mel / f0 / rms / spec
            feature_hashes[kind] = _match_hash(kind, fc, entries)
    return feature_hashes