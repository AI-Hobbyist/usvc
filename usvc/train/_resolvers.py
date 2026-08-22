"""Shared train-CLI helpers (also imported by usvc/train/mlx/cli.py): feature-hash resolution + val-interval mapping."""
from ..data.feature_config import FeatureConfig, FeatureSpec


def resolve_val_interval(val_interval, n_batches):
    # -> lightning val_check_interval. Step-based training (check_val_every_n_epoch=None) REQUIRES an int
    # val_check_interval, so >1 (step-count mode) always returns an int; <=1 is epoch-fraction mode (float)
    if val_interval <= 1:
        return max(float(val_interval), 1e-6)
    return int(val_interval)


def _match_hash(kind, fc: FeatureConfig | None, entries):
    """meta entry hash whose config hash matches fc; falls back to entries[0] when no match (e.g. config-stripped meta)."""
    if fc is None or not entries:
        return entries[0]["hash"] if entries else ""
    want = fc.hash()
    for e in entries:
        if e.get("hash") == want:
            return e["hash"]
    for e in entries:  # best-effort n_mels fallback
        if e.get("config", {}).get("n_mels") == fc.params.get("n_mels"):
            return e["hash"]
    return entries[0]["hash"]


def resolve_feature_hashes(meta, want_spec: FeatureSpec, units_kind=None):
    """Flat {kind: hash} from meta.features: {"mel":[{hash,...}], "f0":[...], "units":{enc:[{hash,...}]}}.

    Each FeatureConfig's stable hash picks the right entry — one dataset dir may hold several mel hashes
    (e.g. sovits 80-bin mel AND reflow_shallow 128-bin mel coexist).
    """
    feature_hashes = {}
    for kind, entries in meta.features.items():
        if kind == "units":
            enc_map = entries or {}
            if units_kind is not None and units_kind in enc_map:
                feature_hashes["units"] = _match_hash(kind, want_spec.units, enc_map[units_kind])
            else:
                for enc_kind, enc_entries in enc_map.items():
                    if enc_entries:
                        feature_hashes["units"] = enc_entries[0]["hash"]
                        break
        elif isinstance(entries, list) and entries:
            feature_hashes[kind] = _match_hash(kind, getattr(want_spec, kind, None), entries)
    return feature_hashes