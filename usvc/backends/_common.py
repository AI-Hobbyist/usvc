"""Shared helpers for backend FEATURE_SPEC construction."""
from __future__ import annotations

from ..data.feature_config import FeatureConfig

__all__ = ["resolve_f0_config"]


def resolve_f0_config(
    train_cfg: dict, *, default_extractor: str = "fcpe", default_uv_policy: str = "raw"
) -> FeatureConfig:
    """Build an f0 FeatureConfig, allowing train_cfg to override the extractor.

    Supported extractors: ``fcpe`` (default, fast) and ``rmvpe``.
    uv_policy: "raw" (default) keeps f0==0 at unvoiced frames; "interp" fills
    them with linear interpolation (RVC/DDSP/Diffusion-SVC style). Backends
    pass their own default via ``default_uv_policy``; train_cfg can always
    override with an explicit ``uv_policy`` key.
    """
    extractor = train_cfg.get("f0_extractor", default_extractor)
    assert extractor in ("rmvpe", "fcpe"), f"unsupported f0_extractor: {extractor}"
    params = {
        "extractor": extractor,
        "uv_policy": train_cfg.get("uv_policy", default_uv_policy),
    }
    if extractor == "rmvpe":
        # RMVPE needs an external weight; FCPE ships its weight bundled.
        params["model_ckpt"] = train_cfg.get("rmvpe_ckpt", "pretrained/rmvpe/model.pt")
    return FeatureConfig("f0", params)
