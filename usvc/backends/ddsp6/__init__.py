"""ddsp6 backend feature spec (DSP-002 §3.2), from DDSP-SVC-6.3 defaults (LYNXNet2).

Version-aware: 6.1 units default to contentvec (non-tta2x, hop 320) + LYNXNet;
6.2 units default to contentvec_tta2x + LYNXNet; 6.3 -> LYNXNet2;
6.5 -> dual-timestep LYNXNet2 (PyTorch). Same
mel/f0/rms as rift_dit; user can switch to hubertsoft.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from .._defaults import MEL_44K, UNITS_CV, UNITS_CV_TTA2X, build_default_spec, hubertsoft_units
from ..base import Backend

__all__ = ["DDSP6Backend"]


def _default_spec() -> FeatureSpec:
    return build_default_spec(mel=MEL_44K, uv_policy="interp", rms_win=1024, units=UNITS_CV_TTA2X)


class DDSP6Backend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = DDSP6Backend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg, default_uv_policy="interp")
        version = str(train_cfg.get("version", "6.3"))
        units_kind = train_cfg.get("units_kind")
        if version == "6.1" and units_kind is None:  # 6.1's default encoder: contentvec768l12 (hop 320)
            units_kind = "contentvec"
        if units_kind == "hubertsoft":
            units = hubertsoft_units(train_cfg.get("units_ckpt"))
        elif units_kind == "contentvec":  # plain (no tta2x)
            units = FeatureConfig("units", dict(UNITS_CV))
        else:
            units = base.units
        spec = FeatureSpec(mel=base.mel, f0=f0, rms=base.rms, units=units)
        # Pitch augmentation (DDSP-SVC default ON): offline aug_mel + per-file
        # keyshift; dataset applies them 50% of the time during training.
        if train_cfg.get("use_pitch_aug"):
            spec = FeatureSpec(
                mel=base.mel, f0=f0, rms=base.rms, units=units,
                aug_shift=FeatureConfig("aug_shift", {"range": [-5.0, 5.0]}),
                aug_mel=FeatureConfig("aug_mel", dict(base.mel.params)),
            )
        return spec

    def build_dataset(
        self, dataset_dir: str, split: str, feature_hashes: dict[str, str]
    ):
        from .dataset import DDSPDataset

        spec = self.feature_spec({})
        encoder_kind = spec.units.params.get("kind", "contentvec_tta2x")
        return DDSPDataset(
            dataset_dir=dataset_dir, split=split, feature_hashes=feature_hashes,
            max_frames=172, augment=(split == "train"), encoder_kind=encoder_kind,
        )

    def build_module(self, train_cfg):
        from .module import DDSPModule

        return DDSPModule(train_cfg)
