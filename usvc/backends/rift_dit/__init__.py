"""rift_dit backend: DiT + Rectified Flow (DSP-002 §3.1).

Feature spec distilled from RIFT-SVC defaults. The actual DiT model + RF sampler
is the next implementation phase (P1); this module first fixes the feature
contract so preprocessing can be shared with ddsp6/diffusion.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from .._defaults import MEL_44K, UNITS_CV_TTA2X, build_default_spec, hubertsoft_units
from ..base import Backend

__all__ = ["RIFTDiTBackend"]


def _default_spec() -> FeatureSpec:
    return build_default_spec(mel=MEL_44K, uv_policy="raw", rms_win=2048, units=UNITS_CV_TTA2X)


class RIFTDiTBackend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = RIFTDiTBackend.DEFAULT_FEATURES
        # Allow user to swap the units encoder (e.g. to hubertsoft) via train_cfg.
        units_kind = train_cfg.get("units_kind")
        units = base.units
        if units_kind == "hubertsoft":
            units = hubertsoft_units(train_cfg.get("units_ckpt"))
        # Allow f0 extractor override (rmvpe default, fcpe faster).
        f0 = resolve_f0_config(train_cfg)
        return FeatureSpec(mel=base.mel, f0=f0, rms=base.rms, units=units)

    def build_dataset(
        self, dataset_dir: str, split: str, feature_hashes: dict[str, str]
    ):
        from .dataset import RIFTDataset

        # Determine encoder_kind from feature spec (default contentvec_tta2x)
        spec = self.feature_spec({})
        encoder_kind = spec.units.params.get("kind", "contentvec_tta2x")
        return RIFTDataset(
            dataset_dir=dataset_dir,
            split=split,
            feature_hashes=feature_hashes,
            max_frames=256,
            augment=(split == "train"),
            encoder_kind=encoder_kind,
        )

    def build_module(self, train_cfg):
        from .module import RIFTModule

        return RIFTModule(train_cfg)
