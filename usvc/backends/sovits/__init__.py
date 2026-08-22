"""so-vits backend feature spec (DSP-002 §3.4).

so-vits uses a different mel config (80 mel, fmin=0, fmax=sr/2) and needs a
linear spectrogram (spec). Units default to hubertsoft (so-vits historical
default), switchable to contentvec.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from .._defaults import MEL_80BIN, UNITS_CV, build_default_spec, hubertsoft_units
from ..base import Backend

__all__ = ["SoVitsBackend"]


def _default_spec() -> FeatureSpec:
    return build_default_spec(mel=MEL_80BIN, uv_policy="raw", rms_win=512, units=UNITS_CV, has_spec=True)


class SoVitsBackend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = SoVitsBackend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg)
        if train_cfg.get("units_kind") == "hubertsoft":
            # 4.0 兼容: hubertsoft (256 维),固定 ckpt
            units = hubertsoft_units()
        else:
            units = base.units  # 默认 contentvec (4.1 推荐)
        return FeatureSpec(mel=base.mel, spec=base.spec, f0=f0, rms=base.rms, units=units)

    def build_dataset(
        self, dataset_dir: str, split: str, feature_hashes: dict[str, str]
    ):
        from .dataset import SoVITSDataset

        spec = self.feature_spec({})
        encoder_kind = spec.units.params.get("kind", "contentvec")
        return SoVITSDataset(
            dataset_dir=dataset_dir,
            split=split,
            feature_hashes=feature_hashes,
            max_frames=790,
            augment=(split == "train"),
            encoder_kind=encoder_kind,
        )

    def build_module(self, train_cfg):
        from .module import SoVITSModule

        return SoVITSModule(train_cfg)
