"""diffusion backend feature spec (DSP-002 §3.3).

Distilled from Diffusion-SVC defaults. Same mel config as rift_dit/ddsp6; units
default to plain contentvec (no tta2x). spk_emb is an experimental optional
feature (GE2E), not in the core spec — diffusion uses spk_id like everyone else.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from .._defaults import MEL_44K, UNITS_CV, build_default_spec, hubertsoft_units
from ..base import Backend

__all__ = ["DiffusionBackend"]


def _default_spec() -> FeatureSpec:
    return build_default_spec(mel=MEL_44K, uv_policy="interp", rms_win=2048, units=UNITS_CV)


class DiffusionBackend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = DiffusionBackend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg, default_uv_policy="interp")
        # mel config may be overridden by train_cfg (so-vits shallow diffusion
        # uses 80 bins / fmin=0 / fmax=22050 while diffusion default is
        # 128 bins / fmin=40 / fmax=16000).
        mel = FeatureConfig(
            "mel",
            {
                "n_fft": 2048,
                "win_length": 2048,
                "n_mels": train_cfg.get("n_mels", 128),
                "fmin": train_cfg.get("fmin", 40.0),
                "fmax": train_cfg.get("fmax", 16000.0),
                "center": False,
            },
        )
        if train_cfg.get("units_kind") == "hubertsoft":
            units = hubertsoft_units(train_cfg.get("units_ckpt"))
        else:
            units = base.units
        spec = FeatureSpec(mel=mel, f0=f0, rms=base.rms, units=units)
        # Pitch augmentation (Diffusion-SVC default OFF; enable via config).
        # aug_mel must use the SAME mel config as ``mel`` (incl. n_mels/fmin/fmax).
        if train_cfg.get("use_pitch_aug"):
            spec = FeatureSpec(
                mel=mel,
                f0=f0,
                rms=base.rms,
                units=units,
                aug_shift=FeatureConfig("aug_shift", {"range": [-5.0, 5.0]}),
                aug_mel=FeatureConfig("aug_mel", dict(mel.params)),
            )
        return spec

    def build_dataset(
        self, dataset_dir: str, split: str, feature_hashes: dict[str, str]
    ):
        from .dataset import DiffusionDataset

        spec = self.feature_spec({})
        encoder_kind = spec.units.params.get("kind", "contentvec")
        return DiffusionDataset(
            dataset_dir=dataset_dir,
            split=split,
            feature_hashes=feature_hashes,
            max_frames=172,
            augment=(split == "train"),
            encoder_kind=encoder_kind,
        )

    def build_module(self, train_cfg):
        from .module import DiffusionModule

        return DiffusionModule(train_cfg)
