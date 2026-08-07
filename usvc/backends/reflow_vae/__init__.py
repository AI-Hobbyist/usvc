"""reflow_vae backend (ReFlow-VAE-SVC).

Feature spec matches the Diffusion-SVC family (mel 128-bin / 44.1k / hop 512,
contentvec units, rmvpe f0) — same data can be shared with the diffusion /
ddsp6 backends via config-hash.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from ..base import Backend
from ..diffusion import _default_spec as _diffusion_default_spec

__all__ = ["ReflowVAEBackend"]


class ReflowVAEBackend(Backend):
    DEFAULT_FEATURES = _diffusion_default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = ReflowVAEBackend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg)
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
            units = FeatureConfig(
                "units",
                {
                    "kind": "hubertsoft",
                    "sr_in": 16000,
                    "hop_native": 320,
                    "align_mode": "gather",
                    "model_ckpt": train_cfg.get("units_ckpt", "pretrained/hubert/hubert-soft-0d54a1f4.pt"),
                },
            )
        else:
            units = base.units
        spec = FeatureSpec(mel=mel, f0=f0, rms=base.rms, units=units)
        # Pitch augmentation (off by default; enable via config).
        # aug_mel must use the SAME mel config as ``mel``.
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
        from .dataset import ReflowVAEDataset

        spec = self.feature_spec({})
        encoder_kind = spec.units.params.get("kind", "contentvec")
        return ReflowVAEDataset(
            dataset_dir=dataset_dir,
            split=split,
            feature_hashes=feature_hashes,
            max_frames=172,
            augment=(split == "train"),
            encoder_kind=encoder_kind,
        )

    def build_module(self, train_cfg):
        from .module import ReflowVAEModule

        return ReflowVAEModule(train_cfg)
