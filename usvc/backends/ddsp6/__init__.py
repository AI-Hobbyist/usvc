"""ddsp6 backend feature spec (DSP-002 §3.2).

Distilled from DDSP-SVC-6.3 defaults (velocity_net LYNXNet2). Version-aware:
  - version 6.1: units default to contentvec (non-tta2x, hop 320),
    velocity_net LYNXNet, dims per 6.1 configs/reflow.yaml
  - version 6.2: units default to contentvec_tta2x, velocity_net LYNXNet
  - version 6.3: units default to contentvec_tta2x, velocity_net LYNXNet2
Same mel/f0/rms as rift_dit; user can switch to hubertsoft.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from ..base import Backend

__all__ = ["DDSP6Backend"]


def _default_spec() -> FeatureSpec:
    return FeatureSpec(
        mel=FeatureConfig(
            "mel",
            {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False},
        ),
        f0=FeatureConfig(
            "f0",
            {"extractor": "rmvpe", "uv_policy": "raw", "model_ckpt": "pretrained/rmvpe/model.pt"},
        ),
        rms=FeatureConfig("rms", {"win_length": 1024}),  # DDSP6 volume_smooth_size
        units=FeatureConfig(
            "units",
            {
                "kind": "contentvec_tta2x",
                "layer": 12,
                "tta_2x": True,
                "sr_in": 16000,
                "hop_native": 160,
                "align_mode": "gather",
                "model_ckpt": "pretrained/contentvec/pytorch_model.bin",
            },
        ),
    )


class DDSP6Backend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = DDSP6Backend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg)
        version = str(train_cfg.get("version", "6.3"))
        units_kind = train_cfg.get("units_kind")
        # 6.1's default encoder is contentvec768l12 (no tta2x, hop 320).
        if version == "6.1" and units_kind is None:
            units_kind = "contentvec"
        if units_kind == "hubertsoft":
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
        elif units_kind == "contentvec":  # plain (no tta2x)
            units = FeatureConfig(
                "units",
                {
                    "kind": "contentvec",
                    "layer": 12,
                    "sr_in": 16000,
                    "hop_native": 320,
                    "align_mode": "gather",
                    "model_ckpt": "pretrained/contentvec/pytorch_model.bin",
                },
            )
        else:
            units = base.units
        return FeatureSpec(mel=base.mel, f0=f0, rms=base.rms, units=units)

    def build_dataset(
        self, dataset_dir: str, split: str, feature_hashes: dict[str, str]
    ):
        from .dataset import DDSPDataset

        spec = self.feature_spec({})
        encoder_kind = spec.units.params.get("kind", "contentvec_tta2x")
        return DDSPDataset(
            dataset_dir=dataset_dir,
            split=split,
            feature_hashes=feature_hashes,
            max_frames=172,
            augment=(split == "train"),
            encoder_kind=encoder_kind,
        )

    def build_module(self, train_cfg):
        from .module import DDSPModule

        return DDSPModule(train_cfg)
