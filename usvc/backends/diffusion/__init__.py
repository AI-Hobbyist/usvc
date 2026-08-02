"""diffusion backend feature spec (DSP-002 §3.3).

Distilled from Diffusion-SVC defaults. Same mel config as rift_dit/ddsp6; units
default to plain contentvec (no tta2x). spk_emb is an experimental optional
feature (GE2E), not in the core spec — diffusion uses spk_id like everyone else.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from ..base import Backend

__all__ = ["DiffusionBackend"]


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
        rms=FeatureConfig("rms", {"win_length": 2048}),
        units=FeatureConfig(
            "units",
            {
                "kind": "contentvec",
                "layer": 12,
                "sr_in": 16000,
                "hop_native": 320,
                "align_mode": "gather",
                "model_ckpt": "pretrained/contentvec/pytorch_model.bin",
            },
        ),
    )


class DiffusionBackend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = DiffusionBackend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg)
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
        return FeatureSpec(mel=base.mel, f0=f0, rms=base.rms, units=units)

    def build_dataset(self, dataset_dir: str, split: str, feature_hashes: dict[str, str]):
        raise NotImplementedError("diffusion dataset wiring is P2")
