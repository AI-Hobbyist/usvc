"""so-vits backend feature spec (DSP-002 §3.4).

so-vits uses a different mel config (80 mel, fmin=0, fmax=sr/2) and needs a
linear spectrogram (spec). Units default to hubertsoft (so-vits historical
default), switchable to contentvec.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from ..base import Backend

__all__ = ["SoVitsBackend"]


def _default_spec() -> FeatureSpec:
    return FeatureSpec(
        mel=FeatureConfig(
            "mel",
            # so-vits: 44.1k/512/2048 but 80 mel, fmin=0, fmax=22050 (sr/2)
            {"n_fft": 2048, "win_length": 2048, "n_mels": 80, "fmin": 0.0, "fmax": 22050.0, "center": False},
        ),
        spec=FeatureConfig("spec", {"n_fft": 2048, "win_length": 2048, "center": False}),
        f0=FeatureConfig(
            "f0",
            {"extractor": "fcpe", "uv_policy": "raw", "model_ckpt": "pretrained/rmvpe/model.pt"},
        ),
        rms=FeatureConfig("rms", {"win_length": 512}),
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


class SoVitsBackend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = SoVitsBackend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg)
        if train_cfg.get("units_kind") == "hubertsoft":
            # 4.0 兼容: hubertsoft (256 维)
            units = FeatureConfig(
                "units",
                {
                    "kind": "hubertsoft",
                    "sr_in": 16000,
                    "hop_native": 320,
                    "align_mode": "gather",
                    "model_ckpt": "pretrained/hubert/hubert-soft-0d54a1f4.pt",
                },
            )
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
