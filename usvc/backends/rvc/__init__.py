"""rvc backend feature spec (DSP-002 §3.5).

RVC v2 (current) uses last_hidden_state (768-d, no proj); RVC v1 uses layer 9 +
final_proj (256-d). Default to v2. Note RVC's mel config is 48k/hop480, so RVC
typically needs its own dataset (sr/hop differ from the 44.1k/512 family).
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from ..base import Backend

__all__ = ["RVCBackend"]


def _default_spec() -> FeatureSpec:
    return FeatureSpec(
        mel=FeatureConfig(
            "mel",
            # RVC v2: 48k/480/2048/128mel/fmin0/fmax=None
            {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 0.0, "fmax": None, "center": False},
        ),
        spec=FeatureConfig("spec", {"n_fft": 2048, "win_length": 2048, "center": False}),
        f0=FeatureConfig(
            "f0",
            {"extractor": "rmvpe", "uv_policy": "interp", "model_ckpt": "pretrained/rmvpe/model.pt"},
        ),
        rms=FeatureConfig("rms", {"win_length": 2048}),
        units=FeatureConfig(
            "units",
            {
                "kind": "contentvec",  # RVC v2: last_hidden_state, no proj
                "layer": 12,
                "sr_in": 16000,
                "hop_native": 320,
                "align_mode": "gather",
                "model_ckpt": "pretrained/contentvec/pytorch_model.bin",
            },
        ),
    )


class RVCBackend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = RVCBackend.DEFAULT_FEATURES
        if train_cfg.get("rvc_version") == "v1":
            # RVC v1: layer 9 + final_proj -> 256-d
            return FeatureSpec(
                mel=base.mel,
                spec=base.spec,
                f0=base.f0,
                rms=base.rms,
                units=FeatureConfig(
                    "units",
                    {
                        "kind": "contentvec_proj",
                        "layer": 9,
                        "proj": "final_proj",
                        "sr_in": 16000,
                        "hop_native": 320,
                        "align_mode": "gather",
                        "model_ckpt": "pretrained/contentvec/pytorch_model.bin",
                    },
                ),
            )
        return base

    def build_dataset(
        self, dataset_dir: str, split: str, feature_hashes: dict[str, str]
    ):
        from .dataset import RVCDataset

        spec = self.feature_spec({})
        encoder_kind = spec.units.params.get("kind", "contentvec")
        return RVCDataset(
            dataset_dir=dataset_dir,
            split=split,
            feature_hashes=feature_hashes,
            max_frames=790,
            augment=(split == "train"),
            encoder_kind=encoder_kind,
        )

    def build_module(self, train_cfg):
        from .module import RVCModule

        return RVCModule(train_cfg)
