"""reflow_shallow backend feature spec.

A standalone reflow shallow-diffusion enhancer for sovits/rvc.  Unlike DDSP6,
this backend needs no units (the velocity network is conditioned on mel
alone) and no DDSP synthesiser — it learns (corrupted_mel → clean_mel)
refinement directly.

mel/f0/rms use the same 44.1k/512 config family as the other mel-domain
backends so that the PC NSF-HiFiGAN vocoder and sovits datasets are reused.
"""
from __future__ import annotations

from ...data.feature_config import FeatureConfig, FeatureSpec
from .._common import resolve_f0_config
from ..base import Backend

__all__ = ["ReflowShallowBackend"]


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
        rms=FeatureConfig("rms", {"win_length": 1024}),
        units=FeatureConfig(
            "units",
            {"kind": "contentvec", "layer": 12, "sr_in": 16000, "hop_native": 320, "align_mode": "gather", "model_ckpt": "pretrained/contentvec/pytorch_model.bin"},
        ),
    )


class ReflowShallowBackend(Backend):
    DEFAULT_FEATURES = _default_spec()

    @staticmethod
    def feature_spec(train_cfg: dict) -> FeatureSpec:
        base = ReflowShallowBackend.DEFAULT_FEATURES
        f0 = resolve_f0_config(train_cfg)
        return FeatureSpec(mel=base.mel, f0=f0, rms=base.rms, units=base.units)

    def build_dataset(
        self, dataset_dir: str, split: str, feature_hashes: dict[str, str]
    ):
        from .dataset import ReflowShallowDataset

        return ReflowShallowDataset(
            dataset_dir=dataset_dir,
            split=split,
            feature_hashes=feature_hashes,
            max_frames=128,
            augment=(split == "train"),
        )

    def build_module(self, train_cfg):
        from .module import ReflowShallowModule

        return ReflowShallowModule(train_cfg)
