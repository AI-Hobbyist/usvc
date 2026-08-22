"""跨 torch/mlx 共享的后端配置默认表(was 各 module 内嵌副本) + W1 参数化 feature-spec builder。

只放纯数据表 + 纯函数:不 import 任何 backend 模块/子包(防循环);唯一 usvc
依赖是 data.feature_config(stdlib-only,无反向依赖)。表中的 _* 目标是:改一处
配置,torch 与 mlx 训练/推理路径同步生效。
"""
from __future__ import annotations

from ..data.feature_config import FeatureConfig, FeatureSpec

DDSP6_VERSION_DEFAULTS = {
    "6.1": dict(velocity_net="LYNXNet", n_aux_layers=3, n_aux_chans=256,
                n_layers=6, n_chans=512, t_start=0.7, use_norm=False),
    "6.2": dict(velocity_net="LYNXNet", n_aux_layers=6, n_aux_chans=512,
                n_layers=6, n_chans=1024, t_start=0.0, use_norm=True),
    "6.3": dict(velocity_net="LYNXNet2", n_aux_layers=6, n_aux_chans=512,
                n_layers=6, n_chans=1024, t_start=0.0, use_norm=True),
}


def resolve_ddsp6_opts(opts: dict) -> dict:
    """Version defaults overlaid by explicit backend_opts.

    Returns a dict containing the version defaults PLUS every other key present
    in ``opts`` (n_units / num_speaker / out_dims / sampling_rate / hop_length
    / etc.), so downstream ``opts.get(...)`` calls can actually read user
    overrides. Previously only keys already in the version-defaults dict
    survived, which silently dropped multi-speaker / units-dim overrides.
    """
    version = str(opts.get("version", "6.3"))
    if version not in DDSP6_VERSION_DEFAULTS:
        raise ValueError(
            f"ddsp6 version must be one of {list(DDSP6_VERSION_DEFAULTS)}, got {version!r}"
        )
    resolved = dict(DDSP6_VERSION_DEFAULTS[version])
    # Merge ALL user opts on top of the version defaults (user wins). This keeps
    # the version-specific knobs (velocity_net / n_layers / ...) overridable while
    # also passing through architecture keys not listed in DDSP6_VERSION_DEFAULTS.
    for k, v in opts.items():
        if k != "version":
            resolved[k] = v
    return resolved


# RVC generator upsample 默认表(镜像 torch RVCModule._RVC_UPSAMPLE)。
# 查表语义(ValueError 包装 / 直接 KeyError)由各调用方保留,这里只存数据。
RVC_UPSAMPLE = {
    ("v1", 32000): ([10, 4, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v1", 40000): ([10, 10, 2, 2], [16, 16, 4, 4]),
    ("v1", 48000): ([10, 6, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v2", 32000): ([10, 8, 2, 2], [20, 16, 4, 4]),
    ("v2", 48000): ([12, 10, 2, 2], [24, 20, 4, 4]),
}


# ---------------------------------------------------------------------------
# W1: parameterized feature-spec builder (6 families share one implementation;
# dicts are copied at construction so families never alias each other's params)
# ---------------------------------------------------------------------------

# mel grids: 44.1k/512 family (ddsp6/diffusion/reflow_shallow/rift_dit),
# RVC (48k/480), so-vits (80-bin, fmax=sr/2)
MEL_44K = {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False}
MEL_RVC = {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 0.0, "fmax": None, "center": False}
MEL_80BIN = {"n_fft": 2048, "win_length": 2048, "n_mels": 80, "fmin": 0.0, "fmax": 22050.0, "center": False}
# linear spectrogram (rvc/sovits only)
SPEC_LINEAR = {"n_fft": 2048, "win_length": 2048, "center": False}

# units encoders: contentvec (plain 320-hop) vs contentvec_tta2x (160-hop)
UNITS_CV = {
    "kind": "contentvec",
    "layer": 12,
    "sr_in": 16000,
    "hop_native": 320,
    "align_mode": "gather",
    "model_ckpt": "pretrained/contentvec/pytorch_model.bin",
}
UNITS_CV_TTA2X = {
    "kind": "contentvec_tta2x",
    "layer": 12,
    "tta_2x": True,
    "sr_in": 16000,
    "hop_native": 160,
    "align_mode": "gather",
    "model_ckpt": "pretrained/contentvec/pytorch_model.bin",
}


__all__ = [
    "DDSP6_VERSION_DEFAULTS",
    "RVC_UPSAMPLE",
    "MEL_44K",
    "MEL_RVC",
    "MEL_80BIN",
    "SPEC_LINEAR",
    "UNITS_CV",
    "UNITS_CV_TTA2X",
    "build_default_spec",
    "hubertsoft_units",
]


def build_default_spec(
    *,
    mel: dict,
    uv_policy: str,
    rms_win: int,
    units: dict,
    has_spec: bool = False,
) -> FeatureSpec:
    """Parameterized default FeatureSpec (f0 = fcpe + uv_policy).

    All params dicts are copied so each family's DEFAULT_FEATURES stays
    independent (byte-equal to the old per-family literals).
    """
    spec = FeatureSpec(
        mel=FeatureConfig("mel", dict(mel)),
        f0=FeatureConfig("f0", {"extractor": "fcpe", "uv_policy": uv_policy}),
        rms=FeatureConfig("rms", {"win_length": rms_win}),
        units=FeatureConfig("units", dict(units)),
    )
    if has_spec:
        spec = FeatureSpec(
            mel=spec.mel,
            spec=FeatureConfig("spec", dict(SPEC_LINEAR)),
            f0=spec.f0,
            rms=spec.rms,
            units=spec.units,
        )
    return spec


def hubertsoft_units(ckpt: str | None = None) -> FeatureConfig:
    """hubertsoft units swap (ddsp6/diffusion/rift_dit/sovits feature_spec).

    ``ckpt`` mirrors ``train_cfg.get("units_ckpt", <stock path>)``: families
    pass ``train_cfg.get("units_ckpt")``; so-vits passes nothing (fixed ckpt).
    """
    return FeatureConfig(
        "units",
        {
            "kind": "hubertsoft",
            "sr_in": 16000,
            "hop_native": 320,
            "align_mode": "gather",
            "model_ckpt": ckpt if ckpt is not None else "pretrained/hubert/hubert-soft-0d54a1f4.pt",
        },
    )