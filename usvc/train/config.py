"""Training configuration.

TrainConfig is the single entry point for all training parameters.
Loaded from YAML, backend-agnostic (backend-specific opts in .backend_opts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["OptimizerConfig", "ScheduleConfig", "TrainConfig"]


@dataclass
class OptimizerConfig:
    type: str = "adamw"  # 'schedulefree' | 'muon_adamw' | 'adamw'
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    # muon-specific
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> OptimizerConfig:
        return cls(
            type=d.get("type", "adamw"),
            lr=d.get("lr", 1e-4),
            betas=tuple(d.get("betas", (0.9, 0.999))),
            eps=d.get("eps", 1e-8),
            weight_decay=d.get("weight_decay", 0.0),
            muon_momentum=d.get("muon_momentum", 0.95),
            muon_weight_decay=d.get("muon_weight_decay", 0.1),
        )


@dataclass
class ScheduleConfig:
    type: str | None = None  # None | 'step' | 'exponential' | 'linear_warmup_decay'
    warmup_steps: int = 0
    decay_step: int = 100000
    gamma: float = 0.5
    min_lr: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> ScheduleConfig:
        if d is None:
            return cls()
        return cls(
            type=d.get("type"),
            warmup_steps=d.get("warmup_steps", 0),
            decay_step=d.get("decay_step", 100000),
            gamma=d.get("gamma", 0.5),
            min_lr=d.get("min_lr", 0.0),
        )


@dataclass
class TrainConfig:
    backend: str = "rift_dit"
    dataset_dir: str = ""
    exp_dir: str = "experiments/default"
    val_dir: str | None = None  # separate val dataset; None = use train split 'val'

    # Data
    batch_size: int = 32
    max_frames: int = 256
    num_workers: int = 8
    augment: bool = True

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    grad_clip: float | None = 1.0

    # Precision
    precision: str = "bf16-mixed"

    # Training length
    max_epochs: int | None = None
    max_steps: int = 500_000
    val_interval: int = 2000
    save_interval: int = 2000

    # Resume
    resume: str | None = None
    pretrained: str | None = None

    # Backend-specific
    backend_opts: dict = field(default_factory=dict)

    # Feature hashes (resolved at runtime, not in yaml)
    feature_hashes: dict = field(default_factory=dict)

    # Throughput-optimal defaults per backend (empirically tuned on A10G 24GB).
    # flow/diffusion: AMP bf16 safe (<0.03% divergence). GAN: TF32-only + large batch.
    # Set apply_optimal_defaults=True in apply_optimal_throughput() to populate.
    _OPTIMAL_THROUGHPUT: dict = field(default_factory=lambda: {
        "rift_dit":  {"batch_size": 64, "precision": "bf16-mixed"},
        "ddsp6":    {"batch_size": 48, "precision": "bf16-mixed"},
        "diffusion": {"batch_size": 48, "precision": "bf16-mixed"},
        "sovits":   {"batch_size": 48, "precision": "bf16-mixed"},  # 完整GAN B=48, VRAM 10.5GB
        "rvc":      {"batch_size": 32, "precision": "bf16-mixed"},  # 完整GAN B=32, VRAM 10.4GB
    })

    def apply_optimal_throughput(self, override: bool = False):
        """Apply backend-optimal batch_size + precision for max throughput.

        Only fills unset values unless override=True. Safe (near-lossless):
        - flow/diffusion: TF32 + AMP bf16 (verified <0.03% loss divergence)
        - GAN (sovits/rvc): TF32 + larger batch (math unchanged, throughput up)
        """
        opt = self._OPTIMAL_THROUGHPUT.get(self.backend, {})
        if opt:
            if override or self.batch_size == 32:  # 32 is the dataclass default
                self.batch_size = opt["batch_size"]
            if override or self.precision == "bf16-mixed":
                self.precision = opt["precision"]

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            backend=d["backend"],
            dataset_dir=d["dataset_dir"],
            exp_dir=d.get("exp_dir", "experiments/default"),
            val_dir=d.get("val_dir"),
            batch_size=d.get("batch_size", 32),
            max_frames=d.get("max_frames", 256),
            num_workers=d.get("num_workers", 8),
            augment=d.get("augment", True),
            optimizer=OptimizerConfig.from_dict(d.get("optimizer", {})),
            schedule=ScheduleConfig.from_dict(d.get("schedule")),
            grad_clip=d.get("grad_clip", 1.0),
            precision=d.get("precision", "bf16-mixed"),
            max_epochs=d.get("max_epochs"),
            max_steps=d.get("max_steps", 500_000),
            val_interval=d.get("val_interval", 2000),
            save_interval=d.get("save_interval", 2000),
            resume=d.get("resume"),
            pretrained=d.get("pretrained"),
            backend_opts=d.get("backend_opts", {}),
        )
