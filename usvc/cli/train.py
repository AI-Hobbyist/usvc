"""usvc-train CLI: unified training entry for all backends.

Wraps :func:`usvc.train.cli.run_training` with argparse so the
``usvc-train`` console-script entry point (pyproject.toml) actually resolves.

Usage:
    usvc-train --config configs/rift_dit.yaml
    usvc-train -c configs/sovits.yaml --resume experiments/sovits/last.ckpt
    usvc-train -c configs/rvc.yaml --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="usvc-train",
        description=(
            "Unified SVC training (all backends: rift_dit / ddsp6 / diffusion / "
            "sovits / rvc / reflow_vae / reflow_shallow)."
        ),
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="training config yaml (required; see configs/ for examples)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="checkpoint to resume from (.ckpt); overrides cfg.resume",
    )
    parser.add_argument(
        "--device", default=None,
        help="training device: cpu | cuda | cuda:N (default: auto)",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1

    try:
        from ..train.config import TrainConfig
        from ..train.cli import run_training
    except ImportError as exc:
        # Missing 'train' extras (lightning / schedulefree / tensorboard).
        print(
            f"error: training stack unavailable: {exc}\n"
            f"       install with: uv sync --extra train   (or pip install -e '.[train]')",
            file=sys.stderr,
        )
        return 1

    try:
        cfg = TrainConfig.from_yaml(cfg_path)
    except KeyError as exc:
        print(f"error: {cfg_path} is missing required key {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # yaml.YAMLError, OSError, ...
        print(f"error: cannot load config {cfg_path}: {exc}", file=sys.stderr)
        return 1

    resume = args.resume or getattr(cfg, "resume", None)
    if args.resume and not Path(args.resume).is_file():
        print(f"error: resume checkpoint not found: {args.resume}", file=sys.stderr)
        return 1

    try:
        run_training(cfg, resume=resume, device=args.device)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
