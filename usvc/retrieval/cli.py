"""CLI for retrieval-index and cluster building.

Commands:
    usvc-index build <dataset_dir> [--units-kind K] [--global] [--device D]
    usvc-cluster build <dataset_dir> [--units-kind K] [--device D] [--n-clusters N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..data.meta import load_meta
from .build_cluster import ClusterTrainer
from .build_index import IndexBuilder

__all__ = ["main_index", "main_cluster"]


def _resolve_units(dataset_dir: str, units_kind: str | None, units_hash: str | None) -> tuple[str, str]:
    """Pick (kind, hash) from meta.json's registered units features.

    If units_kind is given, pick the newest registered hash for that kind;
    otherwise pick the kind with the most registered vectors (best corpus).
    If units_hash is given explicitly, it wins (no meta lookup needed).
    """
    meta = load_meta(dataset_dir)
    units = meta.features.get("units", {})

    if units_hash is not None:
        if units_kind is None:
            # discover kind from existing artifact dirs
            for kind, entries in units.items():
                if any(e.get("hash") == units_hash for e in entries):
                    units_kind = kind
                    break
            if units_kind is None:
                # fall back to artifact layout on disk
                for kind_dir in (Path(dataset_dir) / "units").iterdir():
                    if (kind_dir / units_hash).is_dir():
                        units_kind = kind_dir.name
                        break
            if units_kind is None:
                raise SystemExit(f"cannot resolve units kind for hash {units_hash}")
        return units_kind, units_hash

    if not units:
        raise SystemExit(f"no units features registered in {dataset_dir}/meta.json — preprocess first, or pass --units-hash")

    if units_kind is not None:
        if units_kind not in units:
            raise SystemExit(f"units kind '{units_kind}' not registered. Available: {sorted(units)}")
        entries = units[units_kind]
        if not entries:
            raise SystemExit(f"units kind '{units_kind}' has no registered hashes")
        return units_kind, max(entries, key=lambda e: e.get("count", 0))["hash"]

    # Auto: kind with the most vectors.
    kind = max(units, key=lambda k: sum(e.get("count", 0) for e in units[k]))
    entries = units[kind]
    return kind, max(entries, key=lambda e: e.get("count", 0))["hash"]


def _parse_build(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build retrieval index / cluster model")
    b.add_argument("dataset_dir")
    b.add_argument("--units-kind", default=None, help="units encoder kind (auto-pick if omitted)")
    b.add_argument("--units-hash", default=None,
                   help="units config hash (bypasses meta.json lookup)")
    b.add_argument("--device", default="cuda")
    b.add_argument("--global", dest="global_index", action="store_true",
                   help="single global index (RVC style); default per-speaker (sovits style)")
    b.add_argument("--n-clusters", type=int, default=10000, dest="n_clusters",
                   help="cluster count / downsampling target (default 10000)")
    b.add_argument("--threshold", type=int, default=200000,
                   help="downsample index when vectors exceed this (default 200000)")
    b.set_defaults(func=lambda a: _run_build(a, prog))
    return p


def _run_build(args, prog: str) -> None:
    kind, h = _resolve_units(args.dataset_dir, args.units_kind, args.units_hash)
    print(f"units: kind={kind} hash={h}")
    if prog == "usvc-index":
        b = IndexBuilder(
            args.dataset_dir, kind, h,
            n_clusters_downsample=args.n_clusters, downsample_threshold=args.threshold,
        )
        out = b.build(device=args.device, per_speaker=not args.global_index)
        for p in out:
            print(f"index: {p}")
    else:
        b = ClusterTrainer(args.dataset_dir, kind, h, n_clusters=args.n_clusters)
        print(f"cluster: {b.build(device=args.device)}")


def main_index(argv: list[str] | None = None) -> int:
    p = _parse_build("usvc-index")
    args = p.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def main_cluster(argv: list[str] | None = None) -> int:
    p = _parse_build("usvc-cluster")
    args = p.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import os

    # Installed entry points (usvc-index / usvc-cluster) call main_index /
    # main_cluster directly and never reach here. `-m` execution routes by
    # argv[0] so debugging `python -m usvc.retrieval.cli ...` picks correctly.
    prog = os.path.basename(sys.argv[0])
    sys.exit(main_cluster() if prog.startswith("usvc-cluster") else main_index())
