"""usvc-index / usvc-cluster CLI:
    usvc-index build <dataset_dir> [--units-kind K] [--global] [--device D]
    usvc-cluster build <dataset_dir> [--units-kind K] [--device D] [--n-clusters N]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..data.meta import load_meta
from .build_cluster import ClusterTrainer
from .build_index import IndexBuilder


def _resolve_units(dataset_dir, units_kind, units_hash) -> tuple[str, str]:
    """Pick (kind, hash): explicit hash wins; else newest hash of --units-kind; else the kind
    with the most registered vectors."""
    units = load_meta(dataset_dir).features.get("units", {})

    if units_hash is not None:
        if units_kind is None:
            for kind, entries in units.items():
                if any(e.get("hash") == units_hash for e in entries):
                    units_kind = kind; break
            if units_kind is None:  # discover from artifact layout on disk
                for kind_dir in (Path(dataset_dir) / "units").iterdir():
                    if (kind_dir / units_hash).is_dir(): units_kind = kind_dir.name; break
            if units_kind is None:
                raise SystemExit(f"cannot resolve units kind for hash {units_hash}")
        return units_kind, units_hash

    if not units:
        raise SystemExit(f"no units features registered in {dataset_dir}/meta.json — "
                         f"preprocess first, or pass --units-hash")
    if units_kind is not None:
        if units_kind not in units:
            raise SystemExit(f"units kind '{units_kind}' not registered. Available: {sorted(units)}")
        if not units[units_kind]:
            raise SystemExit(f"units kind '{units_kind}' has no registered hashes")
    else:  # auto: kind with the most vectors
        units_kind = max(units, key=lambda k: sum(e.get("count", 0) for e in units[k]))
    return units_kind, max(units[units_kind], key=lambda e: e.get("count", 0))["hash"]


def _parse_build(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build retrieval index / cluster model")
    b.add_argument("dataset_dir")
    b.add_argument("--units-kind", default=None, help="units encoder kind (auto-pick if omitted)")
    b.add_argument("--units-hash", default=None, help="units config hash (bypasses meta.json lookup)")
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
        b = IndexBuilder(args.dataset_dir, kind, h,
                         n_clusters_downsample=args.n_clusters, downsample_threshold=args.threshold)
        for p in b.build(device=args.device, per_speaker=not args.global_index):
            print(f"index: {p}")
    else:
        b = ClusterTrainer(args.dataset_dir, kind, h, n_clusters=args.n_clusters)
        print(f"cluster: {b.build(device=args.device)}")


def main_index(argv=None) -> int:
    args = _parse_build("usvc-index").parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr); return 1
    return 0


def main_cluster(argv=None) -> int:
    args = _parse_build("usvc-cluster").parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    # entry points call main_index/main_cluster directly; -m routes by argv[0]
    import os
    prog = os.path.basename(sys.argv[0])
    sys.exit(main_cluster() if prog.startswith("usvc-cluster") else main_index())
