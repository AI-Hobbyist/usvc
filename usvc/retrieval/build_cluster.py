"""Cluster trainer: per-speaker KMeans over units features (so-vits train_cluster.py semantics;
pure torch, no sklearn/faiss). Centers map query features at inference (timbre-leakage control)."""
from __future__ import annotations

from pathlib import Path

import torch

from ._store import load_speaker, units_root
from .kmeans import KMeansGPU

SCHEMA_VERSION = 1


class ClusterTrainer:
    def __init__(self, dataset_dir, units_kind, units_hash,
                 n_clusters=10000, max_iter=100, tol=1e-2):
        self.dataset_dir = Path(dataset_dir)
        self.units_kind = units_kind
        self.units_hash = units_hash  # provenance guard against meta.json
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def out_path(self) -> Path:
        d = self.dataset_dir / "index"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"kmeans-{self.n_clusters}-{self.units_kind}-{self.units_hash[:8]}.pt"

    def build(self, device="cuda") -> Path:
        root = units_root(self.dataset_dir, self.units_kind, self.units_hash)
        device = device if torch.cuda.is_available() else "cpu"
        payload: dict = {
            "schema_version": SCHEMA_VERSION,
            "kind": "units_cluster",
            "n_clusters": self.n_clusters,
            "units": {"encoder_kind": self.units_kind, "config_hash": self.units_hash},
        }
        trained = 0
        for spk_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            vecs = load_speaker(spk_dir)
            if vecs.shape[0] < self.n_clusters:
                print(f"skip {spk_dir.name}: {vecs.shape[0]} vectors < n_clusters={self.n_clusters}")
                continue
            km = KMeansGPU(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol,
                           verbose=0, device=device)
            km.fit_predict(vecs.to(device))
            payload[spk_dir.name] = {"cluster_centers_": km.centroids.cpu().contiguous(),
                                     "n_features_in_": vecs.shape[1]}
            trained += 1
            torch.cuda.empty_cache()
        if trained == 0: raise RuntimeError("no speaker had enough vectors for clustering")
        out = self.out_path()
        torch.save(payload, out)
        print(f"saved cluster model: {out} ({trained} speakers)")
        return out
