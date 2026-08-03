"""Cluster trainer: per-speaker KMeans over units features (pure torch).

Replicates so-vits-svc ``cluster/train_cluster.py`` semantics: one KMeans model
per speaker over the units features, producing cluster centers used at
inference time to map query features onto the nearest center (timbre-leakage
control). Uses the pure-torch GPU KMeans (no sklearn/faiss).
"""
from __future__ import annotations

from pathlib import Path

import torch

from .kmeans import KMeansGPU

__all__ = ["ClusterTrainer", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1


class ClusterTrainer:
    def __init__(
        self,
        dataset_dir: str | Path,
        units_kind: str,
        units_hash: str,
        n_clusters: int = 10000,
        max_iter: int = 100,
        tol: float = 1e-2,
    ):
        """Args:
        dataset_dir: usvc dataset root (has meta.json + units/ artifacts).
        units_kind: "hubertsoft" | "contentvec" | ... (units encoder kind).
        units_hash: config hash of the units feature (provenance guard).
        n_clusters: KMeans cluster count (so-vits default 10000).
        max_iter / tol: KMeans convergence control.
        """
        self.dataset_dir = Path(dataset_dir)
        self.units_kind = units_kind
        self.units_hash = units_hash
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    def units_root(self) -> Path:
        for root in (self.dataset_dir / "units", self.dataset_dir / "features" / "units"):
            if (root / self.units_kind / self.units_hash).is_dir():
                return root / self.units_kind / self.units_hash
        raise FileNotFoundError(
            f"units artifacts not found: {self.dataset_dir}/units/{self.units_kind}/{self.units_hash}"
        )

    def out_path(self) -> Path:
        d = self.dataset_dir / "index"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"kmeans-{self.n_clusters}-{self.units_kind}-{self.units_hash[:8]}.pt"

    # ------------------------------------------------------------------
    def _load_speaker(self, spk_dir: Path) -> torch.Tensor:
        vecs = []
        for p in sorted(spk_dir.glob("*.pt")):
            t = torch.load(p, map_location="cpu", weights_only=True)
            if t.ndim == 3:  # [1, T, dim] legacy layout
                t = t.squeeze(0)
            vecs.append(t.float())
        if not vecs:
            return torch.empty(0, 0)
        return torch.cat(vecs, dim=0)

    # ------------------------------------------------------------------
    def build(self, device: str = "cuda") -> Path:
        """Train per-speaker clusters. Returns artifact path."""
        root = self.units_root()
        device = device if torch.cuda.is_available() else "cpu"
        payload: dict = {
            "schema_version": SCHEMA_VERSION,
            "kind": "units_cluster",
            "n_clusters": self.n_clusters,
            "units": {"encoder_kind": self.units_kind, "config_hash": self.units_hash},
        }
        spk_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        trained = 0
        for spk_dir in spk_dirs:
            vecs = self._load_speaker(spk_dir)
            if vecs.shape[0] < self.n_clusters:
                print(f"skip {spk_dir.name}: {vecs.shape[0]} vectors < n_clusters={self.n_clusters}")
                continue
            km = KMeansGPU(
                n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol,
                verbose=0, device=device,
            )
            km.fit_predict(vecs.to(device))
            payload[spk_dir.name] = {
                "cluster_centers_": km.centroids.cpu().contiguous(),
                "n_features_in_": vecs.shape[1],
            }
            trained += 1
            torch.cuda.empty_cache()

        if trained == 0:
            raise RuntimeError("no speaker had enough vectors for clustering")

        out = self.out_path()
        torch.save(payload, out)
        print(f"saved cluster model: {out} ({trained} speakers)")
        return out
