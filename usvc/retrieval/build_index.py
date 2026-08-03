"""Index builder: aggregate units features -> retrieval index (pure torch).

Consumes the preprocessed units artifacts (``{dataset_dir}/units/{kind}/{hash}/{spk}/*.pt``)
and produces a single .pt index per speaker (sovits style) or one global index
(RVC style). Large corpora (> ``downsample_threshold`` vectors) are compressed
to ``n_clusters_downsample`` centroids via the pure-torch GPU KMeans — the same
strategy the original projects use (MiniBatchKMeans -> 10000 centers).

The index is a plain tensor — no faiss. Inference-time search (top-K squared-L2
nearest neighbors + (1/score)^2 weights) is in ``usvc.retrieval.predict``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .kmeans import KMeansGPU

__all__ = ["IndexBuilder", "SCHEMA_VERSION", "INDEX_SUFFIX"]

SCHEMA_VERSION = 1
INDEX_SUFFIX = "-k8.pt"


class IndexBuilder:
    def __init__(
        self,
        dataset_dir: str | Path,
        units_kind: str,
        units_hash: str,
        n_clusters_downsample: int = 10000,
        downsample_threshold: int = 200000,
    ):
        """Args:
        dataset_dir: usvc dataset root (has meta.json + units/ artifacts).
        units_kind: "hubertsoft" | "contentvec" | ... (units encoder kind).
        units_hash: config hash of the units feature (provenance guard).
        n_clusters_downsample: target centroids when compressing big corpora.
        downsample_threshold: vectors above this are compressed (original 200k).
        """
        self.dataset_dir = Path(dataset_dir)
        self.units_kind = units_kind
        self.units_hash = units_hash
        self.n_clusters_downsample = n_clusters_downsample
        self.downsample_threshold = downsample_threshold

    # ------------------------------------------------------------------
    def units_root(self) -> Path:
        # Historical layouts: {dataset}/units/... (current) and
        # {dataset}/features/units/... (legacy). Prefer the current one.
        for root in (self.dataset_dir / "units", self.dataset_dir / "features" / "units"):
            if (root / self.units_kind / self.units_hash).is_dir():
                return root / self.units_kind / self.units_hash
        raise FileNotFoundError(
            f"units artifacts not found: {self.dataset_dir}/units/{self.units_kind}/{self.units_hash}"
        )

    def out_dir(self) -> Path:
        d = self.dataset_dir / "index"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    def _load_speaker(self, spk_dir: Path, device: str) -> torch.Tensor:
        vecs = []
        for p in sorted(spk_dir.glob("*.pt")):
            t = torch.load(p, map_location="cpu", weights_only=True)
            if t.ndim == 3:  # [1, T, dim] legacy layout
                t = t.squeeze(0)
            vecs.append(t.float())
        if not vecs:
            return torch.empty(0, 0)
        return torch.cat(vecs, dim=0)  # [N, dim]

    def _maybe_downsample(self, vectors: torch.Tensor, device: str) -> torch.Tensor:
        n = vectors.shape[0]
        if n <= self.downsample_threshold:
            return vectors
        km = KMeansGPU(n_clusters=self.n_clusters_downsample, max_iter=100, tol=1e-2, verbose=0, device=device)
        km.fit_predict(vectors.to(device))
        return km.centroids.cpu()  # [10000, dim]

    def _save(
        self,
        vectors: torch.Tensor,
        spk: str | None,
        n_total: int,
        *,
        device: str,
    ) -> Path:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "units_index",
            "vectors": vectors.contiguous(),
            "spk_id": spk,
            "units": {"encoder_kind": self.units_kind, "config_hash": self.units_hash},
            "config": {"k": 8, "metric": "l2_sq", "downsample": {
                "n_clusters": self.n_clusters_downsample,
                "threshold": self.downsample_threshold,
                "applied": vectors.shape[0] < n_total,
            }},
            "stats": {"n_total": n_total, "n_indexed": vectors.shape[0], "dim": vectors.shape[1]},
        }
        tag = spk if spk is not None else "global"
        out = self.out_dir() / f"{self.units_kind}-{self.units_hash[:8]}-{tag}{INDEX_SUFFIX}"
        torch.save(payload, out)
        return out

    # ------------------------------------------------------------------
    def build(self, device: str = "cuda", per_speaker: bool = True) -> list[Path]:
        """Build index(es). Returns list of artifact paths."""
        root = self.units_root()
        device = device if torch.cuda.is_available() else "cpu"

        if per_speaker:
            spk_dirs = sorted(d for d in root.iterdir() if d.is_dir())
            out_paths = []
            for spk_dir in spk_dirs:
                vecs = self._load_speaker(spk_dir, device)
                if vecs.shape[0] == 0:
                    continue
                n_total = vecs.shape[0]
                if device != "cpu" and vecs.shape[0] > self.downsample_threshold:
                    vecs = self._maybe_downsample(vecs, device)
                out_paths.append(self._save(vecs, spk_dir.name, n_total, device=device))
                torch.cuda.empty_cache()
            return out_paths

        # Global (RVC style): concat all speakers, one index.
        vecs_list = [self._load_speaker(d, device) for d in sorted(root.iterdir()) if d.is_dir()]
        vecs_list = [v for v in vecs_list if v.shape[0] > 0]
        if not vecs_list:
            raise FileNotFoundError(f"no units artifacts under {root}")
        vecs = torch.cat(vecs_list, dim=0)
        n_total = vecs.shape[0]
        if device != "cpu" and vecs.shape[0] > self.downsample_threshold:
            vecs = self._maybe_downsample(vecs, device)
        return [self._save(vecs, None, n_total, device=device)]
