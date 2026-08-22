"""Index builder: units features -> retrieval index (pure torch, no faiss). One .pt per speaker
(sovits style) or one global index (RVC style); corpora above ``downsample_threshold`` vectors are
compressed to KMeans centroids (same strategy as the originals' MiniBatchKMeans->10000).
Search itself is at inference time in usvc.retrieval.predict."""
from __future__ import annotations

from pathlib import Path

import torch

from ._store import load_speaker, units_root
from .kmeans import KMeansGPU

SCHEMA_VERSION = 1
INDEX_SUFFIX = "-k8.pt"


class IndexBuilder:
    def __init__(self, dataset_dir, units_kind, units_hash,
                 n_clusters_downsample=10000, downsample_threshold=200000):
        self.dataset_dir = Path(dataset_dir)
        self.units_kind = units_kind
        self.units_hash = units_hash  # provenance guard against meta.json
        self.n_clusters_downsample = n_clusters_downsample
        self.downsample_threshold = downsample_threshold

    def _maybe_downsample(self, vectors, device):
        km = KMeansGPU(n_clusters=self.n_clusters_downsample, max_iter=100, tol=1e-2, verbose=0, device=device)
        km.fit_predict(vectors.to(device))
        return km.centroids.cpu()

    def _save(self, vectors, spk, n_total) -> Path:
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
        out = self.out_dir() / f"{self.units_kind}-{self.units_hash[:8]}-{spk or 'global'}{INDEX_SUFFIX}"
        torch.save(payload, out)
        return out

    def out_dir(self) -> Path:
        d = self.dataset_dir / "index"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def build(self, device="cuda", per_speaker=True) -> list[Path]:
        root = units_root(self.dataset_dir, self.units_kind, self.units_hash)
        device = device if torch.cuda.is_available() else "cpu"
        if not per_speaker:  # global (RVC style): concat all speakers, one index
            vecs_list = [load_speaker(d) for d in sorted(root.iterdir()) if d.is_dir()]
            vecs_list = [v for v in vecs_list if v.shape[0] > 0]
            if not vecs_list: raise FileNotFoundError(f"no units artifacts under {root}")
            vecs = torch.cat(vecs_list, dim=0)
            n_total = vecs.shape[0]
            if device != "cpu" and vecs.shape[0] > self.downsample_threshold:
                vecs = self._maybe_downsample(vecs, device)
            return [self._save(vecs, None, n_total)]
        out_paths = []
        for spk_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            vecs = load_speaker(spk_dir)
            if vecs.shape[0] == 0: continue
            n_total = vecs.shape[0]
            if device != "cpu" and vecs.shape[0] > self.downsample_threshold:
                vecs = self._maybe_downsample(vecs, device)
            out_paths.append(self._save(vecs, spk_dir.name, n_total))
            torch.cuda.empty_cache()
        return out_paths
