"""Retrieval index + cluster: pure torch, zero faiss/sklearn.

Consumes preprocessed units artifacts and produces:
  - per-speaker or global retrieval indexes  (``IndexBuilder``)
  - per-speaker KMeans cluster models        (``ClusterTrainer``)

Inference-time helpers (``retrieve`` / ``apply_cluster``) numerically
replicate RVC feature retrieval and so-vits cluster mapping.
"""
from __future__ import annotations

from .build_cluster import ClusterTrainer
from .build_index import IndexBuilder
from .kmeans import KMeansGPU
from .predict import apply_cluster, retrieve

__all__ = [
    "IndexBuilder",
    "ClusterTrainer",
    "KMeansGPU",
    "retrieve",
    "apply_cluster",
]
