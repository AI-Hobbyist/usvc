"""Inference-time retrieval and cluster mapping — pure torch, no faiss/sklearn.

Numerically replicates the original projects' formulas:

RVC / so-vits-svc feature retrieval (faiss IVF,Flat):
    score, ix = index.search(npy, k=8)          # score is *squared* L2
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    npy = np.sum(index_vectors[ix] * weight[..., None], axis=1)
    feats = ratio * npy + (1 - ratio) * feats

so-vits-svc cluster mapping:
    idx = model[speaker].predict(x)             # argmin L2 to centers
    mapped = model[speaker].cluster_centers_[idx]
    feats = ratio * mapped + (1 - ratio) * feats

NOTE: faiss returns SQUARED L2 distances; the weight formula uses
``(1/score)^2``. We reproduce it with ``torch.cdist(...).pow(2)`` — using a raw
L2 distance here would bias the weights (measured ~0.2 error).
"""
from __future__ import annotations

import torch

__all__ = ["retrieve", "apply_cluster"]


def retrieve(
    index_vectors: torch.Tensor,
    query: torch.Tensor,
    k: int = 8,
    ratio: float = 0.75,
    eps: float = 1e-8,
    chunk: int = 8192,
) -> torch.Tensor:
    """RVC/so-vits feature retrieval with (1/squared-L2)^2 weights.

    Args:
        index_vectors: [N, dim] index corpus (float32).
        query: [T, dim] query features (float32).
        k: top-K neighbors (original default 8).
        ratio: blend ratio (RVC ``index_rate`` / sovits ``cluster_infer_ratio``).
        eps: clamp floor for 1/score (avoids div-by-zero on exact matches).
        chunk: rows of ``query`` processed per cdist call (memory control).

    Returns:
        [T, dim] retrieval-enhanced features.
    """
    if index_vectors.dim() != 2:
        raise ValueError(f"index_vectors must be [N, dim], got {tuple(index_vectors.shape)}")
    if query.dim() != 2:
        raise ValueError(f"query must be [T, dim], got {tuple(query.shape)}")
    k = min(k, index_vectors.shape[0])

    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i : i + chunk].float()
        d2 = torch.cdist(q, index_vectors.float()).pow(2)  # [chunk, N] squared L2
        score, ix = d2.topk(k, dim=1, largest=False)
        weight = (1.0 / score.clamp_min(eps)).pow(2)
        weight = weight / weight.sum(dim=1, keepdim=True)
        out = (index_vectors[ix] * weight.unsqueeze(-1)).sum(dim=1)  # [chunk, dim]
        results.append(ratio * out + (1.0 - ratio) * q)
    return torch.cat(results, dim=0)


def apply_cluster(
    centers: torch.Tensor,
    query: torch.Tensor,
    ratio: float = 0.75,
    chunk: int = 8192,
) -> torch.Tensor:
    """so-vits cluster mapping: argmin-L2 -> cluster center -> blend.

    Args:
        centers: [n_clusters, dim] cluster centers.
        query: [T, dim] query features.
        ratio: blend ratio (0 = no clustering, 1 = pure centers).
        chunk: rows of ``query`` processed per cdist call (memory control).

    Returns:
        [T, dim] cluster-mapped features.
    """
    if centers.dim() != 2:
        raise ValueError(f"centers must be [n_clusters, dim], got {tuple(centers.shape)}")

    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i : i + chunk].float()
        dist = torch.cdist(q, centers.float())  # [chunk, n_clusters] L2
        idx = dist.argmin(dim=1)  # [chunk]
        mapped = centers[idx]
        results.append(ratio * mapped + (1.0 - ratio) * q)
    return torch.cat(results, dim=0)
