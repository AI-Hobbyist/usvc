"""Inference-time retrieval + cluster mapping — pure torch, no faiss/sklearn. Numerically
replicates the originals:

RVC/so-vits retrieval (faiss IVF,Flat): score is SQUARED L2; weight=(1/score)^2, row-normalized;
    out = sum(index_vectors[ix] * weight[..., None], axis=1); feats = ratio*out + (1-ratio)*feats
sovits cluster: idx=argmin L2 to centers; mapped=centers[idx]; feats = ratio*mapped+(1-ratio)*feats

NOTE: faiss returns squared L2 and the weight formula needs it — raw L2 here would bias the
weights (measured ~0.2 error), hence torch.cdist(...).pow(2)."""
from __future__ import annotations

import torch


def retrieve(index_vectors, query, k=8, ratio=0.75, eps=1e-8, chunk=8192):
    if index_vectors.dim() != 2:
        raise ValueError(f"index_vectors must be [N, dim], got {tuple(index_vectors.shape)}")
    if query.dim() != 2:
        raise ValueError(f"query must be [T, dim], got {tuple(query.shape)}")
    k = min(k, index_vectors.shape[0])

    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk].float()
        d2 = torch.cdist(q, index_vectors.float()).pow(2)  # [chunk, N] squared L2
        score, ix = d2.topk(k, dim=1, largest=False)
        weight = (1.0 / score.clamp_min(eps)).pow(2)
        weight = weight / weight.sum(dim=1, keepdim=True)
        out = (index_vectors[ix] * weight.unsqueeze(-1)).sum(dim=1)  # [chunk, dim]
        results.append(ratio * out + (1.0 - ratio) * q)
    return torch.cat(results, dim=0)


def apply_cluster(centers, query, ratio=0.75, chunk=8192):
    if centers.dim() != 2:
        raise ValueError(f"centers must be [n_clusters, dim], got {tuple(centers.shape)}")

    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk].float()
        idx = torch.cdist(q, centers.float()).argmin(dim=1)  # [chunk]
        results.append(ratio * centers[idx] + (1.0 - ratio) * q)
    return torch.cat(results, dim=0)
