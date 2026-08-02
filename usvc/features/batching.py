"""Batching utilities for GPU feature extractors.

Length-based bucket batching: group variable-length clips so each batch can be
padded to a common length with minimal waste. Tight buckets keep padding < 10%,
which keeps numerical drift from attention-padding contamination negligible
(< 1e-2 vs single-clip reference).
"""
from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["bucket_batch", "pad_batch"]


def bucket_batch(
    lengths: Sequence[int],
    *,
    batch_size: int,
    max_pad_ratio: float = 0.10,
) -> list[list[int]]:
    """Group clip indices into batches with near-equal lengths.

    Sort by length, then greedily fill each batch up to ``batch_size`` while
    keeping the padding ratio (max_len - min_len) / max_len under
    ``max_pad_ratio``. A clip that would exceed the ratio starts a new batch.
    Returns a list of index-lists (original indices).
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_min = None
    for idx in order:
        L = lengths[idx]
        if cur and len(cur) >= batch_size:
            batches.append(cur)
            cur = []
            cur_min = None
        if cur_min is not None and L > 0:
            pad_ratio = (L - cur_min) / L
            if pad_ratio > max_pad_ratio and len(cur) > 0:
                # start new batch to keep padding small
                batches.append(cur)
                cur = []
                cur_min = None
        if cur_min is None:
            cur_min = L
        cur.append(idx)
    if cur:
        batches.append(cur)
    return batches


def pad_batch(wavs: list[torch.Tensor], *, pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of 1D tensors to equal length. Returns (padded [B, T_max], lengths [B])."""
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    tmax = int(lengths.max().item())
    device = wavs[0].device
    out = torch.full((len(wavs), tmax), pad_value, dtype=wavs[0].dtype, device=device)
    for i, w in enumerate(wavs):
        out[i, : w.shape[-1]] = w
    return out, lengths
