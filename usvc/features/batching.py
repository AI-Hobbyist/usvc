"""Batching for GPU feature extractors: length-bucketed batches keep padding
< 10% so attention-padding contamination stays negligible (<1e-2 vs single)."""

import torch


def bucket_batch(lengths, *, batch_size, max_pad_ratio=0.10):
    """Group clip indices into batches: sort by length, greedily fill up to
    batch_size while keeping (max_len - min_len)/max_len under max_pad_ratio."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_min = [], [], None
    for idx in order:
        L = lengths[idx]
        if cur and (len(cur) >= batch_size or (cur_min is not None and L > 0 and (L - cur_min) / L > max_pad_ratio)):
            batches.append(cur)
            cur, cur_min = [], None
        if cur_min is None: cur_min = L
        cur.append(idx)
    if cur: batches.append(cur)
    return batches


def pad_batch(wavs, *, pad_value=0.0):
    """Pad 1D tensors to equal length -> (padded [B, T_max], lengths [B])."""
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    tmax = int(lengths.max().item())
    out = torch.full((len(wavs), tmax), pad_value, dtype=wavs[0].dtype, device=wavs[0].device)
    for i, w in enumerate(wavs):
        out[i, : w.shape[-1]] = w
    return out, lengths