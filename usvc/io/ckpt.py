"""Checkpoint state-dict loader (shared by engine + enhancers).

Accepts common top-level wrappers:
- lightning: ``{"state_dict": {...}, ...}``
- so-vits/rvc original: ``{"model": {...}}``
- bare weight dict: ``{"weight": {...}}`` or the dict itself
"""
from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["load_ckpt_state"]


def load_ckpt_state(ckpt_path: str | Path) -> dict:
    """Load a state dict from a torch checkpoint (lightning or bare)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt.get("weight", ckpt)))
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f"cannot interpret checkpoint: {ckpt_path}")
    return sd
