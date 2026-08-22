"""Checkpoint state-dict loader (engine + enhancers): unwraps lightning ``state_dict`` /
so-vits-rvc ``model`` / bare ``weight`` top-level wrappers."""
from __future__ import annotations

import torch


def load_ckpt_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt.get("weight", ckpt))) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict): raise RuntimeError(f"cannot interpret checkpoint: {ckpt_path}")
    return sd
