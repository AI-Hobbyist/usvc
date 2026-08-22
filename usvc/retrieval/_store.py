"""Shared units-artifact access for the index/cluster builders (2 impls existed -> one kernel)."""
from __future__ import annotations

from pathlib import Path

import torch


def units_root(dataset_dir, kind, h) -> Path:
    # current layout {ds}/units/..., legacy {ds}/features/units/...
    for root in (Path(dataset_dir) / "units", Path(dataset_dir) / "features" / "units"):
        if (root / kind / h).is_dir(): return root / kind / h
    raise FileNotFoundError(f"units artifacts not found: {dataset_dir}/units/{kind}/{h}")


def load_speaker(spk_dir: Path) -> torch.Tensor:
    ts = []
    for p in sorted(spk_dir.glob("*.pt")):
        t = torch.load(p, map_location="cpu", weights_only=True)
        ts.append(t.squeeze(0).float() if t.ndim == 3 else t.float())  # [1,T,dim] legacy
    return torch.cat(ts, dim=0) if ts else torch.empty(0, 0)  # [N, dim]
