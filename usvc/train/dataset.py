"""Unified batch schema (SVCBatch), collate, and dataset base for all backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class SVCBatch:
    """Frame-level fields ([B,T,...]) right-padded to the batch max frame count; frame_len[b] = true (pre-pad) length."""
    units: torch.Tensor
    f0: torch.Tensor
    mel: torch.Tensor | None = None
    wav: torch.Tensor | None = None
    rms: torch.Tensor | None = None
    spk_id: torch.Tensor | None = None
    uv: torch.Tensor | None = None
    aug_shift: torch.Tensor | None = None
    pitch_token: torch.Tensor | None = None  # RVC: quantized f0 token [B, T]
    frame_len: torch.Tensor | None = None
    key: list[str] = field(default_factory=list)


def _pad_stack(tensors, dim=0):
    """list of [T,...] -> [B, Tmax, ...] right-padded with 0 along dim 0."""
    max_len = max(t.shape[0] for t in tensors)
    ndim = tensors[0].ndim
    # F.pad takes trailing pairs (last_dim_first...)
    return torch.stack([
        t if t.shape[0] == max_len else F.pad(t, [0, 0] * (ndim - 1) + [0, max_len - t.shape[0]])
        for t in tensors
    ], dim=0)


def svc_collate(samples):
    """dicts -> SVCBatch; presence of optional fields is keyed off samples[0] (mixed-key batches drop them)."""
    keys = [s.get("key", str(i)) for i, s in enumerate(samples)]
    frame_len = torch.tensor([s["units"].shape[0] for s in samples], dtype=torch.long)
    batch = SVCBatch(units=_pad_stack([s["units"] for s in samples]),
                     f0=_pad_stack([s["f0"].unsqueeze(-1) for s in samples]).squeeze(-1),
                     frame_len=frame_len, key=keys)
    s0 = samples[0]
    if "mel" in s0 and s0["mel"] is not None: batch.mel = _pad_stack([s["mel"] for s in samples])
    if "wav" in s0 and s0["wav"] is not None:
        wavs = [s["wav"] for s in samples]
        max_w = max(w.shape[0] for w in wavs)
        batch.wav = torch.stack([F.pad(w, (0, max_w - w.shape[0])) for w in wavs], dim=0)
    if "rms" in s0 and s0["rms"] is not None: batch.rms = _pad_stack([s["rms"].unsqueeze(-1) for s in samples]).squeeze(-1)
    if "spk_id" in s0 and s0["spk_id"] is not None: batch.spk_id = torch.tensor([s["spk_id"] for s in samples], dtype=torch.long)
    if "uv" in s0 and s0["uv"] is not None: batch.uv = _pad_stack([s["uv"].unsqueeze(-1) for s in samples]).squeeze(-1)
    if "aug_shift" in s0 and s0["aug_shift"] is not None:
        aug = [s["aug_shift"] for s in samples]
        batch.aug_shift = torch.tensor(aug, dtype=torch.float32) if aug[0].ndim == 0 else _pad_stack(aug)
    if "pitch_token" in s0 and s0["pitch_token"] is not None:
        batch.pitch_token = _pad_stack([s["pitch_token"].unsqueeze(-1) for s in samples]).squeeze(-1).long()
    return batch


class BaseSVCDataset(Dataset):
    # Subclass: populate self.keys in __init__, implement _load_one(key)->dict (>= {units, f0})
    def __init__(self, dataset_dir, split="train", max_frames=256, augment=True, seed=42):
        self.dataset_dir = dataset_dir
        self.split, self.max_frames, self.augment = split, max_frames, augment
        self.keys = []
        self._rng = torch.Generator().manual_seed(seed)
        self._aug_mel_dir: Path | None = None  # pitch-aug feature dirs (None = no augmentation)
        self._aug_shift_dir: Path | None = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item = self._load_one(self.keys[idx])
        item["key"] = self.keys[idx]
        item = self._crop(item)
        return self._augment(item) if self.augment and self.split == "train" else item

    def _load_one(self, key):
        raise NotImplementedError

    def _augment(self, item):
        return item

    def _pitch_augment(self, item):
        # DDSP-SVC style: 50% swap in precomputed aug_mel + f0 shift by the file's aug_shift.
        # aug_shift is ALWAYS attached ([1,1], 0 = no shift) so svc_collate never sees a missing key
        # in a mixed batch (it keys off samples[0]) — shape [1,1] matches DDSP convention [[shift]]
        # so aug_shift_embed(aug_shift/5) broadcast to x is [B,1,dim].
        if not self.augment or self._aug_mel_dir is None or self._aug_shift_dir is None:
            return item
        if torch.rand(1, generator=self._rng).item() >= 0.5:
            item["aug_shift"] = torch.zeros(1, 1); return item  # 50% un-augmented
        spk, fn = item["key"]
        stem = fn.rsplit(".", 1)[0] + ".pt"
        aug_mel_path, aug_shift_path = self._aug_mel_dir / spk / stem, self._aug_shift_dir / spk / stem
        if not aug_mel_path.is_file() or not aug_shift_path.is_file():
            item["aug_shift"] = torch.zeros(1, 1); return item
        shift = self._load_pt(aug_shift_path).item()
        item["mel"] = self._load_pt(aug_mel_path)[:item["mel"].shape[0]]
        item["f0"] = item["f0"] * (2.0 ** (shift / 12.0))
        item["aug_shift"] = torch.tensor([[shift]], dtype=torch.float32)
        return item

    def _crop(self, item):
        # random temporal crop to max_frames (train: random start; val/test: 0); shorter samples pass through
        T = item["units"].shape[0]
        if T <= self.max_frames:
            return item
        start = int(torch.randint(0, T - self.max_frames + 1, (1,), generator=self._rng).item()) \
            if self.split == "train" else 0
        return {k: v[start:start + self.max_frames] if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == T
                else v for k, v in item.items()}