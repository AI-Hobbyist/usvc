"""Unified batch dataclass and collate function.

SVCBatch is the common batch schema across all 5 backends. Each backend's
Dataset returns a dict per-sample; svc_collate stacks them into SVCBatch.
Optional fields (mel, wav, rms, uv, aug_shift) are filled per-backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

__all__ = ["SVCBatch", "svc_collate", "BaseSVCDataset"]


@dataclass
class SVCBatch:
    """Unified training batch.

    Frame-level fields ([B, T, ...]) are right-padded to the batch's max frame
    count. frame_len[b] records the true (unpadded) length for masking.

    Attributes:
        units:    [B, T, D] content features (cvec/contentvec/hubert-soft)
        f0:       [B, T]    continuous fundamental frequency (Hz)
        mel:      [B, T, M] target mel spectrogram (flow/diff backends; GAN uses spec)
        wav:      [B, L]    waveform (GAN backends: discriminator input)
        rms:      [B, T]    energy (rms or volume)
        spk_id:   [B]       speaker ID (integer)
        uv:       [B, T]    voicing flag (0/1)
        aug_shift:[B]       pitch augmentation shift (semitones)
        frame_len:[B]       true frame count per sample (pre-pad)
        key:      list[str] sample identifiers (for logging/debugging)
    """

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


def _pad_stack(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Pad a list of [T, ...] tensors to max T along dim, then stack → [B, T_max, ...]."""
    if dim != 0:
        raise NotImplementedError("only dim=0 supported")
    max_len = max(t.shape[0] for t in tensors)
    ndim = tensors[0].ndim
    padded = []
    for t in tensors:
        pad_amount = max_len - t.shape[0]
        if pad_amount > 0:
            # F.pad takes (last_dim_first, last_dim_first, ...)
            pad_dims = []
            for _ in range(ndim - 1):
                pad_dims.extend([0, 0])
            pad_dims.extend([0, pad_amount])  # pad dim 0 on the right
            t = F.pad(t, pad_dims, value=0.0)
        padded.append(t)
    return torch.stack(padded, dim=0)


def svc_collate(samples: list[dict]) -> SVCBatch:
    """Collate a list of per-sample dicts into SVCBatch.

    Each sample dict may contain keys: units, f0, mel, wav, rms, spk_id, uv,
    aug_shift, frame_len, key.
    Variable-length frame fields are padded; frame_len records true lengths.
    """
    keys = [s.get("key", str(i)) for i, s in enumerate(samples)]

    # Always present
    units = _pad_stack([s["units"] for s in samples])
    f0 = _pad_stack([s["f0"].unsqueeze(-1) for s in samples]).squeeze(-1)

    # frame_len: true lengths before padding
    frame_len = torch.tensor(
        [s["units"].shape[0] for s in samples], dtype=torch.long
    )

    batch = SVCBatch(units=units, f0=f0, frame_len=frame_len, key=keys)

    # Optional fields
    if "mel" in samples[0] and samples[0]["mel"] is not None:
        batch.mel = _pad_stack([s["mel"] for s in samples])
    if "wav" in samples[0] and samples[0]["wav"] is not None:
        # wav is 1D [L]; pad along length
        wavs = [s["wav"] for s in samples]
        max_w = max(w.shape[0] for w in wavs)
        padded_w = [
            F.pad(w, (0, max_w - w.shape[0]), value=0.0) for w in wavs
        ]
        batch.wav = torch.stack(padded_w, dim=0)
    if "rms" in samples[0] and samples[0]["rms"] is not None:
        batch.rms = _pad_stack([s["rms"].unsqueeze(-1) for s in samples]).squeeze(-1)
    if "spk_id" in samples[0] and samples[0]["spk_id"] is not None:
        batch.spk_id = torch.tensor(
            [s["spk_id"] for s in samples], dtype=torch.long
        )
    if "uv" in samples[0] and samples[0]["uv"] is not None:
        batch.uv = _pad_stack([s["uv"].unsqueeze(-1) for s in samples]).squeeze(-1)
    if "aug_shift" in samples[0] and samples[0]["aug_shift"] is not None:
        aug = [s["aug_shift"] for s in samples]
        # aug_shift can be a scalar [B] or per-frame [B, T]
        if aug[0].ndim == 0:
            batch.aug_shift = torch.tensor(aug, dtype=torch.float32)
        else:
            batch.aug_shift = _pad_stack(aug)
    if "pitch_token" in samples[0] and samples[0]["pitch_token"] is not None:
        batch.pitch_token = _pad_stack([s["pitch_token"].unsqueeze(-1) for s in samples]).squeeze(-1).long()

    return batch


class BaseSVCDataset(Dataset):
    """Base class for SVC datasets.

    Subclass must:
      1. Populate self.keys (list of sample identifiers) in __init__
      2. Implement _load_one(key) -> dict with at least {units, f0}

    Provides:
      - _crop(): random temporal crop to self.max_frames
      - frame masking via frame_len in collate
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        max_frames: int = 256,
        augment: bool = True,
        seed: int = 42,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.max_frames = max_frames
        self.augment = augment
        self.keys: list[str] = []
        self._rng = torch.Generator().manual_seed(seed)
        # Pitch-aug dirs (None = feature not preprocessed -> no augmentation).
        self._aug_mel_dir: Path | None = None
        self._aug_shift_dir: Path | None = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx]
        item = self._load_one(key)
        item["key"] = key
        item = self._crop(item)
        if self.augment and self.split == "train":
            item = self._augment(item)
        return item

    def _load_one(self, key: str) -> dict:
        """Load one sample. Subclass must implement.

        Returns dict with at least: units [T, D], f0 [T].
        May include: mel, wav, rms, spk_id, uv, aug_shift.
        """
        raise NotImplementedError

    def _augment(self, item: dict) -> dict:
        """Override for backend-specific augmentation (pitch shift, vol aug, etc.)."""
        return item

    def _pitch_augment(self, item: dict) -> dict:
        """Pitch augmentation (DDSP-SVC style): 50% chance swap in the
        pre-computed aug_mel and shift f0 by the file's recorded aug_shift.

        Requires aug_mel/aug_shift features preprocessed (else no-op).
        """
        if not self.augment or self._aug_mel_dir is None or self._aug_shift_dir is None:
            return item
        # Always attach aug_shift so svc_collate never sees a missing key in a
        # mixed batch (it keys off samples[0]). 0 = no shift = un-augmented.
        # Shape [1, 1] per sample -> collate pads to [B, 1, 1], matching the
        # original DDSP-SVC convention (data_loaders.py: [[shift]]), so
        # aug_shift_embed(aug_shift/5) yields [B, 1, dim] broadcastable to x.
        if torch.rand(1, generator=self._rng).item() >= 0.5:
            item["aug_shift"] = torch.zeros(1, 1)
            return item  # 50% un-augmented
        spk, fn = item["key"]
        stem = fn.rsplit(".", 1)[0] + ".pt"
        aug_mel_path = self._aug_mel_dir / spk / stem
        aug_shift_path = self._aug_shift_dir / spk / stem
        if not aug_mel_path.is_file() or not aug_shift_path.is_file():
            item["aug_shift"] = torch.zeros(1, 1)
            return item
        aug_mel = self._load_pt(aug_mel_path)
        shift = self._load_pt(aug_shift_path).item()
        T = item["mel"].shape[0]
        item["mel"] = aug_mel[:T]
        item["f0"] = item["f0"] * (2.0 ** (shift / 12.0))
        item["aug_shift"] = torch.tensor([[shift]], dtype=torch.float32)
        return item

    def _crop(self, item: dict) -> dict:
        """Random temporal crop to max_frames.

        For train split: random start. For val/test: start from 0.
        If the sample is shorter than max_frames, return as-is.
        """
        T = item["units"].shape[0]
        if T <= self.max_frames:
            return item

        if self.split == "train":
            start = int(torch.randint(0, T - self.max_frames + 1, (1,), generator=self._rng).item())
        else:
            start = 0

        end = start + self.max_frames
        # Crop all [T, ...] tensors
        for k, v in item.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == T:
                item[k] = v[start:end]
        return item
