"""Shared dataset kernels (E1 dedup: was duplicated across 6 backend files).

BaseDirDataset:   meta / spk2idx / split-aware keys / torch.load helper
                  boilerplate (previously repeated in every backend dataset).
MelUnitsDataset:  mel/f0/rms/units .pt loader + optional pitch augmentation
                  (previously ddsp6/diffusion/reflow_shallow/rift_dit; the
                  four classes are now thin parameter wrappers).
slice_or_pad / align_spec_to_frames / align_wav:
                  fixed-length crop/pad + spec/wav force-alignment (was
                  duplicated inside rvc/sovits _load_one).

Attribute surface (frozen interface): every dataset keeps self.dataset_dir /
feature_hashes / encoder_kind / meta / spk2idx / keys / _load_pt / _mel_dir /
_f0_dir / _rms_dir / _units_dir / _aug_mel_dir / _aug_shift_dir, plus
self.meta.hop / sr reads; usvc/train/cli.py mutates encoder_kind and
_units_dir in place after construction.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.meta import load_meta
from ..train.dataset import BaseSVCDataset

__all__ = [
    "BaseDirDataset",
    "MelUnitsDataset",
    "slice_or_pad",
    "align_spec_to_frames",
    "align_wav",
]


class BaseDirDataset(BaseSVCDataset):
    """Common skeleton: meta, spk2idx, keys (split-aware), torch.load helper.

    Subclass sets its feature dirs in __init__ (after super().__init__()).
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 256,
        augment: bool = True,
        encoder_kind: str = "contentvec",
    ):
        super().__init__(dataset_dir, split, max_frames, augment)
        self.dataset_dir = Path(dataset_dir)
        self.feature_hashes = feature_hashes
        self.encoder_kind = encoder_kind

        self.meta = load_meta(str(dataset_dir))
        self.spk2idx = {s: i for i, s in enumerate(self.meta.speakers)}

        if self.meta.splits:
            split_files = set(self.meta.splits.get(split, []))
            self.keys = [
                (a.speaker, a.file_name)
                for a in self.meta.audios
                if a.file_name in split_files
            ]
        else:
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios]

        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")


class MelUnitsDataset(BaseDirDataset):
    """mel/f0/rms/units .pt loader, force-aligned to mel frame count.

    Parameters (per-backend defaults lived in the old per-class copies):
      max_frames:  ddsp6/diffusion 172, reflow_shallow 128, rift_dit 256
      encoder_kind: ddsp6/rift_dit contentvec_tta2x, others contentvec
      pitch_aug:  True for ddsp6/diffusion (precomputed aug_mel/aug_shift,
                  50% per sample), False for reflow_shallow/rift_dit.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 172,
        augment: bool = True,
        encoder_kind: str = "contentvec",
        pitch_aug: bool = False,
    ):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment, encoder_kind)
        self._mel_dir = self.dataset_dir / "mel" / feature_hashes.get("mel", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
        self._units_dir = (
            self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        )
        # Pitch augmentation (present only when use_pitch_aug preprocessed them).
        if pitch_aug:
            self._aug_mel_dir = self.dataset_dir / "aug_mel" / feature_hashes.get("aug_mel", "")
            self._aug_shift_dir = self.dataset_dir / "aug_shift" / feature_hashes.get("aug_shift", "")
        self._pitch_aug = pitch_aug

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0] + ".pt" if not fn.endswith(".pt") else fn

        mel = self._load_pt(self._mel_dir / spk / stem)
        f0 = self._load_pt(self._f0_dir / spk / stem)
        rms = self._load_pt(self._rms_dir / spk / stem)
        units = self._load_pt(self._units_dir / spk / stem)

        # Force-align to mel frame count (units may be off-by-one)
        T = mel.shape[0]
        f0 = f0[:T]
        rms = rms[:T]
        units = units[:T]

        return {
            "mel": mel,
            "f0": f0,
            "rms": rms,
            "units": units,
            "spk_id": self.spk2idx.get(spk, 0),
        }

    def _augment(self, item: dict) -> dict:
        return self._pitch_augment(item) if self._pitch_aug else item


def slice_or_pad(t: torch.Tensor, mf: int, T: int, start: int | None = None) -> torch.Tensor:
    """Trim (from ``start``) or right-pad dim 0 to exactly ``mf`` frames.

    ``T`` is the raw frame count; ``start`` is the random crop offset used
    only when T > mf (kept at the call site so each backend keeps its own
    random source). Padding along dim 0 right: F.pad last-dim-first.
    """
    if start is not None:
        return t[start : start + mf]
    if T > mf:
        return t[:mf]
    if T < mf:
        return F.pad(t, [0, 0] * (t.ndim - 1) + [0, mf - T])
    return t


def align_spec_to_frames(spec: torch.Tensor, T: int) -> torch.Tensor:
    """Force-align spec frames to exactly T (units frame count), then [C, T].

    spec and units may differ by ±1 frame due to differing STFT conventions
    (spec uses center=True len//hop+1; units align via gather).
    """
    if spec.ndim == 3:
        spec = spec[0]  # [T_spec, C]
    st = spec.shape[0]
    if st > T:
        spec = spec[:T]
    elif st < T:
        spec = F.pad(spec, [0, 0, 0, T - st])
    return spec.transpose(0, 1)  # [C, T]


def align_wav(wav: torch.Tensor, target: int) -> torch.Tensor:
    """Trim or right-pad wav to ``target`` samples (frame_len * hop)."""
    if wav.shape[0] > target:
        return wav[:target]
    if wav.shape[0] < target:
        return F.pad(wav, [0, target - wav.shape[0]])
    return wav