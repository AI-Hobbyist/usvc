"""rvc dataset: preprocessed features for RVC VITS training.

RVC needs units(contentvec 768d) / pitch(quantized token 1-255) / pitchf
(continuous f0) / spec(linear spectrogram) / wav / sid. Shared skeleton in
usvc/backends/_dataset.py (BaseDirDataset + crop/pad helpers).
"""
from __future__ import annotations

import torch

from .._dataset import BaseDirDataset, align_spec_to_frames, align_wav, slice_or_pad


class RVCDataset(BaseDirDataset):
    def __init__(self, dataset_dir, split, feature_hashes, max_frames=790,
                 augment=True, encoder_kind="contentvec"):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment, encoder_kind)
        self._units_dir = self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")  # vol_embedding；可能不存在
        self._audio_dir = self.dataset_dir / "audio"

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0] + ".pt"
        units = self._load_pt(self._units_dir / spk / stem)
        f0 = self._load_pt(self._f0_dir / spk / stem)
        spec_path = self._spec_dir / spk / stem
        spec = self._load_pt(spec_path) if spec_path.exists() else None
        rms_path = self._rms_dir / spk / stem
        rms = self._load_pt(rms_path) if rms_path.exists() else None
        wav = None
        wav_path = self._audio_dir / spk / (fn.rsplit(".", 1)[0] + ".wav")
        if wav_path.exists():
            import soundfile as sf
            wav = torch.from_numpy(sf.read(str(wav_path), dtype="float32")[0])
        pitchf = f0.clone()
        pitch_token = self._f0_to_coarse(f0)
        T, mf = units.shape[0], self.max_frames
        if spec is not None and spec.ndim == 3:
            spec = spec[0]
        if T > mf:
            # 随机裁剪：RVC 自己的随机源 = GLOBAL 默认 generator（不是 self._rng，勿改）
            start = int(torch.randint(0, T - mf + 1, (1,)).item())
            units, pitch_token, pitchf = (slice_or_pad(u, mf, T, start)
                                          for u in (units, pitch_token, pitchf))
            if rms is not None:
                rms = slice_or_pad(rms, mf, T, start)
            if spec is not None:
                spec = slice_or_pad(spec, mf, T, start)
            T = mf
        elif T < mf:
            units, pitch_token, pitchf = (slice_or_pad(u, mf, T)
                                          for u in (units, pitch_token, pitchf))
            if rms is not None:
                rms = slice_or_pad(rms, mf, T)
            if spec is not None:
                spec = slice_or_pad(spec, mf, T)
            T = mf
        units, pitch_token, pitchf = units[:T], pitch_token[:T], pitchf[:T]
        if rms is not None:
            rms = rms[:T]
        if spec is not None:
            spec = align_spec_to_frames(spec, T)
        if wav is not None:
            wav = align_wav(wav, T * self.meta.hop)  # 帧数可超 wav 采样一帧（len//hop+1 惯例）
        return {"units": units, "f0": pitchf, "pitch_token": pitch_token, "mel": spec,
                "wav": wav, "spk_id": self.spk2idx.get(spk, 0), "rms": rms}

    @staticmethod
    def _f0_to_coarse(f0):
        # 委托 RVCModel.f0_to_coarse（RVC infer/rtrvc.py get_f0_post verbatim），
        # 训练/推理同一 token 网格；f0==0（清音）→ token 1。旧公式（f0_mel_min=0 /
        # clamp[1,254] / 清音=255）与 RVC 及模型推理路径不一致，静默偏移 pitch embedding
        from .model import RVCModel
        return RVCModel.f0_to_coarse(f0)