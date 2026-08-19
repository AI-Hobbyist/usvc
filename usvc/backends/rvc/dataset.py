"""rvc dataset: reads preprocessed features for RVC VITS training.

RVC needs: units(contentvec 768d), pitch(quantized token 1-255), pitchf(continuous f0),
spec(linear spectrogram), wav(waveform), sid(speaker id).
Shared skeleton lives in usvc/backends/_dataset.py (BaseDirDataset + crop/pad
helpers). The random crop keeps RVC's own random source: the GLOBAL default
torch generator (NOT the per-dataset self._rng — see SoVITSDataset).
"""
from __future__ import annotations

import torch

from .._dataset import BaseDirDataset, align_spec_to_frames, align_wav, slice_or_pad

__all__ = ["RVCDataset"]


class RVCDataset(BaseDirDataset):
    """Dataset for rvc backend (VITS + NSF-HiFiGAN, v2)."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        feature_hashes: dict,
        max_frames: int = 790,
        augment: bool = True,
        encoder_kind: str = "contentvec",
    ):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment, encoder_kind)

        self._units_dir = (
            self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        )
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")
        # rms (volume) for vol_embedding; may not exist in all datasets
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
        self._audio_dir = self.dataset_dir / "audio"

    def _load_one(self, key) -> dict:
        spk, fn = key
        stem = fn.rsplit(".", 1)[0]
        pt_stem = stem + ".pt"
        wav_stem = stem + ".wav"

        units = self._load_pt(self._units_dir / spk / pt_stem)
        f0 = self._load_pt(self._f0_dir / spk / pt_stem)

        # spec (linear spectrogram)
        spec_path = self._spec_dir / spk / pt_stem
        spec = self._load_pt(spec_path) if spec_path.exists() else None

        # rms (volume) for vol_embedding; may not exist in all datasets
        rms_path = self._rms_dir / spk / pt_stem
        rms = self._load_pt(rms_path) if rms_path.exists() else None

        # wav
        wav_path = self._audio_dir / spk / wav_stem
        wav = None
        if wav_path.exists():
            import soundfile as sf

            wav, _ = sf.read(str(wav_path), dtype="float32")
            wav = torch.from_numpy(wav)

        # RVC f0: quantize to token (1-255) + keep continuous pitchf
        pitchf = f0.clone()
        pitch_token = self._f0_to_coarse(f0)

        T = units.shape[0]
        # Fixed-length random crop for consistent batching: long audio crops to
        # max_frames (random offset), short audio pads. RVC segment_size must fit.
        mf = self.max_frames
        if T > mf:
            start = int(torch.randint(0, T - mf + 1, (1,)).item())  # global default generator
            units = slice_or_pad(units, mf, T, start)
            pitch_token = slice_or_pad(pitch_token, mf, T, start)
            pitchf = slice_or_pad(pitchf, mf, T, start)
            if rms is not None:
                rms = slice_or_pad(rms, mf, T, start)
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = slice_or_pad(spec, mf, T, start)
            T = mf
        elif T < mf:
            units = slice_or_pad(units, mf, T)
            pitch_token = slice_or_pad(pitch_token, mf, T)
            pitchf = slice_or_pad(pitchf, mf, T)
            if rms is not None:
                rms = slice_or_pad(rms, mf, T)
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = slice_or_pad(spec, mf, T)
            T = mf
        else:
            if spec is not None and spec.ndim == 3:
                spec = spec[0]
        units = units[:T]
        pitch_token = pitch_token[:T]
        pitchf = pitchf[:T]
        if rms is not None:
            rms = rms[:T]
        if spec is not None:
            spec = align_spec_to_frames(spec, T)
        # Align wav to frame_len * hop (frame count may exceed wav samples by one
        # frame due to len//hop + 1 convention). GAN wave slicing uses
        # ids_slice*hop + segment <= T*hop.
        if wav is not None:
            wav = align_wav(wav, T * self.meta.hop)

        return {
            "units": units,
            "f0": pitchf,  # continuous f0 (SVCBatch.f0)
            "pitch_token": pitch_token,  # quantized f0 token [T]
            "mel": spec,
            "wav": wav,
            "spk_id": self.spk2idx.get(spk, 0),
            "rms": rms,  # volume [T] or None (vol_embedding)
        }

    @staticmethod
    def _f0_to_coarse(f0: torch.Tensor) -> torch.Tensor:
        """Quantize f0 to RVC pitch token (1-255).

        Delegates to RVCModel.f0_to_coarse (the verbatim RVC mapping from
        infer/rtrvc.py get_f0_post) so training and inference use the SAME
        token grid. f0==0 (unvoiced) maps to token 1 (the lowest bin), matching
        RVC — NOT 255. Previously this used a different formula (f0_mel_min=0,
        clamp[1,254], unvoiced=255) that was inconsistent with both RVC and the
        model's own inference path, silently shifting the pitch embedding.
        """
        from .model import RVCModel

        return RVCModel.f0_to_coarse(f0)