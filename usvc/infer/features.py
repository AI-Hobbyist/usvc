"""Inference-time feature extraction — same extractors as preprocessing, driven by FeatureSpec."""
from __future__ import annotations

from contextlib import nullcontext as _nullctx
from pathlib import Path

import torch

from ..data.feature_config import FeatureConfig, FeatureSpec
from ..features.f0 import F0Extractor
from ..features.rms import RmsExtractor
from ..features.units import UnitsExtractor
from ..io.audio import load_audio, resample_np


class InferFeatures:
    def __init__(self, spec: FeatureSpec, sr: int = 44100, hop: int = 512,
                 device: str = "cuda", dtype: str = "fp32") -> None:
        assert spec.units is not None and spec.f0 is not None, "backend feature_spec requires units/f0"
        self.spec, self.sr, self.hop, self.device, self.dtype = spec, sr, hop, device, dtype
        self.units_ext = UnitsExtractor(spec.units, sr, hop, device, dtype=dtype)
        self.f0_ext = F0Extractor(spec.f0, sr, hop, device, dtype=dtype)
        self.rms_ext = RmsExtractor(spec.rms, sr, hop, device) if spec.rms is not None else None
        self._f0_extractors = {}
        self._gpu_resamplers = {}
        self._gpu_enabled = device == "cuda"

    def _gpu_resample(self, wav, orig_sr, target_sr):
        # torchaudio GPU sinc resample (0.15ms vs CPU soxr 6ms; SNR 55dB 可接受)
        import numpy as np
        if orig_sr == target_sr:
            return np.ascontiguousarray(wav, dtype=np.float32)
        import torchaudio
        rs = self._gpu_resamplers.get((orig_sr, target_sr))
        if rs is None:
            rs = torchaudio.transforms.Resample(orig_sr, target_sr,
                                                dtype=torch.float32).to(self.device)
            self._gpu_resamplers[(orig_sr, target_sr)] = rs
        return rs(torch.as_tensor(wav, dtype=torch.float32,
                                  device=self.device)).cpu().numpy()

    def _get_f0_extractor(self, method: str) -> F0Extractor:
        # 运行时切换 f0 提取器 (rmvpe/fcpe); 参数复制 spec.f0, 只换 extractor 名
        assert method in ("rmvpe", "fcpe"), f"unsupported f0 method: {method}"
        if method == self.f0_ext.extractor:
            return self.f0_ext
        cached = self._f0_extractors.get(method)
        if cached is None:
            params = dict(self.spec.f0.params)
            params["extractor"] = method
            cached = F0Extractor(FeatureConfig("f0", params), self.sr, self.hop,
                                 self.device, dtype=self.dtype)
            self._f0_extractors[method] = cached
        return cached

    @torch.inference_mode()
    def extract(self, wav_path=None, f0_method=None, *, wav=None) -> dict:
        if wav is None:
            assert wav_path is not None, "either wav_path or wav must be given"
            if self._gpu_enabled:
                import soundfile as sf_read
                wav_raw, orig_sr = sf_read.read(str(wav_path), dtype="float32")
                if wav_raw.ndim > 1:  # stereo -> mono
                    wav_raw = wav_raw.mean(axis=-1)
                wav = self._gpu_resample(wav_raw, orig_sr, self.sr)
            else:
                wav = load_audio(wav_path, sr=self.sr)
        if self._gpu_enabled:
            wav16k = self._gpu_resample(wav, self.sr, 16000)
        else:
            wav16k = resample_np(wav, self.sr, 16000)
        with torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else _nullctx():
            units = self.units_ext.extract(torch.from_numpy(wav16k).float().to(self.device))
        wav_t = torch.from_numpy(wav).float()
        f0 = (self._get_f0_extractor(f0_method) if f0_method is not None
              else self.f0_ext).extract(wav_t)
        rms = self.rms_ext.extract(wav_t) if self.rms_ext is not None else None
        uv = (f0 > 0).float()
        T = units.shape[0]  # 统一裁剪到公共帧数 (frame_len_for 对齐)
        if f0.shape[0] != T:
            f0, uv = f0[:T], uv[:T]
        if rms is not None and rms.shape[0] != T:
            rms = rms[:T]
        return {"units": units.float(), "f0": f0, "rms": rms, "uv": uv,
                "wav": wav, "sr": self.sr}


def synth_test_wav(path, sr=44100, duration=2.0, f0_base=220.0) -> np.ndarray:
    import numpy as np
    import soundfile as sf
    t = np.arange(int(sr * duration)) / sr
    vib = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    wav = 0.5 * np.sin(2 * np.pi * np.cumsum(f0_base + vib) / sr) * np.exp(-t / duration)
    sf.write(path, wav.astype(np.float32), sr)
    return wav