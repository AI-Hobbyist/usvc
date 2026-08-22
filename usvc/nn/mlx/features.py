"""MLX 特征提取(ContentVec units + RMVPE/FCPE f0 + RMS;替代 torch
InferFeatures,全 MLX 无 torch 依赖)。对齐 torch 语义:
units layer12 → floor-gather 到 hop 帧;f0 原生 ~100Hz → 线性插值;
rms 走 nvSTFT center=False 网格。"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

__all__ = ["MlxFeatureExtractor"]

_NATIVE_FRAME_RATE = 100.0  # RMVPE 原生帧率(16k/160hop),与 torch 一致


class MlxFeatureExtractor:
    def __init__(self, sr: int, hop: int, units_ckpt: str | Path | None = None,
                 rmvpe_ckpt: str | Path | None = None,
                 fcpe_ckpt: str | Path | None = None, device: str = "gpu",
                 rms_win_length: int | None = None, uv_policy: str = "interp",
                 units_layer: int = 12):
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.sr, self.hop, self.device = int(sr), int(hop), device
        self.uv_policy = uv_policy
        self.units_layer = int(units_layer)
        self.rms_win_length = int(rms_win_length) if rms_win_length else int(hop)

        from ...hub import ensure_pretrained
        if units_ckpt is None:
            units_ckpt = ensure_pretrained("contentvec/pytorch_model.bin")
        if rmvpe_ckpt is None:
            rmvpe_ckpt = ensure_pretrained("rmvpe/model.pt")

        from .contentvec import contentvec_from_hf_ckpt
        self.cvec = contentvec_from_hf_ckpt(str(units_ckpt))
        self.cvec.eval()

        from .rmvpe import RMVPE
        self.rmvpe = RMVPE.from_ckpt(str(rmvpe_ckpt))
        self.rmvpe.eval()

        self.fcpe = None
        self._fcpe_ckpt = fcpe_ckpt  # FCPE 惰性加载(f0_method='fcpe' 时)

    def extract(self, wav_path: str | Path | None = None, f0_method: str = "rmvpe",
                *, wav: np.ndarray | None = None) -> dict:
        assert f0_method in ("rmvpe", "fcpe"), f"unsupported f0 method: {f0_method}"

        if wav is None:
            assert wav_path is not None, "either wav_path or wav must be given"
            from ...io.audio import load_audio
            wav = load_audio(wav_path, sr=self.sr)  # [L] float32

        units = self._extract_units(wav)  # [T, 768]
        mx.clear_cache()  # ContentVec 中间结果清理,避免与 RMVPE 峰值叠加
        f0 = self._extract_f0(wav, f0_method=f0_method)  # [T]
        rms = self._extract_rms(wav)  # [T]
        uv = (f0 > 0).astype(np.float32)

        T = units.shape[0]  # trim 到公共帧数(与 torch 一致)
        if f0.shape[0] != T:
            f0, uv = f0[:T], uv[:T]
        if rms.shape[0] != T:
            rms = rms[:T]
        return {"units": units, "f0": f0, "rms": rms, "uv": uv,
                "wav": wav, "sr": self.sr}

    def _extract_units(self, wav: np.ndarray) -> np.ndarray:
        """wav [L]@sr → units [T_target, 768](16k 重采样 → encode layer12 →
        floor-gather 映射,等价 torch UnitsExtractor._gather_align)。"""
        import librosa
        if self.sr != 16000:
            wav16k = librosa.resample(wav, orig_sr=self.sr, target_sr=16000,
                                      res_type="soxr_hq").astype(np.float32)
        else:
            wav16k = wav
        hidden = self.cvec.encode(mx.array(wav16k)[None, :, None],
                                  layer=self.units_layer)  # [1, T_native, 768]
        ratio = (self.hop / self.sr) / (320.0 / 16000.0)
        T_target = max(1, (int(round(len(wav16k) * self.sr / 16000.0)) - self.hop) // self.hop + 1)
        idx = mx.clip(mx.floor(ratio * mx.arange(T_target)).astype(mx.int32),
                      0, hidden.shape[1] - 1)
        return np.array(hidden[0][idx], dtype=np.float32)

    def _extract_f0(self, wav: np.ndarray, f0_method: str = "rmvpe") -> np.ndarray:
        if f0_method == "fcpe":
            if self.fcpe is None:
                from ...hub import ensure_pretrained
                ckpt = self._fcpe_ckpt or ensure_pretrained("fcpe/fcpe.pt")
                from .fcpe import FCPE
                self.fcpe = FCPE.from_ckpt(str(ckpt))
                self.fcpe.eval()
            f0_native = np.asarray(self.fcpe.infer_from_audio(mx.array(wav),
                                                              sample_rate=self.sr),
                                   dtype=np.float32)
        else:
            f0_native = np.asarray(self.rmvpe.infer_from_audio(mx.array(wav),
                                                               sample_rate=self.sr),
                                   dtype=np.float32)
        if self.uv_policy == "interp":
            f0_native = self._interp_unvoiced(f0_native)
        return self._align_to_hop(f0_native, max(1, (len(wav) - self.hop) // self.hop + 1))

    @staticmethod
    def _interp_unvoiced(f0: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0, dtype=np.float32).copy()
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        return f0

    def _align_to_hop(self, f0_native: np.ndarray, n_frames: int) -> np.ndarray:
        if len(f0_native) == 0:
            return np.zeros(n_frames, dtype=np.float32)
        t0 = np.arange(len(f0_native), dtype=np.float32) / _NATIVE_FRAME_RATE
        tt = (self.hop / self.sr) * np.arange(n_frames, dtype=np.float32)
        return np.interp(tt, t0, f0_native).astype(np.float32)

    def _extract_rms(self, wav: np.ndarray) -> np.ndarray:
        from .mel import nvstft_pad
        win = self.rms_win_length
        padded = nvstft_pad(mx.array(wav.astype(np.float32)), win, self.hop)
        n_frames = max(1, (len(wav) - self.hop) // self.hop + 1)
        frames = mx.stack([padded[i * self.hop: i * self.hop + win]
                           for i in range(n_frames)])
        return np.array(mx.sqrt((frames * frames).mean(axis=1) + 1e-9), dtype=np.float32)