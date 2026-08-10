"""MLX 版特征提取(ContentVec units + RMVPE f0 + RMS)。

替代 ``usvc.infer.features.InferFeatures`` 的 torch 实现:所有提取
(units / f0 / rms)都在 MLX GPU 上完成,消除特征提取的 torch 依赖。

对齐 torch 版语义(数值已验证):
  - units:ContentVec layer12 → [T_native, 768],gather 对齐到 hop 帧
    (floor 映射,等价 so-vits repeat_expand_2d_left)
  - f0:mlx RMVPE(@sr,内部重采样 16k/160hop)→ 可选 uv interp →
    线性插值到 hop 帧
  - rms:nvSTFT center=False 网格(max(1, (L-hop)//hop+1) 帧)

构造参数(sr/hop/units_ckpt/rmvpe_ckpt)与 backend feature_spec 对齐;
调用方(mlx_engine)从 backend config 解析。
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

__all__ = ["MlxFeatureExtractor"]

# RMVPE 原生帧率(16k / hop160 = 100Hz),与 torch F0Extractor._NATIVE_FRAME_RATE 一致
_NATIVE_FRAME_RATE = 100.0


class MlxFeatureExtractor:
    """MLX 特征提取器(units + f0 + rms,全 GPU)。

    Args:
        sr: 后端采样率(如 RVC 48000 / so-vits 44100)。
        hop: 帧 hop(如 RVC 480 / so-vits 512)。
        units_ckpt: ContentVec HF ckpt 路径。
        rmvpe_ckpt: RMVPE ckpt 路径。
        device: mlx 设备("gpu"/"cpu")。
        rms_win_length: RMS 窗长(默认 hop;RVC=2048,so-vits=512)。
        uv_policy: "raw" | "interp"(浊音帧填充,默认 interp)。
        units_layer: ContentVec 输出层(默认 12)。
    """

    def __init__(
        self,
        sr: int,
        hop: int,
        units_ckpt: str | Path | None = None,
        rmvpe_ckpt: str | Path | None = None,
        fcpe_ckpt: str | Path | None = None,
        device: str = "gpu",
        rms_win_length: int | None = None,
        uv_policy: str = "interp",
        units_layer: int = 12,
    ):
        if device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            mx.set_default_device(mx.cpu)
        self.sr = int(sr)
        self.hop = int(hop)
        self.device = device
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

        # FCPE(情性加载,仅在 f0_method='fcpe' 时)
        self.fcpe = None
        self._fcpe_ckpt = fcpe_ckpt

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def extract(
        self,
        wav_path: str | Path | None = None,
        f0_method: str = "rmvpe",
        *,
        wav: np.ndarray | None = None,
    ) -> dict:
        """提取特征,接口对齐 torch ``InferFeatures.extract``。

        Args:
            wav_path: 输入音频文件路径。
            f0_method: "rmvpe"(mlx 已实现)或 "fcpe"(仅 torch,mlx 未移植)。
            wav: 可选预加载 float32 音频(@self.sr);提供时忽略 wav_path。

        Returns:
            dict:
                units: [T, dim] float32(gather 对齐到 hop 帧)
                f0: [T] float32(Hz,0 = 无声)
                rms: [T] float32
                uv: [T] float32(浊音标志,f0>0)
                wav: [L] float32 numpy(@self.sr)
                sr: int
        """
        assert f0_method in ("rmvpe", "fcpe"), f"unsupported f0 method: {f0_method}"

        if wav is None:
            assert wav_path is not None, "either wav_path or wav must be given"
            from ...io.audio import load_audio
            wav = load_audio(wav_path, sr=self.sr)  # [L] float32

        # units:重采样到 16k → mlx ContentVec → gather 对齐
        units = self._extract_units(wav)          # [T, 768]
        # ContentVec 的中间结果不再需要,清理避免与 RMVPE 峰值叠加
        import mlx.core as _mx
        _mx.clear_cache()
        # f0:mlx RMVPE/FCPE(按 f0_method 选择)→ uv interp → hop 对齐
        f0 = self._extract_f0(wav, f0_method=f0_method)                # [T]
        # rms:nvSTFT grid
        rms = self._extract_rms(wav)              # [T]
        uv = (f0 > 0).astype(np.float32)

        # Trim 到公共帧数(与 torch InferFeatures.extract 一致)
        T = units.shape[0]
        if f0.shape[0] != T:
            f0 = f0[:T]
            uv = uv[:T]
        if rms.shape[0] != T:
            rms = rms[:T]

        return {
            "units": units,
            "f0": f0,
            "rms": rms,
            "uv": uv,
            "wav": wav,
            "sr": self.sr,
        }

    # ------------------------------------------------------------------
    # units(ContentVec layer12 + gather 对齐)
    # ------------------------------------------------------------------

    def _extract_units(self, wav: np.ndarray) -> np.ndarray:
        """wav [L]@sr → units [T_target, 768],gather 对齐到 hop 帧。

        复现 torch ``UnitsExtractor.extract``(contentvec 分支):
          1. 重采样 16k(soxr_hq)
          2. encode(layer=12) → [T_native, 768]
          3. gather_align:floor 映射到 hop 帧
        """
        import librosa
        if self.sr != 16000:
            wav16k = librosa.resample(
                wav, orig_sr=self.sr, target_sr=16000, res_type="soxr_hq"
            ).astype(np.float32)
        else:
            wav16k = wav
        x = mx.array(wav16k)[None, :, None]          # [1, T, 1] (NLC)
        hidden = self.cvec.encode(x, layer=self.units_layer)  # [1, T_native, 768]
        units = hidden[0]                            # [T_native, 768]

        # gather_align(镜像 torch UnitsExtractor._gather_align)
        ratio = (self.hop / self.sr) / (320.0 / 16000.0)
        wav_ds_len = int(round(len(wav16k) * self.sr / 16000.0))
        T_target = max(1, (wav_ds_len - self.hop) // self.hop + 1)
        T_native = units.shape[0]
        idx = mx.clip(
            mx.floor(ratio * mx.arange(T_target)).astype(mx.int32),
            0, T_native - 1,
        )
        units = units[idx]                           # [T_target, 768]
        return np.array(units, dtype=np.float32)

    # ------------------------------------------------------------------
    # f0(RMVPE + uv interp + hop 对齐)
    # ------------------------------------------------------------------

    def _extract_f0(self, wav: np.ndarray, f0_method: str = "rmvpe") -> np.ndarray:
        """wav [L]@sr → f0 [T](hop 帧,0=无声)。

        f0_method='rmvpe'(默认)或 'fcpe'。"""
        if f0_method == "fcpe":
            if self.fcpe is None:
                from ...hub import ensure_pretrained
                ckpt = self._fcpe_ckpt or ensure_pretrained("fcpe/fcpe.pt")
                from .fcpe import FCPE
                self.fcpe = FCPE.from_ckpt(str(ckpt))
                self.fcpe.eval()
            f0_native = np.asarray(
                self.fcpe.infer_from_audio(mx.array(wav), sample_rate=self.sr),
                dtype=np.float32,
            )
        else:
            f0_native = np.asarray(
                self.rmvpe.infer_from_audio(mx.array(wav), sample_rate=self.sr),
                dtype=np.float32,
            )
        if self.uv_policy == "interp":
            f0_native = self._interp_unvoiced(f0_native)

        n_frames = max(1, (len(wav) - self.hop) // self.hop + 1)
        return self._align_to_hop(f0_native, n_frames)

    @staticmethod
    def _interp_unvoiced(f0: np.ndarray) -> np.ndarray:
        """线性填充无声帧(等价 torch F0Extractor._interp_unvoiced)。"""
        f0 = np.asarray(f0, dtype=np.float32).copy()
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        return f0

    def _align_to_hop(self, f0_native: np.ndarray, n_frames: int) -> np.ndarray:
        """~100Hz 原生帧 → hop 帧(线性插值,等价 torch _align_to_hop)。"""
        if len(f0_native) == 0:
            return np.zeros(n_frames, dtype=np.float32)
        target_period = self.hop / self.sr
        origin_time = np.arange(len(f0_native), dtype=np.float32) / _NATIVE_FRAME_RATE
        target_time = target_period * np.arange(n_frames, dtype=np.float32)
        return np.interp(target_time, origin_time, f0_native).astype(np.float32)

    # ------------------------------------------------------------------
    # rms(nvSTFT center=False 网格)
    # ------------------------------------------------------------------

    def _extract_rms(self, wav: np.ndarray) -> np.ndarray:
        """wav [L]@sr → rms [T](等价 torch RmsExtractor.extract)。

        nvSTFT center=False padding + unfold 分帧 + RMS。
        """
        from .mel import nvstft_pad
        wav_mx = mx.array(wav.astype(np.float32))
        win = self.rms_win_length
        padded = nvstft_pad(wav_mx, win, self.hop)     # [L + pad_left + pad_right]
        n = len(wav)
        n_frames = max(1, (n - self.hop) // self.hop + 1)
        # 手动 unfold(mlx 无 unfold):[n_frames, win]
        frames = mx.stack(
            [padded[i * self.hop : i * self.hop + win] for i in range(n_frames)]
        )
        rms = mx.sqrt((frames * frames).mean(axis=1) + 1e-9)
        return np.array(rms, dtype=np.float32)
