"""F0 extractor facade (DSP-001 §4.2).

Supports RMVPE and FCPE. Output is aligned to the dataset hop frame rate
(T = len//hop + 1), matching mel's frame count for length-consistent slicing.
UV policy is "raw" by default (f0==0 at unvoiced); backends needing interpolated
f0 do it online.

Both extractors natively output ~100Hz frames; we linearly interpolate to the
dataset hop rate (same alignment logic regardless of extractor).
"""
from __future__ import annotations

import numpy as np
import torch

from ..io.audio import frame_len_for
from ..data.feature_config import FeatureConfig
from .rmvpe import RMVPE

__all__ = ["F0Extractor"]

_RMVPE_CACHE: dict[str, RMVPE] = {}
_FCPE_CACHE: dict[str, object] = {}
# Both RMVPE (16k/160hop) and FCPE (native 100Hz) output at ~100Hz frame rate.
_NATIVE_FRAME_RATE = 100.0


def _get_rmvpe(ckpt: str, device: str, dtype: torch.dtype = torch.float32) -> RMVPE:
    key = str(ckpt)
    if key not in _RMVPE_CACHE:
        _RMVPE_CACHE[key] = RMVPE(ckpt, hop_length=160, device=device, dtype=dtype)
    m = _RMVPE_CACHE[key]
    if m.device != device or m.dtype != dtype:
        m.model = m.model.to(device=device, dtype=dtype)
        m.mel_extractor = m.mel_extractor.to(device=device)  # keep FP32
        m.device = device
        m.dtype = dtype
    return m


def _get_fcpe(device: str, dtype: torch.dtype = torch.float32):
    # FCPE uses the bundled weight shipped under usvc/features/fcpe/assets/.
    key = ("bundled", device, str(dtype))
    if key not in _FCPE_CACHE:
        from .fcpe.models_infer import spawn_bundled_infer_model

        _FCPE_CACHE[key] = spawn_bundled_infer_model(device=device, dtype=dtype)
    m = _FCPE_CACHE[key]
    # Move cached model to requested device.
    try:
        marker_dev = str(m.tensor_device_marker.device).split(":", 1)[0]
    except Exception:
        marker_dev = "cpu"
    if marker_dev != device:
        m.to(device)
        m.tensor_device_marker = m.tensor_device_marker.to(device)
    return m


class F0Extractor:
    _DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(
        self, fc: FeatureConfig, sr: int, hop: int, device: str = "cpu", *, dtype: str = "fp32"
    ) -> None:
        assert fc.kind == "f0"
        self.extractor = fc.params.get("extractor", "rmvpe")
        assert self.extractor in ("rmvpe", "fcpe"), f"unsupported f0 extractor: {self.extractor}"
        self.fc = fc
        self.sr = sr
        self.hop = hop
        self.device = device
        self.dtype = self._DTYPE_MAP.get(dtype, torch.float32)
        self.thred = float(fc.params.get("thred", 0.03))
        self.use_viterbi = bool(fc.params.get("use_viterbi", False))
        # FCPE-specific
        self.fcpe_threshold = float(fc.params.get("fcpe_threshold", 0.006))
        self.fcpe_decoder = str(fc.params.get("fcpe_decoder", "local_argmax"))

        if self.extractor == "rmvpe":
            ckpt = fc.params.get("model_ckpt", "pretrained/rmvpe/model.pt")
            self.rmvpe = _get_rmvpe(ckpt, device, self.dtype)
        else:  # fcpe — keep FP32 (its mel sanity-check rejects fp16 inputs)
            self.fcpe = _get_fcpe(device, torch.float32)

    @torch.inference_mode()
    def extract(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Returns [T] float32, T = len//hop + 1, aligned to dataset hop."""
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        wav = wav.float().cpu()
        n_frames = frame_len_for(wav.shape[0], self.hop)

        if self.extractor == "rmvpe":
            f0_native = self.rmvpe.infer_from_audio(
                wav, sample_rate=self.sr, device=self.device, thred=self.thred, use_viterbi=self.use_viterbi
            )
            f0_native = np.asarray(f0_native, dtype=np.float32)
        else:  # fcpe
            wav_t = wav.to(self.device).unsqueeze(0)  # [1, T]
            f0_out = self.fcpe(wav_t, sr=self.sr, decoder_mode=self.fcpe_decoder, threshold=self.fcpe_threshold)
            # FCPE output: [1, T_native, 1] -> [T_native]
            f0_native = f0_out.float().squeeze().cpu().numpy().astype(np.float32)

        return self._align_to_hop(f0_native, n_frames)

    @torch.inference_mode()
    def extract_batch(self, wavs: list) -> list[torch.Tensor]:
        """Batched extract. FCPE path batches natively; RMVPE falls back to loop."""
        if self.extractor == "rmvpe":
            return [self.extract(w) for w in wavs]

        # FCPE batched path.
        from .batching import pad_batch

        wavs_t = [torch.from_numpy(w).float() if not torch.is_tensor(w) else w.float() for w in wavs]
        n_frames_list = [frame_len_for(w.shape[0], self.hop) for w in wavs_t]
        padded, _lens = pad_batch([w.to(self.device) for w in wavs_t])  # [B, T]
        f0_b = self.fcpe(padded, sr=self.sr, decoder_mode=self.fcpe_decoder, threshold=self.fcpe_threshold)
        # f0_b: [B, T_native, 1]
        f0_b = f0_b.float().squeeze(-1).cpu().numpy().astype(np.float32)
        results = []
        for i, nf in enumerate(n_frames_list):
            # FCPE native frame count is proportional to input length; slice to actual
            # (padded region produced junk frames we discard).
            ratio = padded.shape[1] / f0_b.shape[1]
            actual_native = int(round(_lens[i].item() / ratio))
            f0_native = f0_b[i, :actual_native]
            results.append(self._align_to_hop(f0_native, nf))
        return results

    def _align_to_hop(self, f0_native: np.ndarray, n_frames: int) -> torch.Tensor:
        """Linear-interp native 100Hz frames to dataset hop frames."""
        if len(f0_native) == 0:
            return torch.zeros(n_frames, dtype=torch.float32)
        target_frame_period = self.hop / self.sr
        origin_time = np.arange(len(f0_native), dtype=np.float32) / _NATIVE_FRAME_RATE
        target_time = target_frame_period * np.arange(n_frames, dtype=np.float32)
        f0 = np.interp(target_time, origin_time, f0_native).astype(np.float32)
        return torch.from_numpy(f0)
