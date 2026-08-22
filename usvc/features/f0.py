"""F0 extractor facade (DSP-001 §4.2): RMVPE or FCPE, output aligned to the
dataset hop frame rate (T = len//hop + 1, same grid as mel).

uv_policy: "raw" keeps f0==0 unvoiced (rift_dit/sovits/reflow_*); "interp"
linear-fills unvoiced frames on the native ~100Hz grid before hop resampling
(rvc/ddsp6/diffusion). Both extractors natively output ~100Hz frames.
"""
from __future__ import annotations

import numpy as np
import torch

from ..io.audio import frame_len_for
from ..data.feature_config import FeatureConfig
from .rmvpe import RMVPE

_RMVPE_CACHE: dict = {}
_FCPE_CACHE: dict = {}
_NATIVE_FRAME_RATE = 100.0


def _get_rmvpe(ckpt, device, dtype=torch.float32):
    key = str(ckpt)
    if key not in _RMVPE_CACHE:
        _RMVPE_CACHE[key] = RMVPE(ckpt, hop_length=160, device=device, dtype=dtype)
    m = _RMVPE_CACHE[key]
    if m.device != device or m.dtype != dtype:  # move cached model; mel stays FP32
        m.model = m.model.to(device=device, dtype=dtype)
        m.mel_extractor = m.mel_extractor.to(device=device)
        m.device, m.dtype = device, dtype
    return m


def _get_fcpe(device, dtype=torch.float32, ckpt=None):
    key = (ckpt or "bundled", device, str(dtype))
    if key not in _FCPE_CACHE:
        from .fcpe.models_infer import spawn_bundled_infer_model, spawn_infer_model_from_pt
        if ckpt:
            _FCPE_CACHE[key] = spawn_infer_model_from_pt(ckpt, device=device, dtype=dtype)
        else:
            _FCPE_CACHE[key] = spawn_bundled_infer_model(device=device, dtype=dtype)
    m = _FCPE_CACHE[key]
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

    def __init__(self, fc, sr, hop, device="cpu", *, dtype="fp32"):
        assert fc.kind == "f0"
        self.extractor = fc.params.get("extractor", "rmvpe")
        assert self.extractor in ("rmvpe", "fcpe"), f"unsupported f0 extractor: {self.extractor}"
        self.fc, self.sr, self.hop, self.device = fc, sr, hop, device
        self.dtype = self._DTYPE_MAP.get(dtype, torch.float32)
        self.thred = float(fc.params.get("thred", 0.03))
        self.use_viterbi = bool(fc.params.get("use_viterbi", False))
        self.fcpe_threshold = float(fc.params.get("fcpe_threshold", 0.006))
        self.fcpe_decoder = str(fc.params.get("fcpe_decoder", "local_argmax"))
        self.uv_policy = fc.params.get("uv_policy", "raw")
        assert self.uv_policy in ("raw", "interp"), f"unsupported uv_policy: {self.uv_policy}"
        from ..hub import ensure_pretrained
        ckpt = ensure_pretrained(fc.params.get("model_ckpt", "pretrained/rmvpe/model.pt" if self.extractor == "rmvpe" else "pretrained/fcpe/fcpe.pt"))
        if self.extractor == "rmvpe":
            self.rmvpe = _get_rmvpe(ckpt, device, self.dtype)
        else:  # fcpe — keep FP32 (its mel sanity-check rejects fp16 anyway)
            self.fcpe = _get_fcpe(device, torch.float32, ckpt=ckpt)

    @torch.inference_mode()
    def extract(self, wav):
        """[T] float32, T = len//hop + 1, aligned to dataset hop."""
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        wav = wav.float().cpu()
        n_frames = frame_len_for(wav.shape[0], self.hop)
        if self.extractor == "rmvpe":
            f0_native = np.asarray(self.rmvpe.infer_from_audio(wav, sample_rate=self.sr, device=self.device, thred=self.thred, use_viterbi=self.use_viterbi), dtype=np.float32)
        else:
            f0_native = self.fcpe(wav.to(self.device).unsqueeze(0), sr=self.sr, decoder_mode=self.fcpe_decoder, threshold=self.fcpe_threshold).float().squeeze().cpu().numpy().astype(np.float32)
        if self.uv_policy == "interp":
            f0_native = self._interp_unvoiced(f0_native)
        return self._align_to_hop(f0_native, n_frames)

    @torch.inference_mode()
    def extract_batch(self, wavs):
        if self.extractor == "rmvpe":
            return [self.extract(w) for w in wavs]
        from .batching import pad_batch
        wavs_t = [torch.from_numpy(w).float() if not torch.is_tensor(w) else w.float() for w in wavs]
        n_frames_list = [frame_len_for(w.shape[0], self.hop) for w in wavs_t]
        padded, _lens = pad_batch([w.to(self.device) for w in wavs_t])
        f0_b = self.fcpe(padded, sr=self.sr, decoder_mode=self.fcpe_decoder, threshold=self.fcpe_threshold).float().squeeze(-1).cpu().numpy().astype(np.float32)
        results = []
        for i, nf in enumerate(n_frames_list):
            # native frames scale with input length; drop the junk padded frames
            f0_native = f0_b[i, :int(round(_lens[i].item() / (padded.shape[1] / f0_b.shape[1])))]
            if self.uv_policy == "interp":
                f0_native = self._interp_unvoiced(f0_native)
            results.append(self._align_to_hop(f0_native, nf))
        return results

    def _interp_unvoiced(self, f0_native):
        # RVC/DDSP-SVC preprocess: uv = f0==0; f0[uv] = np.interp(where(uv), where(~uv), f0[~uv])
        f0 = np.asarray(f0_native, dtype=np.float32).copy()
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        return f0

    def _align_to_hop(self, f0_native, n_frames):
        """Linear-interp native ~100Hz grid to dataset hop grid."""
        if len(f0_native) == 0:
            return torch.zeros(n_frames, dtype=torch.float32)
        target_period = self.hop / self.sr
        f0 = np.interp(target_period * np.arange(n_frames, dtype=np.float32), np.arange(len(f0_native), dtype=np.float32) / _NATIVE_FRAME_RATE, f0_native).astype(np.float32)
        return torch.from_numpy(f0)