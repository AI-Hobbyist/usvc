"""Shared scaffolding for DiffusionEnhancer / ReflowEnhancer (identical blocks, one copy)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.feature_config import FeatureConfig
from ..features.mel import MelExtractor
from ..features.units import UnitsExtractor


def tpad(x: torch.Tensor, T: int) -> torch.Tensor:
    """Trim to T frames / zero-pad the tail ([T,M] or [T,D] frames)."""
    x = x[:T]
    if x.shape[0] < T:
        x = F.pad(x, (0, 0, 0, T - x.shape[0]))
    return x


def reextract(ext, wav, T, sr, device):
    """Re-extract units from the enhancer-domain wav (16k resample, align to T)."""
    if ext is None:
        return None
    import numpy as np
    import librosa
    wav_np = wav if isinstance(wav, np.ndarray) else wav.cpu().numpy()
    w16 = librosa.resample(wav_np.astype(np.float32), orig_sr=sr, target_sr=16000)
    return tpad(ext.extract(torch.from_numpy(w16).float().to(device)), T)


def vol_batch(rms, mel, device):
    """[T] rms (or None, fallback: mel energy as in train) -> [1, T] volume."""
    if rms is not None:
        return rms.float().unsqueeze(0).to(device)
    return (mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)


def pack(units, f0, vol, spk_id, device, last=False):
    u = units.float().unsqueeze(0).to(device)
    f = f0.float().unsqueeze(0).to(device)
    if last:  # DDSP6 expects [B, T, 1]
        f, vol = f.unsqueeze(-1), vol.unsqueeze(-1)
    return u, f, vol, torch.tensor([spk_id], dtype=torch.long, device=device)


def mel_ext(mel_cfg, sr, hop, device, win=2048, n_mels=128, fmax=None):
    return MelExtractor(FeatureConfig("mel", {
        "n_fft": int(mel_cfg.get("n_fft", win)),
        "win_length": int(mel_cfg.get("win_length", win)),
        "n_mels": int(mel_cfg.get("n_mels", n_mels)),
        "fmin": float(mel_cfg.get("fmin", 40.0)),
        "fmax": None if mel_cfg.get("fmax") is None else float(mel_cfg.get("fmax", fmax)),
        "center": bool(mel_cfg.get("center", False)),
    }), sr=int(mel_cfg.get("sampling_rate", sr)), hop=int(mel_cfg.get("hop", hop))).to(device)


def units_ext(kind, units_ckpt, sr, hop, device):
    return UnitsExtractor(FeatureConfig("units", {
        "kind": kind, "sr_in": 16000, "hop_native": 320, "align_mode": "gather",
        "model_ckpt": units_ckpt,
    }), sr, hop, device, dtype=torch.float32)


def vocoder(NSFHiFiGAN, vocoder_dir, device, compile_gen=False):
    """NSF-HiFiGAN fp32 + channels_last generator; optionally compile it."""
    vdir = Path(vocoder_dir)
    v = NSFHiFiGAN(vdir / "config.json", vdir / "model.ckpt", device=device, is_half=False)
    if hasattr(v, "generator"):  # 测试用 _DummyVocoder 无此属性
        v.generator._use_cl = True  # conv1d->conv2d(1,k) 等价变换 (SNR>=64dB 已验证)
        if compile_gen:
            try:
                v.generator = torch.compile(v.generator, mode="reduce-overhead", dynamic=False)
            except Exception:
                pass
    return v