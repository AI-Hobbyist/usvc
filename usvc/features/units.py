"""Units (content encoder) extractor (DSP-001 §4.4).

Supports both HubertSoft (256-d, bshall) and ContentVec (768-d, HF-style),
physically isolated by encoder kind. All variants are loaded via pure-torch
Hubert (no transformers/fairseq). Outputs are aligned to dataset hop frame rate
via gather (DSP-001 §4.4.3 default align_mode=gather).
"""
from __future__ import annotations

import torch

from ..io.audio import frame_len_for
from ..data.feature_config import FeatureConfig
from .cvec.hubert_torch import HubertSoft

__all__ = ["UnitsExtractor"]

_HUBERT_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


class UnitsExtractor:
    _DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(
        self, fc: FeatureConfig, sr: int, hop: int, device: str = "cpu", *, dtype: str = "fp32"
    ) -> None:
        assert fc.kind == "units"
        self.fc = fc
        self.sr = sr
        self.hop = hop
        self.device = device
        self.dtype = self._DTYPE_MAP.get(dtype, torch.float32)
        p = fc.params
        self.enc_kind: str = p["kind"]  # hubertsoft | contentvec | contentvec_tta2x | contentvec_proj
        self.layer: int = int(p.get("layer", 12))
        self.use_proj: bool = p.get("proj", "none") == "final_proj"
        self.tta_2x: bool = bool(p.get("tta_2x", False))
        self.sr_in: int = int(p.get("sr_in", 16000))
        self.hop_native: int = int(p.get("hop_native", 320))
        self.align_mode: str = p.get("align_mode", "gather")
        self.ckpt: str = p["model_ckpt"]
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

        key = (self.enc_kind, str(self.ckpt))
        cached = _HUBERT_CACHE.get(key)
        if cached is not None:
            return cached.to(device=self.device, dtype=self.dtype)

        if self.enc_kind == "hubertsoft":
            m: torch.nn.Module = HubertSoft()
            ckpt = torch.load(self.ckpt, map_location="cpu")
            consume_prefix_in_state_dict_if_present(ckpt, "module.")
            m.load_state_dict(ckpt)
            m.eval()
        else:
            # ContentVec family: pure-torch ContentVec (HF wav2vec2 structure).
            # Loads HF pytorch_model.bin with strict key match, zero remapping.
            from .cvec.contentvec import contentvec_from_hf_ckpt
            m, miss, unexp = contentvec_from_hf_ckpt(self.ckpt)
            if miss or unexp:
                raise RuntimeError(
                    f"ContentVec ckpt key mismatch: missing={miss[:5]} unexpected={unexp[:5]}"
                )
        _HUBERT_CACHE[key] = m
        return m.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def extract(self, wav16k: torch.Tensor) -> torch.Tensor:
        """wav16k: 1d float tensor at 16kHz. Returns [T_out, dim].

        ``align_mode='gather'`` (default) pre-aligns to dataset hop frames so the
        output length matches mel/f0/rms. ``align_mode='interp'`` returns native
        frames for the dataset to interpolate online.
        """
        wav16k = wav16k.to(device=self.device, dtype=self.dtype)
        wav16k_len = wav16k.shape[-1]
        if wav16k.dim() == 1:
            wav16k = wav16k.unsqueeze(0)  # [1, T]
        if wav16k.dim() == 2:
            wav16k = wav16k.unsqueeze(1)  # [1, 1, T] — conv0 expects channel dim

        if self.enc_kind == "hubertsoft":
            units = self.model.units(wav16k)  # [1, T_native, 256]
        else:
            # ContentVec family: pure-torch impl takes [B,1,T], handles padding internally.
            hidden = self.model.encode(wav16k, layer=self.layer if self.layer else None)  # [1, T, 768]
            if self.use_proj:
                # RVC's 3_feature256/768: ContentVec final_proj (768 -> 256).
                units = self.model.final_proj(hidden)  # [1, T, 256]
            else:
                units = hidden  # [1, T_native, 768]

            if self.tta_2x:
                # TTA 2x (RIFT-style): shift by 160 samples (half native hop), zeroing
                # the wrapped tail so length is preserved -> both forwards match.
                shifted = torch.roll(wav16k, shifts=-160, dims=-1)
                shifted[:, -160:] = 0
                hidden2 = self.model.encode(shifted, layer=self.layer if self.layer else None)
                n, d = units.shape[1], units.shape[2]
                units = torch.stack([units, hidden2], dim=2).view(1, n * 2, d)

        units = units.squeeze(0)  # [T_native, dim]

        if self.align_mode == "gather":
            units = self._gather_align(units, wav16k_len)
        return units.float()

    def _gather_align(self, units: torch.Tensor, wav16k_len: int) -> torch.Tensor:
        """Align native encoder frames to dataset hop frames (DSP-001 §4.4.3).

        T_target must equal mel/f0/rms frame count, which is derived from the
        audio length at the dataset sample rate (not the 16kHz encoder input rate).
        We convert wav16k_len back to the equivalent dataset-sr length.
        Uses floor mapping to match so-vits-svc repeat_expand_2d_left.
        """
        # Map dataset hop frame index -> native encoder frame index.
        ratio = (self.hop / self.sr) / (self.hop_native / self.sr_in)
        # Audio length at dataset sr (mel/f0/rms frame basis).
        wav_ds_len = int(round(wav16k_len * self.sr / self.sr_in))
        T_target = frame_len_for(wav_ds_len, self.hop)  # same formula as mel/f0
        T_native = units.shape[0]
        # floor mapping (matches so-vits repeat_expand_2d_left: src[floor(i*ratio)])
        idx = torch.clamp(
            torch.floor(ratio * torch.arange(T_target, device=units.device)).long(),
            max=T_native - 1,
        )
        return units[idx]
