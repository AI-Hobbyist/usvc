"""Units (content encoder) extractor (DSP-001 §4.4): HubertSoft (256-d) or
ContentVec (768-d) via pure-torch Hubert; gather-aligned to dataset hop frames
(floor mapping = so-vits repeat_expand_2d_left)."""
from __future__ import annotations

import torch

from ..io.audio import frame_len_for
from ..data.feature_config import FeatureConfig
from .cvec.hubert_torch import HubertSoft

_HUBERT_CACHE = {}


class UnitsExtractor:
    _DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(self, fc, sr, hop, device="cpu", *, dtype="fp32"):
        assert fc.kind == "units"
        p = fc.params
        self.fc, self.sr, self.hop, self.device = fc, sr, hop, device
        self.dtype = self._DTYPE_MAP.get(dtype, torch.float32)
        self.enc_kind = p["kind"]  # hubertsoft | contentvec | contentvec_tta2x | contentvec_proj
        self.layer = int(p.get("layer", 12))
        self.use_proj = p.get("proj", "none") == "final_proj"
        self.tta_2x = bool(p.get("tta_2x", False))
        self.sr_in, self.hop_native = int(p.get("sr_in", 16000)), int(p.get("hop_native", 320))
        self.align_mode = p.get("align_mode", "gather")
        from ..hub import ensure_pretrained
        self.ckpt = ensure_pretrained(p["model_ckpt"])
        self.model = self._load_model()

    def _load_model(self):
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        key = (self.enc_kind, str(self.ckpt))
        cached = _HUBERT_CACHE.get(key)
        if cached is not None:
            return cached.to(device=self.device, dtype=self.dtype)
        if self.enc_kind == "hubertsoft":
            m = HubertSoft()
            ckpt = torch.load(self.ckpt, map_location="cpu")
            consume_prefix_in_state_dict_if_present(ckpt, "module.")
            m.load_state_dict(ckpt)
            m.eval()
        else:
            from .cvec.contentvec import contentvec_from_hf_ckpt
            m, miss, unexp = contentvec_from_hf_ckpt(self.ckpt)
            if miss or unexp:
                raise RuntimeError(f"ContentVec ckpt key mismatch: missing={miss[:5]} unexpected={unexp[:5]}")
        _HUBERT_CACHE[key] = m
        return m.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def extract(self, wav16k):  # 1d 16k wav -> [T_out, dim]; gather pre-aligns to hop frames
        wav16k = wav16k.to(device=self.device, dtype=self.dtype)
        wav16k_len = wav16k.shape[-1]
        if wav16k.dim() == 1:
            wav16k = wav16k.unsqueeze(0)
        if wav16k.dim() == 2:
            wav16k = wav16k.unsqueeze(1)  # conv0 expects channel dim
        if self.enc_kind == "hubertsoft":
            units = self.model.units(wav16k)
        else:
            if self.tta_2x:  # RIFT-style TTA: original + 160-sample-shifted clip in one batched call
                shifted = torch.roll(wav16k, shifts=-160, dims=-1)
                shifted[:, -160:] = 0
                hidden_both = self.model.encode(torch.cat([wav16k, shifted], dim=0), layer=self.layer if self.layer else None)
                hidden, hidden2 = hidden_both[0:1], hidden_both[1:2]
            else:
                hidden = self.model.encode(wav16k, layer=self.layer if self.layer else None)
            units = self.model.final_proj(hidden) if self.use_proj else hidden  # RVC 3_feature256 if proj
            if self.tta_2x:
                n, d = units.shape[1], units.shape[2]
                if self.use_proj:
                    hidden2 = self.model.final_proj(hidden2)
                units = torch.stack([units, hidden2], dim=2).view(1, n * 2, d)
        units = units.squeeze(0)
        if self.align_mode == "gather":
            # dataset-hop frame index -> native encoder frame (floor, so-vits semantic);
            # T_target derives from dataset-sr length, not the 16k input length
            ratio = (self.hop / self.sr) / (self.hop_native / self.sr_in)
            T_target = frame_len_for(int(round(wav16k_len * self.sr / self.sr_in)), self.hop)
            idx = torch.clamp(torch.floor(ratio * torch.arange(T_target, device=units.device)).long(), max=units.shape[0] - 1)
            units = units[idx]
        return units.float()