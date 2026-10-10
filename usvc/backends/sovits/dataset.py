# sovits dataset: units(hubert-soft/contentvec) + f0 + spec(linear) + wav + spk_id + uv。
# 骨架在 usvc/backends/_dataset.py（BaseDirDataset + crop/pad）。随机裁剪用每数据集
# 自己的 self._rng（seed 42，非全局 torch generator，与 RVC 一致）。
from __future__ import annotations

import torch

from .._dataset import BaseDirDataset, align_spec_to_frames, align_wav, slice_or_pad

__all__ = ["SoVITSDataset", "interp_unvoiced_f0"]


def interp_unvoiced_f0(f0):
    # 线性插值无声帧（so-vits/RVC 约定）。调用方必须先截取原始 voiced mask（uv=f0>0）：
    # sovits 的 emb_uv 按真实无声帧条件化，而 NSF 源用全浊插值 f0。
    # np.interp 逐位复刻（f64 预计算 slope=(y1-y0)/(x1-x0)，y=slope*(x-x0)+y0，末尾
    # 舍入回 f0 dtype；n=1 时全取首个浊音值）
    voiced = f0 > 0
    if not voiced.any(): return f0
    device, dtype = f0.device, f0.dtype
    f0f = f0.detach().float()
    x = torch.nonzero(~voiced, as_tuple=False).squeeze(-1).double()
    if x.numel() == 0: return f0  # 全浊
    xp = torch.nonzero(voiced, as_tuple=False).squeeze(-1).double()
    fp = f0f[xp.long()].double(); n = xp.numel()
    if n == 1:
        out = f0.clone(); out[~voiced] = fp[0].to(dtype=dtype); return out
    j = (torch.searchsorted(xp, x, right=True) - 1).clamp(-1, n)
    jj = j.clamp(0, n - 1); jc = j.clamp(0, n - 2)
    y = fp[jc] + (fp[jc + 1] - fp[jc]) / (xp[jc + 1] - xp[jc]) * (x - xp[jc])
    y = torch.where(x < xp[0], fp[0], y); y = torch.where(x > xp[-1], fp[-1], y)
    y = torch.where(j == n - 1, fp[n - 1], y)
    y = torch.where((j >= 0) & (j < n - 1) & (xp[jj] == x), fp[jj], y)
    out = f0.clone(); out[~voiced] = y.float().to(device=device, dtype=dtype)
    return out


class SoVITSDataset(BaseDirDataset):
    strict_speakers = True
    def __init__(self, dataset_dir, split, feature_hashes, max_frames=790, augment=True, encoder_kind="hubertsoft"):
        super().__init__(dataset_dir, split, feature_hashes, max_frames, augment, encoder_kind)
        self._units_dir = self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")  # 线性谱，可缺
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")  # vol_embedding 用，可缺
        self._audio_dir = self.dataset_dir / "audio"
        self.sr = self.meta.sr; self.hop = self.meta.hop

    def _load_one(self, key):
        spk, fn = key
        stem = fn.rsplit(".", 1)[0]
        pt_stem = stem + ".pt"; wav_stem = stem + ".wav"
        units = self._load_pt(self._units_dir / spk / pt_stem)
        f0 = self._load_pt(self._f0_dir / spk / pt_stem)
        spec_path = self._spec_dir / spk / pt_stem
        spec = self._load_pt(spec_path) if spec_path.exists() else None
        rms_path = self._rms_dir / spk / pt_stem
        rms = self._load_pt(rms_path) if rms_path.exists() else None
        wav = None
        wav_path = self._audio_dir / spk / wav_stem
        if wav_path.exists():
            import soundfile as sf
            wav = torch.from_numpy(sf.read(str(wav_path), dtype="float32")[0])
        # uv 必须在插值前截取（见 interp_unvoiced_f0 注释）
        uv = (f0 > 0).float()
        f0 = interp_unvoiced_f0(f0)
        if spec is not None and spec.ndim == 3: spec = spec[0]
        # spec 存为 [T, C]（frames first），转置成 [C, T] 后 base._crop 无法再按 units
        # 对齐（shape[0] != T），所以定长裁剪/补齐必须在本处、转置之前做
        mf, T = self.max_frames, units.shape[0]
        start = int(torch.randint(0, T - mf + 1, (1,), generator=self._rng).item()) if T > mf else None
        units, f0, uv = [slice_or_pad(t, mf, T, start) for t in (units, f0, uv)]
        if rms is not None: rms = slice_or_pad(rms, mf, T, start)
        if spec is not None: spec = slice_or_pad(spec, mf, T, start)
        # 帧数可能比 wav 采样多一帧（len//hop+1 约定）；GAN 波形容许
        # ids_slice*hop + segment <= T*hop
        if wav is not None: wav = align_wav(wav, mf * self.meta.hop)
        if spec is not None: spec = align_spec_to_frames(spec, mf)  # 强制 spec 帧数 = units 帧数
        return {"units": units, "f0": f0, "uv": uv, "mel": spec, "wav": wav, "rms": rms,
                "spk_id": self.spk2idx[spk]}
