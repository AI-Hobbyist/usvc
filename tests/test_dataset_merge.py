"""E1 anchors: merged datasets are bit-identical to the per-backend originals.

Each reference class below embeds the ORIGINAL pre-merge __init__/_load_one
code (verbatim from git HEAD before the E1 dedup) so the shared kernel in
usvc/backends/_dataset.py is pinned to the old behavior:
  - mel quartet (ddsp6/diffusion/reflow_shallow/rift_dit + reflow_vae alias)
  - rvc / sovits fixed-length crop/pad + spec/wav alignment (with their own
    random sources preserved: rvc = global default generator, sovits =
    self._rng seed 42).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch

from usvc.backends.ddsp6.dataset import DDSPDataset
from usvc.backends.diffusion.dataset import DiffusionDataset
from usvc.backends.reflow_shallow.dataset import ReflowShallowDataset
from usvc.backends.reflow_vae.dataset import ReflowVAEDataset
from usvc.backends.rift_dit.dataset import RIFTDataset
from usvc.backends.rvc.dataset import RVCDataset
from usvc.backends.sovits.dataset import SoVITSDataset, interp_unvoiced_f0
from usvc.data.meta import AudioEntry, init_dataset_meta, load_meta
from usvc.train.dataset import BaseSVCDataset

HASHS = {"mel": "m1", "f0": "f1", "rms": "r1", "units": "u1",
         "aug_mel": "am1", "aug_shift": "as1", "spec": "s1"}


# ---------------------------------------------------------------------------
# fixture: synthetic preprocessed dataset dir
# ---------------------------------------------------------------------------

def _write_pt(path: Path, t: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)


def _make_dataset(tmp_path: Path, *, with_wav: bool = True) -> Path:
    """meta.json + mel/f0/rms/units(/spec/rms/aug/audio) .pt files.

    a1: long clip (T=1000, exercises random crop), b1: short clip (T=100,
    exercises padding for the GAN datasets; returned as-is for mel quartet).
    """
    meta = init_dataset_meta(tmp_path, sr=44100, hop=512, f0_min=40.0, f0_max=16000.0)
    meta.speakers = ["spk_a", "spk_b"]
    meta.audios = [
        AudioEntry("spk_a", "a1.wav", 1000, 11.0, 512 * 1000 + 100),
        AudioEntry("spk_b", "b1.wav", 100, 1.1, 512 * 100 + 50),
    ]
    meta.splits = {"train": ["a1.wav", "b1.wav"], "val": []}
    meta.save()

    torch.manual_seed(0)
    for spk, stem, T in (("spk_a", "a1", 1000), ("spk_b", "b1", 100)):
        _write_pt(tmp_path / "mel" / "m1" / spk / f"{stem}.pt", torch.rand(T, 128))
        _write_pt(tmp_path / "f0" / "f1" / spk / f"{stem}.pt", torch.rand(T) * 400)
        _write_pt(tmp_path / "rms" / "r1" / spk / f"{stem}.pt", torch.rand(T) * 0.1)
        _write_pt(tmp_path / "units" / "contentvec_tta2x" / "u1" / spk / f"{stem}.pt",
                  torch.rand(T, 768))
        _write_pt(tmp_path / "units" / "contentvec" / "u1" / spk / f"{stem}.pt",
                  torch.rand(T, 768))
        _write_pt(tmp_path / "units" / "hubertsoft" / "u1" / spk / f"{stem}.pt",
                  torch.rand(T, 768))
        _write_pt(tmp_path / "spec" / "s1" / spk / f"{stem}.pt", torch.rand(T, 1025))
        _write_pt(tmp_path / "aug_mel" / "am1" / spk / f"{stem}.pt", torch.rand(T, 128))
        _write_pt(tmp_path / "aug_shift" / "as1" / spk / f"{stem}.pt", torch.tensor([-3.0]))
        if with_wav:
            import soundfile as sf
            wav = torch.rand(T * 512 + 10).numpy().astype("float32")
            (tmp_path / "audio" / spk).mkdir(parents=True, exist_ok=True)
            sf.write(str(tmp_path / "audio" / spk / f"{stem}.wav"), wav, 44100)
    return tmp_path


def _assert_items_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), f"tensor mismatch at {k}"
        else:
            assert a[k] == b[k], f"non-tensor mismatch at {k}"


# ---------------------------------------------------------------------------
# mel quartet: reference replicas of the pre-merge classes
# ---------------------------------------------------------------------------

class _RefMelDataset(BaseSVCDataset):
    """Verbatim pre-merge ddsp6/diffusion/reflow_shallow/rift_dit behavior."""

    def __init__(self, dataset_dir, split, feature_hashes, max_frames, augment,
                 encoder_kind, pitch_aug):
        super().__init__(dataset_dir, split, max_frames, augment)
        self.dataset_dir = Path(dataset_dir)
        self.feature_hashes = feature_hashes
        self.encoder_kind = encoder_kind
        self.meta = load_meta(str(dataset_dir))
        self.spk2idx = {s: i for i, s in enumerate(self.meta.speakers)}
        if self.meta.splits:
            split_files = set(self.meta.splits.get(split, []))
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios
                         if a.file_name in split_files]
        else:
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios]
        self._mel_dir = self.dataset_dir / "mel" / feature_hashes.get("mel", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
        self._units_dir = self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        if pitch_aug:
            self._aug_mel_dir = self.dataset_dir / "aug_mel" / feature_hashes.get("aug_mel", "")
            self._aug_shift_dir = self.dataset_dir / "aug_shift" / feature_hashes.get("aug_shift", "")
        self._pitch_aug = pitch_aug
        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key):
        spk, fn = key
        stem = fn.rsplit(".", 1)[0] + ".pt" if not fn.endswith(".pt") else fn
        mel = self._load_pt(self._mel_dir / spk / stem)
        f0 = self._load_pt(self._f0_dir / spk / stem)
        rms = self._load_pt(self._rms_dir / spk / stem)
        units = self._load_pt(self._units_dir / spk / stem)
        T = mel.shape[0]
        f0 = f0[:T]
        rms = rms[:T]
        units = units[:T]
        return {"mel": mel, "f0": f0, "rms": rms, "units": units,
                "spk_id": self.spk2idx.get(spk, 0)}

    def _augment(self, item):
        return self._pitch_augment(item) if self._pitch_aug else item


MEL_CASES = [
    (DDSPDataset, "contentvec_tta2x", 172, True, "ddsp6"),
    (DiffusionDataset, "contentvec", 172, True, "diffusion"),
    (ReflowShallowDataset, "contentvec", 128, False, "reflow_shallow"),
    (RIFTDataset, "contentvec_tta2x", 256, False, "rift_dit"),
    (ReflowVAEDataset, "contentvec", 172, True, "reflow_vae"),
]


@pytest.mark.parametrize("cls,enc,mf,pitch_aug,label", MEL_CASES,
                         ids=[c[4] for c in MEL_CASES])
def test_mel_quartet_getitem_bit_equal(tmp_path, cls, enc, mf, pitch_aug, label):
    ds_dir = _make_dataset(tmp_path)
    ref = _RefMelDataset(str(ds_dir), "train", HASHS, mf, True, enc, pitch_aug)
    new = cls(str(ds_dir), "train", HASHS, max_frames=mf, augment=True, encoder_kind=enc)
    assert len(ref) == len(new) == 2
    for i in range(len(ref)):
        _assert_items_equal(ref[i], new[i])


# ---------------------------------------------------------------------------
# rvc / sovits: reference replicas of the pre-merge _load_one (incl. rng source)
# ---------------------------------------------------------------------------

class _RefRVCDataset(BaseSVCDataset):
    """Verbatim pre-merge RVCDataset.__init__ + _load_one."""

    def __init__(self, dataset_dir, split, feature_hashes, max_frames=790,
                 augment=True, encoder_kind="contentvec"):
        super().__init__(dataset_dir, split, max_frames, augment)
        self.dataset_dir = Path(dataset_dir)
        self.encoder_kind = encoder_kind
        self.meta = load_meta(str(dataset_dir))
        self.spk2idx = {s: i for i, s in enumerate(self.meta.speakers)}
        if self.meta.splits:
            split_files = set(self.meta.splits.get(split, []))
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios
                         if a.file_name in split_files]
        else:
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios]
        self._units_dir = self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
        self._audio_dir = self.dataset_dir / "audio"
        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key):
        spk, fn = key
        stem = fn.rsplit(".", 1)[0]
        pt_stem = stem + ".pt"
        wav_stem = stem + ".wav"
        units = self._load_pt(self._units_dir / spk / pt_stem)
        f0 = self._load_pt(self._f0_dir / spk / pt_stem)
        spec_path = self._spec_dir / spk / pt_stem
        spec = self._load_pt(spec_path) if spec_path.exists() else None
        rms_path = self._rms_dir / spk / pt_stem
        rms = self._load_pt(rms_path) if rms_path.exists() else None
        wav_path = self._audio_dir / spk / wav_stem
        wav = None
        if wav_path.exists():
            import soundfile as sf
            wav, _ = sf.read(str(wav_path), dtype="float32")
            wav = torch.from_numpy(wav)
        pitchf = f0.clone()
        pitch_token = self._f0_to_coarse(f0)
        T = units.shape[0]
        mf = self.max_frames
        if T > mf:
            start = int(torch.randint(0, T - mf + 1, (1,)).item())
            units = units[start:start + mf]
            pitch_token = pitch_token[start:start + mf]
            pitchf = pitchf[start:start + mf]
            if rms is not None:
                rms = rms[start:start + mf]
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = spec[start:start + mf]
            T = mf
        elif T < mf:
            pad_n = mf - T
            units = torch.nn.functional.pad(units, (0, 0, 0, pad_n))
            pitch_token = torch.nn.functional.pad(pitch_token, (0, pad_n))
            pitchf = torch.nn.functional.pad(pitchf, (0, pad_n))
            if rms is not None:
                rms = torch.nn.functional.pad(rms, (0, pad_n))
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_n))
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
            if spec.ndim == 3:
                spec = spec[0]
            spec_T = spec.shape[0]
            if spec_T > T:
                spec = spec[:T]
            elif spec_T < T:
                spec = torch.nn.functional.pad(spec, (0, 0, 0, T - spec_T))
            spec = spec.transpose(0, 1)
        if wav is not None:
            hop = self.meta.hop
            target_len = T * hop
            if wav.shape[0] > target_len:
                wav = wav[:target_len]
            elif wav.shape[0] < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        return {
            "units": units, "f0": pitchf, "pitch_token": pitch_token,
            "mel": spec, "wav": wav, "spk_id": self.spk2idx.get(spk, 0),
            "rms": rms,
        }

    @staticmethod
    def _f0_to_coarse(f0):
        from usvc.backends.rvc.model import RVCModel
        return RVCModel.f0_to_coarse(f0)


class _RefSoVITSDataset(BaseSVCDataset):
    """Verbatim pre-merge SoVITSDataset.__init__ + _load_one."""

    def __init__(self, dataset_dir, split, feature_hashes, max_frames=790,
                 augment=True, encoder_kind="hubertsoft"):
        super().__init__(dataset_dir, split, max_frames, augment)
        self.dataset_dir = Path(dataset_dir)
        self.encoder_kind = encoder_kind
        self.meta = load_meta(str(dataset_dir))
        self.spk2idx = {s: i for i, s in enumerate(self.meta.speakers)}
        if self.meta.splits:
            split_files = set(self.meta.splits.get(split, []))
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios
                         if a.file_name in split_files]
        else:
            self.keys = [(a.speaker, a.file_name) for a in self.meta.audios]
        self._units_dir = self.dataset_dir / "units" / encoder_kind / feature_hashes.get("units", "")
        self._f0_dir = self.dataset_dir / "f0" / feature_hashes.get("f0", "")
        self._spec_dir = self.dataset_dir / "spec" / feature_hashes.get("spec", "")
        self._rms_dir = self.dataset_dir / "rms" / feature_hashes.get("rms", "")
        self._audio_dir = self.dataset_dir / "audio"
        self.sr = self.meta.sr
        self.hop = self.meta.hop
        self._load_pt = partial(torch.load, weights_only=True, map_location="cpu")

    def _load_one(self, key):
        spk, fn = key
        stem = fn.rsplit(".", 1)[0]
        pt_stem = stem + ".pt"
        wav_stem = stem + ".wav"
        units = self._load_pt(self._units_dir / spk / pt_stem)
        f0 = self._load_pt(self._f0_dir / spk / pt_stem)
        spec_path = self._spec_dir / spk / pt_stem
        spec = self._load_pt(spec_path) if spec_path.exists() else None
        rms_path = self._rms_dir / spk / pt_stem
        rms = self._load_pt(rms_path) if rms_path.exists() else None
        wav_path = self._audio_dir / spk / wav_stem
        wav = None
        if wav_path.exists():
            import soundfile as sf
            wav, _ = sf.read(str(wav_path), dtype="float32")
            wav = torch.from_numpy(wav)
        uv = (f0 > 0).float()
        f0 = interp_unvoiced_f0(f0)
        mf = self.max_frames
        T = units.shape[0]
        if T > mf:
            start = int(torch.randint(0, T - mf + 1, (1,), generator=self._rng).item())
            units = units[start:start + mf]
            f0 = f0[start:start + mf]
            uv = uv[start:start + mf]
            if rms is not None:
                rms = rms[start:start + mf]
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = spec[start:start + mf]
            T = mf
        elif T < mf:
            pad_n = mf - T
            units = torch.nn.functional.pad(units, (0, 0, 0, pad_n))
            f0 = torch.nn.functional.pad(f0, (0, pad_n))
            uv = torch.nn.functional.pad(uv, (0, pad_n))
            if rms is not None:
                rms = torch.nn.functional.pad(rms, (0, pad_n))
            if spec is not None:
                if spec.ndim == 3:
                    spec = spec[0]
                spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_n))
            T = mf
        else:
            if spec is not None and spec.ndim == 3:
                spec = spec[0]
        f0 = f0[:T]
        uv = uv[:T]
        if wav is not None:
            hop = self.meta.hop
            target_len = T * hop
            if wav.shape[0] > target_len:
                wav = wav[:target_len]
            elif wav.shape[0] < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        if spec is not None:
            if spec.ndim == 3:
                spec = spec[0]
            spec_T = spec.shape[0]
            if spec_T > T:
                spec = spec[:T]
            elif spec_T < T:
                spec = torch.nn.functional.pad(spec, (0, 0, 0, T - spec_T))
            spec = spec.transpose(0, 1)
        return {
            "units": units, "f0": f0, "uv": uv, "mel": spec, "wav": wav,
            "rms": rms, "spk_id": self.spk2idx.get(spk, 0),
        }


@pytest.mark.parametrize("cls,ref_cls,enc,label", [
    (RVCDataset, _RefRVCDataset, "contentvec", "rvc"),
    (SoVITSDataset, _RefSoVITSDataset, "hubertsoft", "sovits"),
], ids=["rvc", "sovits"])
def test_gan_datasets_getitem_bit_equal(tmp_path, cls, ref_cls, enc, label):
    ds_dir = _make_dataset(tmp_path)
    ref = ref_cls(str(ds_dir), "train", HASHS, max_frames=790, augment=True, encoder_kind=enc)
    new = cls(str(ds_dir), "train", HASHS, max_frames=790, augment=True, encoder_kind=enc)
    assert len(ref) == len(new) == 2
    for i in range(len(ref)):
        # rvc's crop draws from the GLOBAL default generator: reset it before
        # each side so ref/new see the identical random stream.
        torch.manual_seed(0)
        r = ref[i]
        torch.manual_seed(0)
        x = new[i]
        _assert_items_equal(r, x)


# ---------------------------------------------------------------------------
# attribute surface (F.1) + cli.py mutation flow
# ---------------------------------------------------------------------------

def test_attribute_surface_and_cli_mutation(tmp_path):
    ds_dir = _make_dataset(tmp_path)
    from usvc.backends.rift_dit import RIFTDiTBackend
    from usvc.backends.rvc import RVCBackend

    train_ds = RIFTDiTBackend().build_dataset(str(ds_dir), "train", HASHS)
    for name in ("dataset_dir", "feature_hashes", "encoder_kind", "meta", "spk2idx",
                 "keys", "_load_pt", "_mel_dir", "_f0_dir", "_rms_dir", "_units_dir"):
        assert hasattr(train_ds, name), name

    # cli.py mutates encoder_kind + _units_dir after construction; _load_one
    # must pick up the new path.
    train_ds.encoder_kind = "hubertsoft"
    train_ds._units_dir = train_ds.dataset_dir / "units" / "hubertsoft" / HASHS["units"]
    key0 = train_ds.keys[0]
    item = train_ds._load_one(key0)
    stem = key0[1].rsplit(".", 1)[0] + ".pt"
    hubert_units = torch.load(train_ds._units_dir / key0[0] / stem, weights_only=True)
    assert torch.equal(item["units"], hubert_units)  # loaded from the NEW dir
    assert item["units"].shape[0] == item["mel"].shape[0]

    rvc_ds = RVCBackend().build_dataset(str(ds_dir), "train", HASHS)
    for name in ("encoder_kind", "_units_dir", "_f0_dir", "_spec_dir", "_rms_dir",
                 "_audio_dir", "meta", "spk2idx", "keys", "_load_pt"):
        assert hasattr(rvc_ds, name), name