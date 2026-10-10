import pytest

from usvc.backends.ddsp6.dataset import DDSPDataset
from usvc.backends.sovits.dataset import SoVITSDataset
from usvc.data.meta import AudioEntry, init_dataset_meta
from usvc.speakers import bind_dataset_speakers


def make_meta(path, names=("甲", "乙"), splits=None):
    meta = init_dataset_meta(path, sr=44100, hop=512, f0_min=65, f0_max=800)
    meta.speakers = list(names)
    meta.audios = [AudioEntry(n, "a.wav", 7, .1, 3584) for n in names]
    meta.splits = splits or {}
    meta.save()
    return meta


@pytest.mark.parametrize("cls", [DDSPDataset, SoVITSDataset])
def test_full_identity_and_remapping(tmp_path, cls):
    meta = make_meta(tmp_path, splits={"train": ["甲/a.wav"], "val": ["乙/a.wav"]})
    assert cls(tmp_path, "train", {}).keys == [("甲", "a.wav")]
    assert cls(tmp_path, "val", {}).keys == [("乙", "a.wav")]
    meta.splits = {}; meta.speakers.reverse(); meta.save()
    ds = cls(tmp_path, "train", {})
    bind_dataset_speakers(ds, {"甲": 0, "乙": 1}, training=True)
    assert ds.spk2idx["甲"] == 0
    make_meta(tmp_path / "val", names=("乙",))
    val = cls(tmp_path / "val", "val", {})
    bind_dataset_speakers(val, ds.spk2idx)
    assert val.spk2idx["乙"] == 1


@pytest.mark.parametrize("cls", [DDSPDataset, SoVITSDataset])
def test_invalid_data_and_splits(tmp_path, cls):
    meta = make_meta(tmp_path, splits={"train": ["a.wav"]})
    with pytest.raises(ValueError, match="ambiguous"):
        cls(tmp_path, "train", {})
    meta.splits = {"train": ["甲/a.wav"], "val": ["甲/a.wav"]}; meta.save()
    with pytest.raises(ValueError, match="overlap"):
        cls(tmp_path, "train", {})
    meta.splits = {"train": [], "val": []}; meta.save()
    with pytest.raises(ValueError, match="empty"):
        bind_dataset_speakers(cls(tmp_path, "train", {}), {"甲": 0, "乙": 1}, training=True)
    meta.splits = {"train": ["甲/a.wav"]}; meta.save()
    with pytest.raises(ValueError, match="each speaker"):
        bind_dataset_speakers(cls(tmp_path, "train", {}), {"甲": 0, "乙": 1}, training=True)
    meta.audios[0].speaker = "unknown"; meta.splits = {}; meta.save()
    with pytest.raises(ValueError, match="unknown"):
        cls(tmp_path, "train", {})


def test_unique_legacy_single_speaker(tmp_path):
    make_meta(tmp_path, names=("甲",), splits={"train": ["a.wav"]})
    ds = DDSPDataset(tmp_path, "train", {})
    bind_dataset_speakers(ds, {"甲": 0}, training=True)
    assert ds.keys == [("甲", "a.wav")]


def test_preprocess_preserves_order_and_rejects_renames(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from usvc.cli import preprocess
    meta = make_meta(tmp_path, names=("乙", "甲"))
    monkeypatch.setattr(preprocess, "prepare_audio_dir", lambda *a, **kw: [("甲", "a.wav"), ("乙", "a.wav")])
    monkeypatch.setattr("usvc.io.audio.load_audio", lambda *a, **kw: np.zeros(3584))
    args = SimpleNamespace(dataset_dir=tmp_path, src=tmp_path, sr=44100, hop=512,
                           f0_min=65, f0_max=800, ext=["wav"])
    preprocess._cmd_init(args)
    from usvc.data.meta import load_meta
    assert load_meta(tmp_path).speakers == meta.speakers
    monkeypatch.setattr(preprocess, "prepare_audio_dir", lambda *a, **kw: [("新名字", "a.wav")])
    with pytest.raises(ValueError, match="speaker set changed"):
        preprocess._cmd_init(args)
    assert load_meta(tmp_path).speakers == meta.speakers
