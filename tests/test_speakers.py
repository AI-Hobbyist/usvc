import pytest

from usvc.speakers import validate_mapping, resolve_training_speakers, resolve_speaker
from usvc.train.config import TrainConfig
from usvc.data.meta import init_dataset_meta


@pytest.mark.parametrize("mapping", [{}, {"": 0}, {"a": True}, {"a": 0.0}, {"a": -1}, {"a": 1}, {"a": 0, "b": 0}])
def test_invalid_mapping(mapping):
    with pytest.raises(ValueError):
        validate_mapping(mapping)


def test_names_and_id_zero():
    mapping = {"歌手 甲": 0, "歌手乙": 1}
    assert validate_mapping(mapping) == mapping
    assert resolve_speaker("歌手 甲", mapping, 2) == 0
    assert resolve_speaker(0, mapping, 2) == 0
    for selection in (None, True, -1, 2, 0.0, "unknown"):
        with pytest.raises(ValueError):
            resolve_speaker(selection, mapping, 2)


@pytest.mark.parametrize("explicit", [False, True])
def test_capacity_and_yaml(tmp_path, explicit):
    path = tmp_path / "train.yaml"
    path.write_text("backend: ddsp6\nspk:\n  歌手甲: 0\n  歌手乙: 1\n" +
                    ("backend_opts:\n  num_speaker: 1\n" if explicit else ""), encoding="utf-8")
    cfg = TrainConfig.from_yaml(path)
    meta = init_dataset_meta(tmp_path, sr=44100, hop=512, f0_min=65, f0_max=800)
    meta.speakers = ["歌手乙", "歌手甲"]
    if explicit:
        with pytest.raises(ValueError, match="num_speaker"):
            resolve_training_speakers(cfg, meta)
    else:
        resolve_training_speakers(cfg, meta)
        assert cfg.backend_opts["num_speaker"] == 2
        assert cfg.spk["歌手甲"] == 0


def test_programmatic_defaults_and_invalid_meta(tmp_path):
    meta = init_dataset_meta(tmp_path, sr=44100, hop=512, f0_min=65, f0_max=800)
    meta.speakers = ["乙", "甲"]
    cfg = TrainConfig(backend="sovits")
    assert resolve_training_speakers(cfg, meta) == {"乙": 0, "甲": 1}
    for names in ([], ["a", "a"], ["unknown"]):
        meta.speakers = names
        with pytest.raises(ValueError):
            resolve_training_speakers(cfg, meta)


@pytest.mark.parametrize("filename", ["ddsp6.yaml", "ddsp65.yaml", "sovits.yaml"])
def test_existing_configs_are_compatible(tmp_path, filename):
    from pathlib import Path
    cfg = TrainConfig.from_yaml(Path("configs") / filename)
    meta = init_dataset_meta(tmp_path, sr=44100, hop=512, f0_min=65, f0_max=800)
    meta.speakers = ["原有角色"]
    assert resolve_training_speakers(cfg, meta) == {"原有角色": 0}
    assert cfg.backend_opts["num_speaker"] == 1


def test_duplicate_yaml_names_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("backend: ddsp6\nspk:\n  甲: 0\n  甲: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate speaker"):
        TrainConfig.from_yaml(path)
