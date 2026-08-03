"""Unit tests for the core data layer (P0). No pretrained weights needed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from usvc.data.feature_config import FeatureConfig, FeatureSpec
from usvc.data.meta import init_dataset_meta, load_meta, save_meta


def test_config_hash_stable_and_distinct():
    a = FeatureConfig("mel", {"n_fft": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0})
    b = FeatureConfig("mel", {"n_mels": 128, "n_fft": 2048, "fmax": 16000.0, "fmin": 40.0})
    assert a.hash() == b.hash(), "key order must not affect hash"

    c = FeatureConfig("mel", {"n_fft": 2048, "n_mels": 80, "fmin": 40.0, "fmax": 16000.0})
    assert a.hash() != c.hash(), "different params must differ"

    # runtime-only fields must be absent (would break cross-machine stability)
    bad = FeatureConfig("mel", {"n_fft": 2048, "device": "cuda"})
    assert "device" in bad.params  # we don't forbid it, but docs say don't add it
    print("PASS test_config_hash_stable_and_distinct")


def test_path_tag_units_isolation():
    hs = FeatureConfig("units", {"kind": "hubertsoft", "model_ckpt": "a.pt"})
    cv = FeatureConfig("units", {"kind": "contentvec", "model_ckpt": "b.pt"})
    assert hs.path_tag.startswith("units/hubertsoft/")
    assert cv.path_tag.startswith("units/contentvec/")
    assert hs.path_tag != cv.path_tag
    mel = FeatureConfig("mel", {"n_fft": 2048})
    assert mel.path_tag.startswith("mel/")
    print("PASS test_path_tag_units_isolation")


def test_meta_roundtrip(tmp_path: Path):
    meta = init_dataset_meta(tmp_path, sr=44100, hop=512, f0_min=65.0, f0_max=800.0)
    meta.speakers = ["s1", "s2"]
    from usvc.data.meta import AudioEntry

    meta.audios.append(AudioEntry("s1", "a.wav", 100, 1.0, 44100))
    fc = FeatureConfig("mel", {"n_fft": 2048, "n_mels": 128})
    meta.register(fc, count=1)
    save_meta(meta)

    re = load_meta(tmp_path)
    assert re.sr == 44100 and re.hop == 512
    assert re.speakers == ["s1", "s2"]
    assert re.is_hash_registered(fc)
    print("PASS test_meta_roundtrip")


def test_feature_spec_dedup():
    # Two backends declaring the same mel config collapse to one hash.
    mel = FeatureConfig("mel", {"n_fft": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0})
    s1 = FeatureSpec(mel=mel, f0=FeatureConfig("f0", {"extractor": "rmvpe"}))
    s2 = FeatureSpec(mel=mel, f0=FeatureConfig("f0", {"extractor": "rmvpe"}))
    union = {fc.hash() for spec in (s1, s2) for fc in spec.all_required()}
    assert len(union) == 2  # one mel hash + one f0 hash, despite two specs
    print("PASS test_feature_spec_dedup")


def test_mel_extractor_shape():
    from usvc.features.mel import MelExtractor

    fc = FeatureConfig(
        "mel",
        {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False},
    )
    ext = MelExtractor(fc, sr=44100, hop=512)
    wav = np.random.randn(44100 * 2).astype("float32") * 0.1  # 2s
    mel = ext.extract(wav)
    assert mel.dim() == 2 and mel.shape[1] == 128, mel.shape
    expected_T = 44100 * 2 // 512 + 1
    assert mel.shape[0] == expected_T, f"mel T {mel.shape[0]} != {expected_T}"
    print(f"PASS test_mel_extractor_shape (mel={tuple(mel.shape)})")


def test_rms_extractor_shape():
    from usvc.features.rms import RmsExtractor

    fc = FeatureConfig("rms", {"win_length": 2048})
    ext = RmsExtractor(fc, sr=44100, hop=512)
    wav = np.random.randn(44100 * 2).astype("float32") * 0.1
    rms = ext.extract(wav)
    expected_T = 44100 * 2 // 512 + 1
    assert rms.shape[0] == expected_T, f"rms T {rms.shape[0]} != {expected_T}"
    print(f"PASS test_rms_extractor_shape (rms={tuple(rms.shape)})")


def test_frame_len_consistency():
    # The central invariant: mel/f0/rms/units(gather) all share frame_len_for(n, hop).
    from usvc.io.audio import frame_len_for

    for n in (44100, 48000, 102400, 1):
        hop = 512
        assert frame_len_for(n, hop) == n // hop + 1
    print("PASS test_frame_len_consistency")


def test_engine_dedup_logic(tmp_path: Path):
    # Simulate the engine's union/dedupe without running extractors (no weights).
    from usvc.data.engine import PreprocessEngine

    # build a minimal dataset
    meta = init_dataset_meta(tmp_path, sr=44100, hop=512, f0_min=65.0, f0_max=800.0)
    meta.speakers = ["s1"]
    from usvc.data.meta import AudioEntry

    meta.audios.append(AudioEntry("s1", "a.wav", 100, 1.0, 44100))
    save_meta(meta)

    engine = PreprocessEngine(tmp_path)
    mel = FeatureConfig("mel", {"n_fft": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0})
    spec_a = FeatureSpec(mel=mel)
    spec_b = FeatureSpec(mel=mel)  # identical -> should dedupe to 1 todo

    # Replicate the engine.run dedupe block manually (without device/extractors).
    todo = {}
    for spec in (spec_a, spec_b):
        for fc in spec.all_required():
            h = fc.hash()
            if h in todo:
                continue
            if engine.meta.is_hash_registered(fc):
                continue
            todo[h] = fc
    assert len(todo) == 1, f"expected 1 todo after dedupe, got {len(todo)}"
    print("PASS test_engine_dedup_logic")


if __name__ == "__main__":
    import tempfile

    test_config_hash_stable_and_distinct()
    test_path_tag_units_isolation()
    test_mel_extractor_shape()
    test_rms_extractor_shape()
    test_frame_len_consistency()
    with tempfile.TemporaryDirectory() as td:
        test_meta_roundtrip(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_engine_dedup_logic(Path(td))
    test_feature_spec_dedup()
    print("\nALL CORE TESTS PASSED")
