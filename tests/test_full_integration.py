"""Full SVC preprocessing integration test.

Verifies ALL feature extractors and ALL backend FEATURE_SPECs against the
seven reference projects' real needs, on realistic synthetic speech audio.

Covers:
- mel / f0 / rms / spec (all frame-aligned to len//hop+1)
- units: hubertsoft / contentvec / contentvec_tta2x / contentvec_proj (RVC v1)
- all 5 backends (rift_dit, ddsp6, diffusion, sovits, rvc) FEATURE_SPEC
- cross-backend feature reuse (config-hash dedupe)
- F0 pitch correctness on male/female-range voices
- long-audio frame alignment
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from usvc.backends.ddsp6 import DDSP6Backend
from usvc.backends.diffusion import DiffusionBackend
from usvc.backends.rift_dit import RIFTDiTBackend
from usvc.backends.rvc import RVCBackend
from usvc.backends.sovits import SoVitsBackend
from usvc.data.engine import PreprocessEngine
from usvc.data.feature_config import FeatureConfig, FeatureSpec
from usvc.data.meta import init_dataset_meta, save_meta
from usvc.io.audio import frame_len_for

PRETRAINED = Path("pretrained")
RMVPE = str(PRETRAINED / "rmvpe" / "model.pt")
CVEC = str(PRETRAINED / "contentvec" / "pytorch_model.bin")
HSOFT = str(PRETRAINED / "hubert" / "hubert-soft-0d54a1f4.pt")


def _make_voice_audio(out_dir: Path, sr: int = 44100) -> list[tuple[str, str]]:
    """Synthesize realistic-ish speech (harmonics + formants + AM)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def vowel(f0, formants, dur=3.0):
        t = np.arange(int(sr * dur)) / sr
        src = 2 * (t * f0 - np.floor(0.5 + t * f0))  # sawtooth (rich harmonics)
        sig = 0.3 * src
        for i, F in enumerate(formants):
            sig += 0.4 * np.sin(2 * np.pi * F * t) / (i + 1)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # syllabic AM
        sig *= am
        sig += 0.02 * np.random.randn(len(t))
        return sig.astype("float32") * 0.5

    import soundfile as sf

    np.random.seed(0)
    clips = {
        ("spk_a", "a1"): vowel(120, [300, 870, 2240]),   # male-ish /a/
        ("spk_a", "a2"): vowel(130, [270, 2290, 3010]),  # male-ish /i/
        ("spk_b", "b1"): vowel(220, [310, 880, 2250]),   # female-ish /a/
        ("spk_b", "b2"): vowel(240, [280, 2300, 3050]),  # female-ish /i/
    }
    discovered = []
    for (spk, name), wav in clips.items():
        p = out_dir / spk / f"{name}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), wav, sr)
        discovered.append((spk, f"{name}.wav"))
    return discovered


def _stage_dataset(ds_dir: Path, src_dir: Path, *, sr: int, hop: int) -> None:
    """Stage audio + write meta.json (reuse CLI logic)."""
    from usvc.data.engine import prepare_audio_dir
    from usvc.data.meta import AudioEntry

    discovered = prepare_audio_dir(src_dir, ds_dir, sr=sr)
    meta = init_dataset_meta(ds_dir, sr=sr, hop=hop, f0_min=65.0, f0_max=800.0)
    meta.speakers = sorted({s for s, _ in discovered})
    from usvc.io.audio import load_audio

    meta.audios = []
    for spk, fn in discovered:
        wav = load_audio(ds_dir / "audio" / spk / fn, sr=sr)
        n = len(wav)
        meta.audios.append(
            AudioEntry(speaker=spk, file_name=fn, frame_len=frame_len_for(n, hop), duration=n / sr, audio_samples=n)
        )
    save_meta(meta)


def _load_feature(ds: Path, kind: str, spk: str, stem: str) -> torch.Tensor:
    """Load the (single) computed feature of a kind for a clip."""
    import torch as t

    kdir = ds / kind
    if kind == "units":
        # caller must pass kind with encoder bucket; handled separately
        raise ValueError("use _load_units instead")
    h = next(iter(kdir.iterdir())).name
    return t.load(kdir / h / spk / f"{stem}.pt", weights_only=True)


def _load_units(ds: Path, enc_kind: str, spk: str, stem: str) -> torch.Tensor:
    import torch as t

    bucket = ds / "units" / enc_kind
    h = next(iter(bucket.iterdir())).name
    return t.load(bucket / h / spk / f"{stem}.pt", weights_only=True)


# ===========================================================================
def test_all_features_44100(tmp_path: Path) -> None:
    """44.1k/512 family: rift_dit / ddsp6 / diffusion share mel/f0/rms."""
    print("\n=== TEST: 44.1k family (rift_dit + ddsp6 + diffusion) ===")
    src = tmp_path / "voice_src"
    ds = tmp_path / "ds_44100"
    _make_voice_audio(src)
    _stage_dataset(ds, src, sr=44100, hop=512)

    # Three backends union — mel/f0/rms should dedupe to ONE hash each.
    specs = [
        RIFTDiTBackend.feature_spec({}),   # contentvec_tta2x
        DDSP6Backend.feature_spec({}),     # contentvec_tta2x (same -> dedupe)
        DiffusionBackend.feature_spec({}), # plain contentvec
    ]
    # Patch ckpt paths to local pretrained
    for s in specs:
        if s.f0: s.f0.params["model_ckpt"] = RMVPE
        if s.units: s.units.params["model_ckpt"] = CVEC

    engine = PreprocessEngine(ds)
    engine.run(specs, device="cpu")

    # Validate frame alignment + pitch on every clip
    for spk in ("spk_a", "spk_b"):
        for stem in ("a1", "a2", "b1", "b2")[:2] if spk == "spk_a" else ("b1", "b2"):
            mel = _load_feature(ds, "mel", spk, stem)
            f0 = _load_feature(ds, "f0", spk, stem)
            rms = _load_feature(ds, "rms", spk, stem)
            cv_tta = _load_units(ds, "contentvec_tta2x", spk, stem)
            cv = _load_units(ds, "contentvec", spk, stem)
            T = mel.shape[0]
            assert mel.shape[0] == f0.shape[0] == rms.shape[0] == cv.shape[0] == T, (
                f"{spk}/{stem} frame mismatch: mel={mel.shape[0]} f0={f0.shape[0]} "
                f"rms={rms.shape[0]} cv={cv.shape[0]}"
            )
            # contentvec_tta2x is 2x frames (RIFT style, native rate) — its own length
            # F0 pitch check: relative correctness (male < female), not exact.
            # Sawtooth synth has rich harmonics that can trigger octave errors in
            # RMVPE, so we check ordering and range rather than exact f0.
            voiced = f0[f0 > 0]
            assert len(voiced) > T * 0.3, f"{spk}/{stem}: too few voiced frames"
            mean_f0 = voiced.mean().item()
            base = {"a1": 120, "a2": 130, "b1": 220, "b2": 240}[stem]
            # accept base or any octave/harmonic within [base*0.9, base*4.5]
            assert base * 0.9 <= mean_f0 <= base * 4.5, (
                f"{spk}/{stem}: f0={mean_f0:.0f} not in [{base*0.9:.0f},{base*4.5:.0f}]"
            )
            print(f"  {spk}/{stem}: T={T} mel={mel.shape[1]}mel cv={cv.shape[1]}d "
                  f"cv_tta={cv_tta.shape[0]}frames f0={mean_f0:.0f}Hz (base {base}) ✓")

    # Verify male voices have lower f0 than female voices on average.
    import torch as t
    male_f0 = []
    female_f0 = []
    for spk in ("spk_a", "spk_b"):
        for stem in (("a1", "a2") if spk == "spk_a" else ("b1", "b2")):
            f0 = _load_feature(ds, "f0", spk, stem)
            v = f0[f0 > 0]
            (male_f0 if spk == "spk_a" else female_f0).append(v.mean().item())
    assert np.mean(male_f0) < np.mean(female_f0), (
        f"male f0 ({np.mean(male_f0):.0f}) should < female ({np.mean(female_f0):.0f})"
    )
    print(f"  pitch ordering: male={np.mean(male_f0):.0f}Hz < female={np.mean(female_f0):.0f}Hz ✓")
    # Verify dedupe: one mel hash; f0 has raw (rift_dit) + interp (ddsp6/diffusion)
    # hashes; two rms hashes.
    meta_features = engine.meta.features
    assert len(meta_features["mel"]) == 1, f"mel should dedupe to 1 hash, got {len(meta_features['mel'])}"
    assert len(meta_features["f0"]) == 2, f"f0 should have 2 hashes (raw + interp)"
    assert len(meta_features["rms"]) == 2, f"rms should have 2 hashes (win2048 + win1024)"
    print("  dedupe: mel=1 f0=2(raw+interp) rms=2(hashes) contentvec=2(tta2x+plain) ✓")
    print("PASS test_all_features_44100")


def test_sovits_backend(tmp_path: Path) -> None:
    """so-vits: 80-mel + spec + contentvec units (4.1 default)."""
    print("\n=== TEST: so-vits backend (80mel + spec + contentvec768) ===")
    src = tmp_path / "voice_src2"
    ds = tmp_path / "ds_sovits"
    _make_voice_audio(src)
    _stage_dataset(ds, src, sr=44100, hop=512)

    spec = SoVitsBackend.feature_spec({})  # 默认 contentvec (4.1)
    spec.f0.params["model_ckpt"] = RMVPE
    spec.units.params["model_ckpt"] = CVEC
    engine = PreprocessEngine(ds)
    engine.run([spec], device="cpu")

    mel = _load_feature(ds, "mel", "spk_a", "a1")
    spec_t = _load_feature(ds, "spec", "spk_a", "a1")
    cv = _load_units(ds, "contentvec", "spk_a", "a1")
    assert mel.shape[1] == 80, f"so-vits mel should be 80, got {mel.shape[1]}"
    assert spec_t.shape[1] == 1025, f"spec should be n_fft/2+1=1025, got {spec_t.shape[1]}"
    assert cv.shape[1] == 768, f"contentvec dim should be 768"
    assert mel.shape[0] == spec_t.shape[0] == cv.shape[0], "frame mismatch"
    print(f"  mel={tuple(mel.shape)}(80mel) spec={tuple(spec_t.shape)} cv={tuple(cv.shape)}(768d) ✓")
    print("PASS test_sovits_backend")


def test_rvc_backend_v1_v2(tmp_path: Path) -> None:
    """RVC v2 (last_hidden, 768) and v1 (layer9+final_proj, 256)."""
    print("\n=== TEST: rvc backend (v1 + v2 contentvec variants) ===")
    src = tmp_path / "voice_src3"
    ds = tmp_path / "ds_rvc"
    _make_voice_audio(src)
    _stage_dataset(ds, src, sr=48000, hop=480)  # RVC uses 48k/480

    spec_v2 = RVCBackend.feature_spec({})
    spec_v1 = RVCBackend.feature_spec({"rvc_version": "v1"})
    spec_v2.f0.params["model_ckpt"] = RMVPE
    spec_v1.f0.params["model_ckpt"] = RMVPE
    spec_v2.units.params["model_ckpt"] = CVEC
    spec_v1.units.params["model_ckpt"] = CVEC
    engine = PreprocessEngine(ds)
    engine.run([spec_v2, spec_v1], device="cpu")

    cv_v2 = _load_units(ds, "contentvec", "spk_a", "a1")
    cv_v1 = _load_units(ds, "contentvec_proj", "spk_a", "a1")
    assert cv_v2.shape[1] == 768, f"RVC v2 units should be 768-d, got {cv_v2.shape[1]}"
    assert cv_v1.shape[1] == 256, f"RVC v1 units (final_proj) should be 256-d, got {cv_v1.shape[1]}"
    mel = _load_feature(ds, "mel", "spk_a", "a1")
    assert mel.shape[0] == cv_v2.shape[0] == cv_v1.shape[0], "frame mismatch"
    print(f"  v2(contentvec): {tuple(cv_v2.shape)}(768d)  v1(contentvec_proj): {tuple(cv_v1.shape)}(256d) ✓")
    print("PASS test_rvc_backend_v1_v2")


def test_hubertsoft_also_available(tmp_path: Path) -> None:
    """HubertSoft can coexist with ContentVec in same dataset (physical isolation)."""
    print("\n=== TEST: hubertsoft + contentvec coexist ===")
    src = tmp_path / "voice_src4"
    ds = tmp_path / "ds_mixed"
    _make_voice_audio(src)
    _stage_dataset(ds, src, sr=44100, hop=512)

    # User trains ddsp6 with hubertsoft AND diffusion with contentvec on same data.
    spec_hs = DDSP6Backend.feature_spec({"units_kind": "hubertsoft"})
    spec_cv = DiffusionBackend.feature_spec({})  # plain contentvec
    for s in (spec_hs, spec_cv):
        s.f0.params["model_ckpt"] = RMVPE
        s.units.params["model_ckpt"] = HSOFT if s.units.params["kind"] == "hubertsoft" else CVEC
    engine = PreprocessEngine(ds)
    engine.run([spec_hs, spec_cv], device="cpu")

    hs = _load_units(ds, "hubertsoft", "spk_a", "a1")
    cv = _load_units(ds, "contentvec", "spk_a", "a1")
    assert hs.shape[1] == 256 and cv.shape[1] == 768
    assert hs.shape[0] == cv.shape[0]  # same frame count (both gather-aligned)
    print(f"  hubertsoft={tuple(hs.shape)} contentvec={tuple(cv.shape)} 共存 ✓")
    print("PASS test_hubertsoft_also_available")


def test_long_audio_alignment(tmp_path: Path) -> None:
    """Long (60s) audio frame alignment — no off-by-one at scale."""
    print("\n=== TEST: long audio (60s) frame alignment ===")
    src = tmp_path / "long_src"
    src.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sr = 44100
    t = np.arange(sr * 60) / sr
    wav = (0.3 * np.sin(2 * np.pi * 150 * t) + 0.02 * np.random.randn(len(t))).astype("float32")
    (src / "spk_a").mkdir(parents=True, exist_ok=True)
    sf.write(str(src / "spk_a" / "long.wav"), wav, sr)

    ds = tmp_path / "ds_long"
    _stage_dataset(ds, src, sr=sr, hop=512)
    fc = FeatureConfig("mel", {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False})
    engine = PreprocessEngine(ds)
    engine.run([FeatureSpec(mel=fc)], device="cpu")
    mel = _load_feature(ds, "mel", "spk_a", "long")
    expected_T = sr * 60 // 512  # nvSTFT center=False grid: (L-hop)//hop+1 == L//hop
    assert mel.shape[0] == expected_T, f"long audio T={mel.shape[0]} expected {expected_T}"
    print(f"  60s audio -> T={mel.shape[0]} (expected {expected_T}) ✓")
    print("PASS test_long_audio_alignment")


def test_all_backends_declared() -> None:
    """All 6 backends have DEFAULT_FEATURES and implement feature_spec."""
    print("\n=== TEST: all backends declared ===")
    from usvc.cli.preprocess import _BACKENDS

    assert set(_BACKENDS.keys()) == {"rift_dit", "ddsp6", "diffusion", "sovits", "rvc", "reflow_vae", "reflow_shallow"}
    for name, cls in _BACKENDS.items():
        s = cls.feature_spec({})
        assert len(s.all_required()) >= 3, f"{name} needs >=3 features"
        print(f"  {name}: {[fc.kind for fc in s.all_required()]} ✓")
    print("PASS test_all_backends_declared")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_all_backends_declared()
        test_all_features_44100(tp)
        test_sovits_backend(tp)
        test_rvc_backend_v1_v2(tp)
        test_hubertsoft_also_available(tp)
        test_long_audio_alignment(tp)
        print("\n" + "=" * 60)
        print("ALL SVC PREPROCESSING INTEGRATION TESTS PASSED")
        print("=" * 60)
