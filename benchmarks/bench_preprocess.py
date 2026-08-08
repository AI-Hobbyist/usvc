"""Preprocessing speed benchmark on a wavtest corpus (real game voice corpus).

The corpus dir defaults to <repo>/data/wavtest (set USVC_WAVTEST_DIR to
override; a typical corpus is 1540 files, 48kHz, ~2.6h total). Measures:
- per-feature latency (CPU vs GPU) on a representative clip
- full engine throughput (files/s, realtime factor) on a 100-file subset
- breakdown by feature kind
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from usvc.data.engine import PreprocessEngine, prepare_audio_dir
from usvc.data.feature_config import FeatureConfig, FeatureSpec
from usvc.data.meta import AudioEntry, init_dataset_meta, save_meta
from usvc.features.f0 import F0Extractor
from usvc.features.mel import MelExtractor
from usvc.features.rms import RmsExtractor
from usvc.features.spec import SpecExtractor
from usvc.features.units import UnitsExtractor
from usvc.io.audio import frame_len_for, load_audio

REPO = Path(__file__).resolve().parents[1]
WAVTEST = Path(os.environ.get("USVC_WAVTEST_DIR") or (REPO / "data" / "wavtest"))
BENCH_DIR = Path(os.environ.get("USVC_BENCH_DIR") or (Path(tempfile.gettempdir()) / "usvc_bench"))
N_SUBSET = 100  # subset size for throughput test
WARMUP = 3

PRETRAINED = {
    "rmvpe": str(REPO / "pretrained/rmvpe/model.pt"),
    "cvec": str(REPO / "pretrained/contentvec/pytorch_model.bin"),
    "hsoft": str(REPO / "pretrained/hubert/hubert-soft-0d54a1f4.pt"),
}


def _stage_subset(dst: Path, n: int) -> tuple[list[tuple[str, str]], float]:
    """Stage n files (single speaker) + return (clips, total_duration_s)."""
    if dst.exists():
        shutil.rmtree(dst)
    files = sorted(WAVTEST.glob("*.wav"))[:n]
    total_dur = 0.0
    discovered = []
    (dst / "audio" / "spk0").mkdir(parents=True)
    for f in files:
        wav = load_audio(f, sr=44100)
        from usvc.io.audio import save_audio

        save_audio(dst / "audio" / "spk0" / f.name, wav, sr=44100)
        discovered.append(("spk0", f.name))
        total_dur += len(wav) / 44100
    meta = init_dataset_meta(dst, sr=44100, hop=512, f0_min=65.0, f0_max=800.0)
    meta.speakers = ["spk0"]
    meta.audios = [
        AudioEntry(
            speaker="spk0",
            file_name=fn,
            frame_len=frame_len_for(int(dur := load_audio(dst / "audio" / spk / fn, sr=44100).shape[0]), 512),
            duration=(d := dur) / 44100,
            audio_samples=d,
        )
        for spk, fn in discovered
    ]
    save_meta(meta)
    return discovered, total_dur


def bench_per_feature(clip_path: Path) -> dict:
    """Per-feature latency on one clip, CPU vs GPU."""
    wav44 = load_audio(clip_path, sr=44100)
    wav16 = torch.from_numpy(load_audio(clip_path, sr=16000)).float()
    dur = len(wav44) / 44100
    print(f"\n[per-feature] clip = {clip_path.name} ({dur:.2f}s @ 44.1k)")

    configs = {
        "mel": FeatureConfig("mel", {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False}),
        "rms": FeatureConfig("rms", {"win_length": 2048}),
        "spec": FeatureConfig("spec", {"n_fft": 2048, "win_length": 2048, "center": False}),
        "f0(rmvpe)": FeatureConfig("f0", {"extractor": "rmvpe", "uv_policy": "raw", "model_ckpt": PRETRAINED["rmvpe"]}),
        "f0(fcpe)": FeatureConfig("f0", {"extractor": "fcpe", "uv_policy": "raw"}),
        "units(hsoft)": FeatureConfig("units", {"kind": "hubertsoft", "sr_in": 16000, "hop_native": 320, "align_mode": "gather", "model_ckpt": PRETRAINED["hsoft"]}),
        "units(cvec)": FeatureConfig("units", {"kind": "contentvec", "layer": 12, "sr_in": 16000, "hop_native": 320, "align_mode": "gather", "model_ckpt": PRETRAINED["cvec"]}),
        "units(cvec_tta2x)": FeatureConfig("units", {"kind": "contentvec_tta2x", "layer": 12, "tta_2x": True, "sr_in": 16000, "hop_native": 160, "align_mode": "gather", "model_ckpt": PRETRAINED["cvec"]}),
    }

    results = {}
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        print(f"\n  --- device: {device} ---")
        for name, fc in configs.items():
            # skip GPU for mel/rms/spec (CPU-bound, GPU slower due to transfer)
            if device == "cuda" and name in ("rms",):
                continue
            try:
                ext = _build(fc, device)
                # warmup
                for _ in range(WARMUP):
                    _run_one(ext, fc, wav44, wav16)
                if device == "cuda":
                    torch.cuda.synchronize()
                # measure
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    _run_one(ext, fc, wav44, wav16)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                med = statistics.median(times)
                rtf = dur / med  # realtime factor (1.0 = realtime)
                print(f"    {name:18s} {med*1000:7.1f}ms  (RTF {rtf:6.1f}x)")
                results[f"{name}@{device}"] = {"ms": round(med * 1000, 1), "rtf": round(rtf, 1)}
            except Exception as e:
                print(f"    {name:18s} ERROR: {e}")
                results[f"{name}@{device}"] = {"error": str(e)[:60]}
    return {"clip": {"dur_s": round(dur, 2), "samples": len(wav44)}, "timings_ms": results}


def _build(fc: FeatureConfig, device: str):
    if fc.kind == "mel":
        return MelExtractor(fc, 44100, 512).to(device)
    if fc.kind == "rms":
        return RmsExtractor(fc, 44100, 512)
    if fc.kind == "spec":
        return SpecExtractor(fc, 44100, 512).to(device)
    if fc.kind == "f0":
        return F0Extractor(fc, 44100, 512, device=device)
    if fc.kind == "units":
        return UnitsExtractor(fc, 44100, 512, device=device)
    raise ValueError(fc.kind)


def _run_one(ext, fc: FeatureConfig, wav44: np.ndarray, wav16: torch.Tensor) -> None:
    if fc.kind == "units":
        ext.extract(wav16)
    else:
        ext.extract(wav44)


def bench_engine_throughput(device: str) -> dict:
    """Full engine.run on N_SUBSET files, single process sequential."""
    ds = BENCH_DIR / f"ds_{device}"
    clips, total_dur = _stage_subset(ds, N_SUBSET)
    print(f"\n[engine throughput / {device}] {N_SUBSET} files, {total_dur:.1f}s audio ({total_dur/60:.1f}min)")

    # rift_dit-style spec: mel+f0+rms+contentvec_tta2x (the heaviest common case)
    spec = FeatureSpec(
        mel=FeatureConfig("mel", {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False}),
        f0=FeatureConfig("f0", {"extractor": "rmvpe", "uv_policy": "raw", "model_ckpt": PRETRAINED["rmvpe"]}),
        rms=FeatureConfig("rms", {"win_length": 2048}),
        units=FeatureConfig("units", {"kind": "contentvec_tta2x", "layer": 12, "tta_2x": True, "sr_in": 16000, "hop_native": 160, "align_mode": "gather", "model_ckpt": PRETRAINED["cvec"]}),
    )
    engine = PreprocessEngine(ds)
    t0 = time.perf_counter()
    engine.run([spec], device=device)
    elapsed = time.perf_counter() - t0

    rtf = total_dur / elapsed
    files_per_s = N_SUBSET / elapsed
    print(f"  elapsed: {elapsed:.1f}s | {files_per_s:.2f} files/s | RTF {rtf:.1f}x (1min audio -> {60/rtf:.1f}s preprocess)")
    print(f"  extrapolated full corpus (155.6min): {155.6*60/rtf/60:.1f}min")
    return {
        "device": device,
        "n_files": N_SUBSET,
        "audio_s": round(total_dur, 1),
        "elapsed_s": round(elapsed, 1),
        "files_per_s": round(files_per_s, 3),
        "rtf": round(rtf, 1),
        "full_corpus_min": round(155.6 * 60 / rtf / 60, 1),
    }


def bench_per_feature_breakdown(device: str) -> dict:
    """Time each feature kind separately over the subset (shows bottleneck)."""
    ds = BENCH_DIR / f"ds_breakdown_{device}"
    clips, total_dur = _stage_subset(ds, N_SUBSET)
    print(f"\n[per-feature breakdown / {device}] cumulative time over {N_SUBSET} files")
    kinds = [
        ("mel", FeatureConfig("mel", {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False})),
        ("rms", FeatureConfig("rms", {"win_length": 2048})),
        ("f0", FeatureConfig("f0", {"extractor": "rmvpe", "uv_policy": "raw", "model_ckpt": PRETRAINED["rmvpe"]})),
        ("units(cvec_tta2x)", FeatureConfig("units", {"kind": "contentvec_tta2x", "layer": 12, "tta_2x": True, "sr_in": 16000, "hop_native": 160, "align_mode": "gather", "model_ckpt": PRETRAINED["cvec"]})),
    ]
    out = {}
    for name, fc in kinds:
        engine = PreprocessEngine(ds)
        # force recompute by mutating hash dir
        fc2 = FeatureConfig(fc.kind, {**fc.params, "_bench_run": name + str(time.time())})
        t0 = time.perf_counter()
        engine.run([FeatureSpec(**{fc2.kind if fc2.kind != "units" else "units": fc2})], device=device)
        el = time.perf_counter() - t0
        print(f"  {name:20s} {el:6.1f}s  ({el/total_dur:.2f}s per 1s audio)")
        out[name] = {"elapsed_s": round(el, 1), "per_s_audio": round(el / total_dur, 3)}
    return out


def main() -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. per-feature latency on a representative ~6s clip
    clips = sorted(WAVTEST.glob("*.wav"))
    mid_clip = clips[len(clips) // 2]  # ~median duration
    report["per_feature"] = bench_per_feature(mid_clip)

    # 2. engine throughput (sequential, single process)
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        report[f"throughput_{dev}"] = bench_engine_throughput(dev)

    # 3. breakdown by feature kind on GPU (the realistic path)
    if torch.cuda.is_available():
        report["breakdown_cuda"] = bench_per_feature_breakdown("cuda")

    out = REPO / "bench_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n=== report saved to {out} ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
