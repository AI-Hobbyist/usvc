"""CPU/GPU utilization profiler for preprocessing configs.

For each (feature, device) config, runs preprocessing on a subset of wavtest
in a subprocess while sampling CPU% (per-core + total) and GPU% via
nvidia-smi. Reports average utilization + idle ratio (reveals whether the
config is compute-bound or IO/bottleneck-bound).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[1]
WAVTEST = Path(os.environ.get("USVC_WAVTEST_DIR") or (REPO / "data" / "wavtest"))
BENCH = Path(os.environ.get("USVC_BENCH_DIR") or (Path(tempfile.gettempdir()) / "usvc_util"))
N = 60  # files per config (enough to sample for ~10-30s)
SAMPLE_HZ = 20  # sampling rate (every 50ms)

sys.path.insert(0, str(REPO))
from usvc.hub import pretrained_dir  # noqa: E402

PRETRAINED = {
    "rmvpe": str(pretrained_dir() / "rmvpe" / "model.pt"),
    "hsoft": str(pretrained_dir() / "hubert" / "hubert-soft-0d54a1f4.pt"),
    "cvec": str(pretrained_dir() / "contentvec" / "pytorch_model.bin"),
}


def _gpu_util() -> tuple[float, float]:
    """Returns (gpu_pct, mem_used_mb) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip().split(", ")
        return float(out[0]), float(out[1])
    except Exception:
        return -1.0, -1.0


def _stage() -> tuple[Path, float]:
    BENCH.mkdir(parents=True, exist_ok=True)
    ds = BENCH / "ds"
    (ds / "audio" / "spk0").mkdir(parents=True, exist_ok=True)
    import shutil

    if (ds / "meta.json").exists():
        return ds, sum(a.duration for a in __import__("usvc.data.meta", fromlist=["load_meta"]).load_meta(ds).audios)
    files = sorted(WAVTEST.glob("*.wav"))[:N]
    total = 0.0
    sys.path.insert(0, str(REPO))
    from usvc.data.engine import prepare_audio_dir
    from usvc.data.meta import AudioEntry, init_dataset_meta, save_meta
    from usvc.io.audio import frame_len_for, load_audio

    # stage via prepare_audio_dir for correctness
    src = BENCH / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "spk0").mkdir(exist_ok=True)
    import soundfile as sf

    for f in files:
        wav = load_audio(f, sr=44100)
        sf.write(str(src / "spk0" / f.name), wav, 44100)
        total += len(wav) / 44100
    prepare_audio_dir(src, ds, sr=44100)
    meta = init_dataset_meta(ds, sr=44100, hop=512, f0_min=65.0, f0_max=800.0)
    meta.speakers = ["spk0"]
    meta.audios = []
    for spk, fn in [(sp, fn) for sp in ["spk0"] for fn in sorted((ds / "audio" / sp).glob("*.wav"))]:
        wav = load_audio(ds / "audio" / spk / fn.name, sr=44100)
        n = len(wav)
        meta.audios.append(
            AudioEntry(
                speaker=spk,
                file_name=fn.name,
                frame_len=frame_len_for(n, 512),
                duration=n / 44100,
                audio_samples=n,
            )
        )
    save_meta(meta)
    return ds, total


def run_feature(fc_kind: str, fc_params: dict, device: str, ds: Path) -> dict:
    """Run one feature over the dataset in THIS process (so sampling works on the
    same process tree). We sample from a background thread."""
    sys.path.insert(0, str(REPO))
    # Force fresh computation: mutate params to get a unique hash per run.
    fc_params = {**fc_params, "_run": str(time.time())}
    from usvc.data.engine import PreprocessEngine
    from usvc.data.feature_config import FeatureConfig, FeatureSpec

    fc = FeatureConfig(fc_kind, fc_params)
    spec = FeatureSpec(**{fc_kind if fc_kind != "units" else "units": fc})

    samples = {"cpu_total": [], "cpu_per_core": [], "gpu": [], "gpu_mem": []}
    done = {"v": False}

    def sample():
        while not done["v"]:
            samples["cpu_total"].append(psutil.cpu_percent(interval=None))
            samples["cpu_per_core"].append(psutil.cpu_percent(interval=None, percpu=True))
            g, m = _gpu_util()
            samples["gpu"].append(g)
            samples["gpu_mem"].append(m)
            time.sleep(1.0 / SAMPLE_HZ)

    # prime cpu_percent (first call returns 0)
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)
    t = ThreadPoolExecutor(max_workers=1)
    sampler = t.submit(sample)

    engine = PreprocessEngine(ds)
    t0 = time.perf_counter()
    try:
        engine.run([spec], device=device)
    except Exception as e:
        done["v"] = True
        return {"error": str(e)[:80]}
    elapsed = time.perf_counter() - t0
    done["v"] = True
    sampler.result(timeout=5)

    def avg(x):
        return round(statistics.mean(x), 1) if x else 0.0

    def p(x, q):
        return round(statistics.quantiles(x, n=100)[q - 1], 1) if len(x) >= 100 else 0.0

    cpu_total = samples["cpu_total"]
    gpu = samples["gpu"]
    # per-core: average each core across samples, then report max/avg core
    per_core = list(zip(*samples["cpu_per_core"])) if samples["cpu_per_core"] else []
    core_avg = [avg(c) for c in per_core]

    return {
        "elapsed_s": round(elapsed, 2),
        "cpu_total_avg_pct": avg(cpu_total),
        "cpu_total_p50": p(cpu_total, 50) if cpu_total else 0,
        "cpu_cores_used_avg": round(sum(1 for c in core_avg if c > 30), 1),  # cores >30% busy
        "cpu_core_max_pct": max(core_avg) if core_avg else 0,
        "gpu_avg_pct": avg([g for g in gpu if g >= 0]) if gpu else 0,
        "gpu_p50_pct": p([g for g in gpu if g >= 0], 50) if gpu else 0,
        "gpu_mem_mb": avg([m for m in samples["gpu_mem"] if m >= 0]) if samples["gpu_mem"] else 0,
        "n_samples": len(cpu_total),
    }


CONFIGS = {
    "mel": ("mel", {"n_fft": 2048, "win_length": 2048, "n_mels": 128, "fmin": 40.0, "fmax": 16000.0, "center": False}),
    "rms": ("rms", {"win_length": 2048}),
    "spec": ("spec", {"n_fft": 2048, "win_length": 2048, "center": False}),
    "f0_rmvpe": ("f0", {"extractor": "rmvpe", "uv_policy": "raw", "model_ckpt": PRETRAINED["rmvpe"]}),
    "f0_fcpe": ("f0", {"extractor": "fcpe", "uv_policy": "raw"}),
    "units_hsoft": ("units", {"kind": "hubertsoft", "sr_in": 16000, "hop_native": 320, "align_mode": "gather", "model_ckpt": PRETRAINED["hsoft"]}),
    "units_cvec": ("units", {"kind": "contentvec", "layer": 12, "sr_in": 16000, "hop_native": 320, "align_mode": "gather", "model_ckpt": PRETRAINED["cvec"]}),
    "units_cvec_tta2x": ("units", {"kind": "contentvec_tta2x", "layer": 12, "tta_2x": True, "sr_in": 16000, "hop_native": 160, "align_mode": "gather", "model_ckpt": PRETRAINED["cvec"]}),
}


def main():
    import torch

    ds, total_dur = _stage()
    print(f"dataset: {N} files, {total_dur:.1f}s audio ({total_dur/60:.1f}min)\n")

    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    report = {}
    for name, (kind, params) in CONFIGS.items():
        for dev in devices:
            # mel/rms/spec on GPU is pointless (CPU-faster); skip GPU for rms
            if dev == "cuda" and name == "rms":
                continue
            print(f"[{name} / {dev}] running...", flush=True)
            r = run_feature(kind, params, dev, ds)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                report[f"{name}@{dev}"] = r
                continue
            report[f"{name}@{dev}"] = r
            print(
                f"  {r['elapsed_s']:6.1f}s | "
                f"CPU {r['cpu_total_avg_pct']:5.1f}% (p50 {r['cpu_total_p50']}%, ~{r['cpu_cores_used_avg']} cores busy, max-core {r['cpu_core_max_pct']}%) | "
                f"GPU {r['gpu_avg_pct']:5.1f}% (p50 {r['gpu_p50_pct']}%, mem {r['gpu_mem_mb']:.0f}MB)"
            )

    out = REPO / "util_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n=== report -> {out} ===")


if __name__ == "__main__":
    main()
