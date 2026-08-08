"""End-to-end offline inference verification for all 6 backends.

For each backend: build a small random-init module, save a lightning-style
ckpt, run InferenceEngine.convert on a synthetic wav, and check the output
has sane length / no NaN / valid audio range. The last two backends are
wav-domain (no external vocoder); mel-domain use the shared NSF-HiFiGAN.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(os.environ.get("USVC_BENCH_DIR") or (Path(tempfile.gettempdir()) / "usvc_infer"))

from usvc.infer.engine import InferenceEngine  # noqa: E402
from usvc.infer.features import synth_test_wav  # noqa: E402
from usvc.train.config import TrainConfig  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# backend -> (backend_opts, save as generator instead of model)
CONFIGS = {
    "rift_dit": {
        "backend_opts": {"dim": 256, "depth": 4, "head_dim": 64, "cvec_dim": 768,
                         "n_mel_channels": 128, "num_speaker": 1, "eval_sample_steps": 20},
        "gan": False,
    },
    "ddsp6": {
        "backend_opts": {"n_units": 768, "num_speaker": 1, "n_aux_layers": 2,
                         "n_aux_chans": 128, "n_layers": 2, "n_chans": 128,
                         "out_dims": 128},
        "gan": False,
    },
    "diffusion": {
        "backend_opts": {"n_units": 768, "num_speaker": 1, "out_dims": 128,
                         "n_hidden": 128, "cn_layers": 2, "cn_chans": 128,
                         "wn_layers": 2, "wn_chans": 128, "k_step": 1000},
        "gan": False,
    },
    "reflow_vae": {
        "backend_opts": {"n_units": 768, "num_speaker": 1, "out_dims": 128,
                         "n_layers": 2, "n_chans": 128, "n_hidden": 128,
                         "back_bone": "wavenet", "use_attention": False,
                         "win_length": 2048, "sampling_rate": 44100,
                         "block_size": 512},
        "gan": False,
    },
    "sovits": {
        "backend_opts": {"n_units": 768, "spec_channels": 1025, "num_speaker": 1,
                         "gin_channels": 768, "segment_size": 20,
                         "inter_channels": 64, "hidden_channels": 64,
                         "filter_channels": 128, "units_kind": "contentvec"},
        "gan": True,
    },
    "rvc": {
        "backend_opts": {"n_units": 768, "num_speaker": 1, "sr": 48000,
                         "segment_size": 20, "inter_channels": 64,
                         "hidden_channels": 64, "filter_channels": 128},
        "gan": True,
    },
}


def make_ckpt(backend: str, opts: dict, gan: bool, path: Path):
    cfg = TrainConfig(backend=backend)
    cfg.backend_opts = opts
    from usvc.backends import get_backend
    module = get_backend(backend)().build_module(cfg)
    sd = module.state_dict()
    torch.save({"state_dict": sd, "epoch": 0, "global_step": 0}, path)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_in = OUT_DIR / "test.wav"
    synth_test_wav(wav_in, sr=44100, duration=2.0, f0_base=200.0)
    print(f"test wav: {wav_in} ({wav_in.stat().st_size} bytes)")

    results = []
    for backend, spec in CONFIGS.items():
        print(f"\n=== {backend} ===")
        try:
            ckpt = OUT_DIR / f"{backend}.ckpt"
            make_ckpt(backend, spec["backend_opts"], spec["gan"], ckpt)
            cfg_path = OUT_DIR / f"{backend}.yaml"
            opts_lines = "\n".join(
                f"  {k}: {v}" for k, v in spec["backend_opts"].items()
            )
            cfg_path.write_text(
                f"backend: {backend}\ndataset_dir: {OUT_DIR}\nexp_dir: {OUT_DIR}\n"
                f"backend_opts:\n{opts_lines}\n"
            )
            sr = 48000 if backend == "rvc" else 44100
            engine = InferenceEngine(cfg_path, ckpt, device=DEVICE, half=False)
            out = engine.convert(str(wav_in), spk=0, trans=0)
            assert out.dtype == np.float32
            assert out.ndim == 1 and out.shape[0] > 1000
            assert not np.isnan(out).any()
            assert np.abs(out).max() <= 10.0, f"audio too loud: {np.abs(out).max()}"
            dur_in = 2.0
            dur_out = out.shape[0] / sr
            ratio = dur_out / dur_in
            ok_len = 0.6 < ratio < 1.6
            out_path = OUT_DIR / f"out_{backend}.wav"
            engine.convert_file(str(wav_in), str(out_path), spk=0, trans=0)
            print(f"  ✓ {backend}: {out.shape[0]} samples ({dur_out:.2f}s), "
                  f"len_ratio={ratio:.2f}, max={np.abs(out).max():.3f}")
            print(f"  ✓ wav written: {out_path} ({out_path.stat().st_size} bytes)")
            assert ok_len, f"length ratio {ratio} out of range"
            results.append((backend, "PASS"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((backend, f"FAIL: {e}"))

    print("\n" + "=" * 50)
    for b, r in results:
        print(f"  {b:<12} {r}")
    failed = [b for b, r in results if r != "PASS"]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("ALL BACKENDS PASSED")


if __name__ == "__main__":
    main()
