"""End-to-end inference verification for all 6 backends.

Runs offline wav->wav conversion for every backend through the real
InferenceEngine (feature extraction -> model -> vocoder/generator), checks
output length / NaN, and reports per-backend PASS/FAIL.

Usage:
    python benchmarks/verify_infer_e2e.py [--device cuda] [--half]

Notes:
    - rift_dit uses a real trained checkpoint (checkpoints/epoch=0-step=13.ckpt,
      dim=256/depth=4/cvec=768); other backends use freshly-initialized
      random weights (smoke test: shapes/flow, not quality).
    - sovits uses hubertsoft units (256-dim), rvc uses contentvec (768-dim),
      mel-domain backends use the shared openvpi NSF-HiFiGAN vocoder.
"""
from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def synth_test_wav(path, sr=44100, duration=2.0, f0_base=220.0):
    import soundfile as sf

    t = np.arange(int(sr * duration)) / sr
    vib = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0_base + vib) / sr
    wav = 0.5 * np.sin(phase) * np.exp(-t / duration)
    sf.write(path, wav.astype(np.float32), sr)
    return wav


def make_ckpt_yaml(tmp: Path, backend: str, opts: dict, ckpt_name: str):
    """Build a random-weight module ckpt + matching config yaml."""
    from usvc.train.config import TrainConfig

    cfg = TrainConfig(backend=backend, backend_opts=opts)
    mod = _build_module(backend, cfg)
    sd = {k: v for k, v in mod.state_dict().items()}
    ckpt_path = tmp / ckpt_name
    torch.save({"state_dict": sd}, ckpt_path)
    yaml_path = tmp / f"{backend}_test.yaml"
    yaml.safe_dump(
        {"backend": backend, "dataset_dir": str(tmp / "ds"),
         "exp_dir": str(tmp / "exp"), "backend_opts": opts},
        open(yaml_path, "w"),
    )
    return str(yaml_path), str(ckpt_path)


def _build_module(backend: str, cfg):
    if backend == "rift_dit":
        from usvc.backends.rift_dit.module import RIFTModule
        return RIFTModule(cfg)
    if backend == "ddsp6":
        from usvc.backends.ddsp6.module import DDSPModule
        return DDSPModule(cfg)
    if backend == "diffusion":
        from usvc.backends.diffusion.module import DiffusionModule
        return DiffusionModule(cfg)
    if backend == "reflow_vae":
        from usvc.backends.reflow_vae.module import ReflowVAEModule
        return ReflowVAEModule(cfg)
    if backend == "sovits":
        from usvc.backends.sovits.module import SoVITSModule
        return SoVITSModule(cfg)
    if backend == "rvc":
        from usvc.backends.rvc.module import RVCModule
        return RVCModule(cfg)
    raise ValueError(backend)


BACKEND_OPTS = {
    "rift_dit": {"dim": 256, "depth": 4, "head_dim": 64, "cvec_dim": 768,
                 "n_mel_channels": 128, "num_speaker": 1,
                 "time_schedule": "lognorm", "loss_type": "l2",
                 "sampling_rate": 44100, "hop_length": 512},
    "ddsp6": {"n_units": 768, "num_speaker": 1, "n_aux_layers": 3,
              "n_aux_chans": 256, "n_layers": 4, "n_chans": 512,
              "velocity_net": "LYNXNet2", "t_start": 0.0, "use_norm": True,
              "version": "6.3", "sampling_rate": 44100, "block_size": 512},
    "diffusion": {"n_units": 768, "num_speaker": 1, "out_dims": 128,
                  "n_hidden": 256, "denoiser": "convnext", "cn_layers": 4,
                  "cn_chans": 256, "sampling_rate": 44100, "hop_length": 512},
    "reflow_vae": {"n_units": 768, "num_speaker": 1, "out_dims": 128,
                   "n_hidden": 256, "back_bone": "lynxnet", "n_layers": 2,
                   "n_chans": 256, "use_attention": True,
                   "use_pitch_aug": False, "sampling_rate": 44100,
                   "block_size": 512},
    "sovits": {"n_units": 256, "spec_channels": 1025, "inter_channels": 192,
               "hidden_channels": 192, "filter_channels": 768, "n_heads": 2,
               "n_layers": 6, "gin_channels": 256, "num_speaker": 1,
               "use_automatic_f0_prediction": True, "sampling_rate": 44100,
               "hop_length": 512},
    "rvc": {"n_units": 768, "spec_channels": 1025, "inter_channels": 192,
            "hidden_channels": 192, "filter_channels": 768, "n_heads": 2,
            "n_layers": 6, "gin_channels": 256, "num_speaker": 1,
            "spk_embed_dim": 1, "sr": 48000, "hop": 480},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    from usvc.infer.engine import InferenceEngine

    tmp = Path(tempfile.mkdtemp(prefix="usvc_infer_"))
    input_wav = tmp / "in.wav"
    synth_test_wav(input_wav)

    # rift_dit: use the real trained checkpoint if present.
    rift_ckpt = ROOT / "checkpoints" / "epoch=0-step=13.ckpt"
    rift_opts = BACKEND_OPTS["rift_dit"]
    rift_yaml = tmp / "rift_dit_real.yaml"
    yaml.safe_dump(
        {"backend": "rift_dit", "dataset_dir": str(tmp / "ds"),
         "exp_dir": str(tmp / "exp"), "backend_opts": rift_opts},
        open(rift_yaml, "w"),
    )

    results = {}
    for backend in ["rift_dit", "ddsp6", "diffusion", "reflow_vae", "sovits", "rvc"]:
        try:
            if backend == "rift_dit" and rift_ckpt.exists():
                yaml_path, ckpt_path = str(rift_yaml), str(rift_ckpt)
            else:
                yaml_path, ckpt_path = make_ckpt_yaml(
                    tmp, backend, BACKEND_OPTS[backend], f"{backend}.ckpt"
                )
            eng = InferenceEngine(yaml_path, ckpt_path, device=args.device,
                                  half=args.half)
            wav = eng.convert(str(input_wav), spk=0, trans=0)
            ok = (not np.isnan(wav).any()) and len(wav) > 40000
            results[backend] = (ok, len(wav), float(np.abs(wav).mean()),
                                len(wav) / eng.sr)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {backend:<11} wav={len(wav):>7} ({len(wav)/eng.sr:.2f}s) "
                  f"abs_mean={np.abs(wav).mean():.4f} nan={np.isnan(wav).any()}")
        except Exception as e:
            results[backend] = (False, 0, 0.0, 0.0)
            print(f"[FAIL] {backend:<11} {type(e).__name__}: {str(e)[:160]}")

    n_pass = sum(1 for ok, *_ in results.values() if ok)
    print(f"\n{'-' * 60}\n{len(results)} backends: {n_pass} PASS / {len(results) - n_pass} FAIL")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
