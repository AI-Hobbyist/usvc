"""usvc-preprocess CLI.

Usage:
    usvc-preprocess init <dataset_dir> --src <audio_src> --sr 44100 --hop 512
    usvc-preprocess run <dataset_dir> --backend rift_dit [--force] [--device cuda]
    usvc-preprocess run <dataset_dir> --features mel,f0,rms"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..backends.ddsp6 import DDSP6Backend
from ..backends.diffusion import DiffusionBackend
from ..backends.reflow_shallow import ReflowShallowBackend
from ..backends.reflow_vae import ReflowVAEBackend
from ..backends.rift_dit import RIFTDiTBackend
from ..backends.rvc import RVCBackend
from ..backends.sovits import SoVitsBackend
from ..data.engine import PreprocessEngine, prepare_audio_dir
from ..data.feature_config import FeatureConfig, FeatureSpec
from ..data.meta import init_dataset_meta, load_meta, save_meta

_BACKENDS = {
    "rift_dit": RIFTDiTBackend,
    "ddsp6": DDSP6Backend,
    "diffusion": DiffusionBackend,
    "sovits": SoVitsBackend,
    "rvc": RVCBackend,
    "reflow_vae": ReflowVAEBackend,
    "reflow_shallow": ReflowShallowBackend,
}

def _resolve_specs(args) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for name in args.backend or []:
        if name not in _BACKENDS: raise SystemExit(f"unknown backend: {name}")
        specs.append(_BACKENDS[name].feature_spec(args.train_cfg or {}))
    if args.features:
        kind_defaults = {
            "mel": FeatureConfig("mel", {"n_fft": 2048, "win_length": 2048, "n_mels": 128,
                                         "fmin": 40.0, "fmax": 16000.0, "center": False}),
            "f0": FeatureConfig("f0", {"extractor": "fcpe", "uv_policy": "raw",
                                       "model_ckpt": "pretrained/rmvpe/model.pt"}),
            "rms": FeatureConfig("rms", {"win_length": 2048}),
        }
        kw = {f.strip(): kind_defaults[f.strip()] for f in args.features.split(",") if f.strip() in kind_defaults}
        if kw: specs.append(FeatureSpec(**kw))
    if not specs: raise SystemExit("specify --backend and/or --features")
    return specs

def _cmd_init(args) -> None:
    dataset_dir = Path(args.dataset_dir)
    sr, hop = args.sr, args.hop
    previous = load_meta(dataset_dir) if (dataset_dir / "meta.json").exists() else None
    discovered = prepare_audio_dir(args.src, dataset_dir, sr=sr, extensions=tuple(args.ext))

    from ..data.meta import AudioEntry
    from ..io.audio import frame_len_for, load_audio

    meta = init_dataset_meta(dataset_dir, sr=sr, hop=hop, f0_min=float(args.f0_min), f0_max=float(args.f0_max))
    meta.speakers = sorted({spk for spk, _ in discovered})
    if previous is not None:
        if set(previous.speakers) != set(meta.speakers):
            raise ValueError("speaker set changed; refusing to replace fixed dataset identities")
        meta.speakers = previous.speakers
    meta.audios = []
    for spk, file_name in discovered:
        n = len(load_audio(dataset_dir / "audio" / spk / file_name, sr=sr))
        meta.audios.append(AudioEntry(speaker=spk, file_name=file_name,
                                      frame_len=frame_len_for(n, hop), duration=n / sr,
                                      audio_samples=n))
    save_meta(meta)
    print(f"init: staged {len(discovered)} clips, {len(meta.speakers)} speakers -> {dataset_dir}")

def _cmd_run(args) -> None:
    PreprocessEngine(args.dataset_dir).run(_resolve_specs(args), device=args.device,
                                           force=args.force, dtype=args.dtype)
    meta = load_meta(args.dataset_dir)
    print("preprocess done. registered features:")
    for kind, val in meta.features.items():
        for enc, entries in (val.items() if kind == "units" else ((None, val),)):
            for e in entries:
                tag = f"units/{enc}" if enc else kind
                print(f"  {tag}: {e['hash']} (count={e['count']})")

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="svc-preprocess")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="stage audio + init meta.json")
    pi.add_argument("dataset_dir")
    pi.add_argument("--src", required=True)
    pi.add_argument("--sr", type=int, default=44100)
    pi.add_argument("--hop", type=int, default=512)
    pi.add_argument("--f0-min", default=65.0); pi.add_argument("--f0-max", default=800.0)
    pi.add_argument("--ext", nargs="+", default=["wav", "flac", "mp3", "ogg"])
    pi.set_defaults(func=_cmd_init)

    pr = sub.add_parser("run", help="compute features")
    pr.add_argument("dataset_dir")
    pr.add_argument("--backend", nargs="*", default=None); pr.add_argument("--features", default=None, help="comma list: mel,f0,rms,units")
    pr.add_argument("--device", default="cuda" if _cuda() else "cpu")
    pr.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="weight dtype for neural extractors (contentvec/f0); fp32=exact, fp16/bf16=faster")
    pr.add_argument("--force", action="store_true"); pr.add_argument("--train-cfg", dest="train_cfg", default=None, help="path to train yaml (units override)")
    pr.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    # load optional train cfg: mirror TrainConfig.from_yaml by merging top-level feature override
    # keys into backend_opts so preprocessing and training see the SAME feature config
    raw = {}
    if getattr(args, "train_cfg", None):
        import yaml
        with open(args.train_cfg, encoding="utf-8") as f: raw = yaml.safe_load(f) or {}
        backend_opts = dict(raw.get("backend_opts", {}) or {})
        for k in ("units_kind", "units_ckpt", "f0_extractor", "rmvpe_ckpt", "uv_policy", "n_mels",
                  "fmin", "fmax", "rvc_version", "version", "use_pitch_aug"):
            if k in raw and k not in backend_opts: backend_opts[k] = raw[k]
        args.train_cfg = backend_opts
    else:
        args.train_cfg = {}
    args.func(args)
    return 0

def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

if __name__ == "__main__":
    sys.exit(main())
