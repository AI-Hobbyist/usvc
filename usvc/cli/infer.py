"""usvc-infer CLI: offline voice conversion for any backend.

Usage:
    usvc-infer input.wav --config configs/sovits.yaml --ckpt ck.ckpt -o out.wav
    usvc-infer input.wav --config configs/diffusion.yaml --ckpt ck.ckpt \
        --vocoder-dir pretrained/nsf_hifigan --spk 0 --trans 0 --steps 50
    usvc-infer input.wav --config configs/rvc.yaml --ckpt ck.ckpt \
        --index-path index/rvc.pt --index-ratio 0.75
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="usvc-infer",
        description="Offline SVC inference (all backends: rift_dit/ddsp6/"
                    "diffusion/sovits/rvc/reflow_vae).",
    )
    p.add_argument("input", help="input wav path")
    p.add_argument("--config", "-c", required=True, help="training config yaml")
    p.add_argument("--ckpt", required=True, help="model checkpoint (.ckpt)")
    p.add_argument("--output", "-o", default=None, help="output wav path (default: out_<input>_<backend>.wav)")
    p.add_argument("--vocoder-dir", default=None,
                   help="NSF-HiFiGAN dir with config.json + model.ckpt "
                        "(default: usvc/pretrained/nsf_hifigan; only mel-domain backends)")
    p.add_argument("--spk", type=int, default=0, help="speaker id")
    p.add_argument("--trans", type=int, default=0, help="pitch shift in semitones")
    p.add_argument("--steps", type=int, default=None, help="sampling steps (flow/diff backends)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true", help="use fp16 inference")
    p.add_argument("--index-path", default=None, help="retrieval index .pt (RVC style)")
    p.add_argument("--index-ratio", type=float, default=0.0, help="retrieval blend ratio (0-1)")
    p.add_argument("--cluster-path", default=None, help="cluster model .pt (so-vits style)")
    p.add_argument("--cluster-ratio", type=float, default=0.0, help="cluster blend ratio (0-1)")
    args = p.parse_args(argv)

    from ..infer.engine import InferenceEngine

    engine = InferenceEngine(
        args.config, args.ckpt, vocoder_dir=args.vocoder_dir,
        device=args.device, half=args.half,
    )

    output = args.output
    if output is None:
        in_path = Path(args.input)
        output = f"out_{in_path.stem}_{engine.backend_name}.wav"

    engine.convert_file(
        args.input,
        output,
        spk=args.spk,
        trans=args.trans,
        index_path=args.index_path,
        index_ratio=args.index_ratio,
        cluster_path=args.cluster_path,
        cluster_ratio=args.cluster_ratio,
        steps=args.steps,
    )
    print(f"[usvc-infer] {args.input} -> {output} "
          f"(backend={engine.backend_name}, sr={engine.sr}, half={args.half})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
