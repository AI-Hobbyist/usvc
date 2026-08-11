"""usvc-export CLI: extract inference-only weights from a training checkpoint.

Training checkpoints include optimizer states, discriminators, and Lightning
metadata that inference does not need. This command strips them, producing a
compact inference weight file that loads faster and is ~10-17x smaller.

Usage:
    usvc-export --ckpt experiments/nahida_rvc/last.ckpt -o nahida_rvc.pth
    usvc-export --ckpt last.ckpt -o model.pth --half      # fp16 (half size)
    usvc-export --ckpt last.ckpt -o model.pth             # fp32 (full precision)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="usvc-export",
        description="Extract inference-only weights from a training checkpoint "
                    "(strips optimizer + discriminator + metadata).",
    )
    p.add_argument("--ckpt", required=True, help="training checkpoint (.ckpt)")
    p.add_argument("--output", "-o", required=True, help="output inference weight (.pth)")
    p.add_argument("--half", action="store_true",
                   help="convert weights to fp16 (halves size; inference-only safe)")
    args = p.parse_args(argv)

    import torch

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.output)
    if not ckpt_path.exists():
        print(f"error: checkpoint not found: {ckpt_path}")
        return 1

    # Load training checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        print(f"error: cannot interpret checkpoint: {ckpt_path}")
        return 1

    # Drop training-only keys (discriminator / optimizer / criterion / loss / metrics).
    # Inference never loads them (usvc/infer/engine.py tolerates these prefixes via
    # strict=False), so exporting them would only bloat the file.
    drop_prefixes = ("discriminator.", "optimizer.", "criterion.", "loss.", "metrics.")
    infer_sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}
    dropped = len(sd) - len(infer_sd)

    # Optionally convert to half precision
    if args.half:
        infer_sd = {k: (v.half() if v.is_floating_point() else v) for k, v in infer_sd.items()}

    # Save as bare state_dict (no Lightning wrapper, no optimizer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(infer_sd, str(out_path))

    ckpt_file_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    out_file_mb = os.path.getsize(out_path) / 1024 / 1024
    dtype = "fp16" if args.half else "fp32"

    print(f"exported: {ckpt_path} ({ckpt_file_mb:.0f}MB) → {out_path} ({out_file_mb:.0f}MB, {dtype})")
    print(f"  keys: {len(sd)} → {len(infer_sd)} (dropped {dropped} discriminator keys)")
    print(f"  size: {ckpt_file_mb:.0f}MB → {out_file_mb:.0f}MB ({ckpt_file_mb/out_file_mb:.1f}x smaller)")
    print(f"\nusage: usvc-infer input.wav --config <config.yaml> --ckpt {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
