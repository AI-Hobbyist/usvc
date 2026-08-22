"""usvc-export CLI: strip optimizer/discriminator/metadata from a training checkpoint into a
compact inference weight (~10-17x smaller).
Usage: usvc-export --ckpt last.ckpt -o model.pth [--half]"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="usvc-export",
                                description="Extract inference-only weights from a training "
                                            "checkpoint (strips optimizer + discriminator + metadata).")
    p.add_argument("--ckpt", required=True, help="training checkpoint (.ckpt)")
    p.add_argument("--output", "-o", required=True, help="output inference weight (.pth)")
    p.add_argument("--half", action="store_true",
                   help="convert weights to fp16 (halves size; inference-only safe)")
    args = p.parse_args(argv)

    import torch

    ckpt_path, out_path = Path(args.ckpt), Path(args.output)
    if not ckpt_path.exists(): print(f"error: checkpoint not found: {ckpt_path}"); return 1
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        print(f"error: cannot interpret checkpoint: {ckpt_path}"); return 1
    # drop training-only keys; inference tolerates them via strict=False but they only bloat
    infer_sd = {k: v for k, v in sd.items()
                if not k.startswith(("discriminator.", "optimizer.", "criterion.", "loss.", "metrics."))}
    if args.half:
        infer_sd = {k: (v.half() if v.is_floating_point() else v) for k, v in infer_sd.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(infer_sd, str(out_path))  # bare state_dict, no Lightning wrapper

    mb = lambda f: os.path.getsize(f) / 1024 / 1024
    ckb, outb = mb(ckpt_path), mb(out_path)
    print(f"exported: {ckpt_path} ({ckb:.0f}MB) → {out_path} ({outb:.0f}MB, {'fp16' if args.half else 'fp32'})")
    print(f"  keys: {len(sd)} → {len(infer_sd)} (dropped {len(sd) - len(infer_sd)} discriminator keys)")
    print(f"  size: {ckb:.0f}MB → {outb:.0f}MB ({ckb / outb:.1f}x smaller)")
    print(f"\nusage: usvc-infer input.wav --config <config.yaml> --ckpt {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
