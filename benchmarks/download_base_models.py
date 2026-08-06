#!/usr/bin/env python3
"""Batch download + unpack SVC pretrained base models into usvc/pretrained/base_models/.

Sources:
  - SVCFusion on ModelScope (24 .sf_dlc files)
  - RIFT-SVC pretrained on HuggingFace (3 .ckpt files)
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

PROXY = "/etc/waas-script/proxy.sh"
BASE = Path("/root/usvc/pretrained/base_models")

# (output_subdir, filename, expected_mb)
SVCFUSION_MODELS = [
    # ddsp6.1
    ("ddsp6/6.1_1024_10", "ddsp6.1_1024_10.sf_dlc", 343),
    ("ddsp6/6.1_1024_10_attention", "ddsp6.1_1024_10_attention.sf_dlc", 352),
    ("ddsp6/6.1_1024_10_with_noise", "ddsp6.1_1024_10_with_noise.sf_dlc", 343),
    ("ddsp6/6.1_1024_10_with_noise_and_attention", "ddsp6.1_1024_10_with_noise_and_attention.sf_dlc", 352),
    ("ddsp6/6.1_2048_6", "ddsp6.1_2048_6.sf_dlc", 834),
    ("ddsp6/6.1_2048_6_attention", "ddsp6.1_2048_6_attention.sf_dlc", 843),
    ("ddsp6/6.1_2048_6_with_noise", "ddsp6.1_2048_6_with_noise.sf_dlc", 834),
    ("ddsp6/6.1_2048_6_with_noise_and_attention", "ddsp6.1_2048_6_with_noise_and_attention.sf_dlc", 843),
    # ddsp6.3
    ("ddsp6/6.3_10x1024_6x1024", "ddsp6.3_10x1024_6x1024.sf_dlc", 395),
    ("ddsp6/6.3_10x1024_6x512", "ddsp6.3_10x1024_6x512.sf_dlc", 280),
    ("ddsp6/6.3_6x1024_6x1024", "ddsp6.3_6x1024_6x1024.sf_dlc", 314),
    ("ddsp6/6.3_6x512_10x2048", "ddsp6.3_6x512_10x2048.sf_dlc", 989),
    ("ddsp6/6.3_6x512_6x1024", "ddsp6.3_6x512_6x1024.sf_dlc", 210),
    ("ddsp6/6.3_6x512_6x512", "ddsp6.3_6x512_6x512.sf_dlc", 94),
    ("ddsp6/6.3_6x512_6x768", "ddsp6.3_6x512_6x768.sf_dlc", 142),
    # ddsp6 cvec
    ("ddsp6/cvec_1024_12", "ddsp6_cvec_1024_12.sf_dlc", 390),
    ("ddsp6/cvec_512_6", "ddsp6_cvec_512_6.sf_dlc", 65),
    # reflow
    ("reflow_vae/contentvec768l12", "reflow_contentvec768l12.sf_dlc", 678),
    # rvc
    ("rvc/32k_f0", "rvc_32k_f0.sf_dlc", 207),
    ("rvc/40k_f0", "rvc_40k_f0.sf_dlc", 206),
    ("rvc/48k_f0", "rvc_48k_f0.sf_dlc", 208),
    # sovits
    ("sovits/diff_vec768l12", "sovits_diff_vec768l12.sf_dlc", 211),
    ("sovits/so-vits-svc_cvec768l12", "sovits_so-vits-svc cvec768l12.sf_dlc", 378),
    ("sovits/so-vits-svc_cvec768l12_BigDataset", "sovits_so-vits-svc_cvec768l12_BigDataset_Test_2024.12.21%231.sf_dlc", 378),
]

RIFT_DIT_MODELS = [
    ("rift_dit/dit-512-8.ckpt", "pretrain-v3_dit-512-8.ckpt", 153),
    ("rift_dit/dit-768-12.ckpt", "pretrain-v3_dit-768-12.ckpt", 505),
    ("rift_dit/dit-1024-16.ckpt", "pretrain-v3_dit-1024-16.ckpt", 1184),
]

MS_URL = "https://www.modelscope.cn/models/aiguoliuguo/SVCFusion/resolve/master/base_models/{}"
HF_URL = "https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/{}"


def curl_download(url: str, dest: Path, timeout: int = 600) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = ["curl", "-sL", "--retry", "3", "--retry-delay", "2", "-o", str(tmp), url]
    # proxy for HF
    if "huggingface.co" in url and Path(PROXY).exists():
        src = subprocess.run(["bash", "-c", f"source {PROXY} >/dev/null 2>&1; echo $https_proxy"],
                             capture_output=True, text=True).stdout.strip()
        if src:
            cmd = ["curl", "-sL", "--retry", "3", "-x", src, "-o", str(tmp), url]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            print(f"  !! curl failed (rc={r.returncode})")
            return False
        tmp.rename(dest)
        return True
    except subprocess.TimeoutExpired:
        print(f"  !! timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"  !! error: {e}")
        return False


def unpack_sf_dlc(sf_path: Path, out_dir: Path) -> bool:
    """Unpack a .sf_dlc into {out_dir}/model.pt + config.yaml."""
    try:
        ck = torch.load(sf_path, map_location="cpu", weights_only=False)
        files = ck.get("files", {})
        if not files:
            print(f"  !! no 'files' in sf_dlc")
            return False
        out_dir.mkdir(parents=True, exist_ok=True)
        # config.yaml / meta.yaml (str or bytes)
        for name in ("config.yaml", "meta.yaml"):
            if name in files:
                data = files[name]
                if isinstance(data, bytes):
                    data = data.decode("utf-8", "replace")
                (out_dir / name).write_text(data)
        # model_0.pt (bytes or dict) -> model.pt
        if "model_0.pt" in files:
            m = files["model_0.pt"]
            if isinstance(m, dict):
                inner = m  # torch.load already unpacked it
            elif isinstance(m, str):
                inner = torch.load(io.BytesIO(m.encode()), map_location="cpu", weights_only=False)
            else:
                inner = torch.load(io.BytesIO(m), map_location="cpu", weights_only=False)
            torch.save(inner, out_dir / "model.pt")
            n = len(inner.get("model", {})) if isinstance(inner, dict) else 0
            print(f"  -> model.pt ({n} params)")
        elif "model.ckpt" in files:
            m = files["model.ckpt"]
            if isinstance(m, dict):
                inner = m
            elif isinstance(m, str):
                inner = torch.load(io.BytesIO(m.encode()), map_location="cpu", weights_only=False)
            else:
                inner = torch.load(io.BytesIO(m), map_location="cpu", weights_only=False)
            torch.save(inner, out_dir / "model.pt")
            n = len(inner.get("model", inner)) if isinstance(inner, dict) else 0
            print(f"  -> model.pt from model.ckpt ({n} params)")
        else:
            print(f"  !! unknown files keys: {list(files.keys())}")
            return False
        return True
    except Exception as e:
        print(f"  !! unpack error: {e}")
        return False


def main():
    os.makedirs(BASE, exist_ok=True)
    results = []

    # ---- phase 1: SVCFusion .sf_dlc ----
    print(f"=== SVCFusion: {len(SVCFUSION_MODELS)} models ===")
    for i, (out_sub, fname, _mb) in enumerate(SVCFUSION_MODELS, 1):
        out_dir = BASE / out_sub
        if (out_dir / "model.pt").exists():
            print(f"[{i:>2}/{len(SVCFUSION_MODELS)}] SKIP {out_sub} (already exists)")
            results.append((out_sub, "skip"))
            continue
        dest = BASE / fname
        print(f"[{i:>2}/{len(SVCFUSION_MODELS)}] DOWNLOAD {fname}")
        if not curl_download(MS_URL.format(fname), dest):
            print(f"  !! FAILED download {fname}")
            results.append((out_sub, "download-fail"))
            continue
        print(f"  downloaded {dest.stat().st_size / 1024 / 1024:.1f} MB")
        if unpack_sf_dlc(dest, out_dir):
            print(f"  OK {out_sub}")
            results.append((out_sub, "ok"))
        else:
            print(f"  !! FAILED unpack {fname}")
            results.append((out_sub, "unpack-fail"))
        dest.unlink(missing_ok=True)  # free space

    # ---- phase 2: RIFT-DiT HF ckpt ----
    print(f"\n=== RIFT-DiT: {len(RIFT_DIT_MODELS)} models (HF) ===")
    for i, (out_rel, fname, _mb) in enumerate(RIFT_DIT_MODELS, 1):
        dest = BASE / out_rel
        if dest.exists():
            print(f"[{i:>2}/{len(RIFT_DIT_MODELS)}] SKIP {out_rel} (already exists)")
            results.append((out_rel, "skip"))
            continue
        print(f"[{i:>2}/{len(RIFT_DIT_MODELS)}] DOWNLOAD {fname}")
        if curl_download(HF_URL.format(fname), dest):
            print(f"  OK {out_rel} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            results.append((out_rel, "ok"))
        else:
            print(f"  !! FAILED {fname}")
            results.append((out_rel, "fail"))

    # ---- summary ----
    ok = [r for r in results if r[1] == "ok"]
    failed = [r for r in results if r[1] not in ("ok", "skip")]
    print(f"\n=== SUMMARY ===")
    print(f"ok: {len(ok)}, skip: {len(results) - len(ok) - len(failed)}, failed: {len(failed)}")
    for r in failed:
        print(f"  FAILED: {r[0]} ({r[1]})")
    print(f"\nresults saved to /tmp/download_results.json")
    json.dump(results, open("/tmp/download_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
