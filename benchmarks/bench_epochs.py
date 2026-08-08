"""Per-backend training throughput benchmark with real 10-epoch runs.

For rift_dit / ddsp6 / diffusion: runs full 10 epochs on real preprocessed
features from the staged feature dir (default: <repo>/data/u_opt).
For sovits / rvc: synthetic data, fixed step count (no real spec features).

Reports: ms/step, min/epoch, std (stability), GPU power/utilization.
Supports TF32 + AMP toggles for near-lossless optimization.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = Path(os.environ.get("USVC_FEAT_DIR") or (REPO / "data" / "u_opt"))
OUT_DIR = Path(os.environ.get("USVC_BENCH_DIR") or (Path(tempfile.gettempdir()) / "usvc_validate"))

# Per-backend epoch counts
EPOCHS = {"rift_dit": 10, "ddsp6": 10, "diffusion": 10, "sovits": 2, "rvc": 2, "reflow_vae": 10}


def gpu_power():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"], timeout=2).decode().strip()
        p, u, mu = [float(x) for x in out.split(",")]
        return {"power_w": p, "gpu_util_pct": u, "mem_util_pct": mu}
    except Exception:
        return {}


def load_real_dataset(n_max=200):
    """Load real mel/units/f0/rms from FEAT_DIR. Returns list of dicts."""
    import json as _json
    meta = _json.load(open(FEAT_DIR / "meta.json"))
    mel_hash = list((FEAT_DIR / "mel").iterdir())[0].name
    u_hash = list((FEAT_DIR / "units" / "contentvec").iterdir())[0].name
    items = []
    for a in meta["audios"]:
        spk = a["speaker"].replace("/", "_")
        fn = a["file_name"].replace(".wav", ".pt")
        mel_p = FEAT_DIR / "mel" / mel_hash / spk / fn
        u_p = FEAT_DIR / "units" / "contentvec" / u_hash / spk / fn
        if not (mel_p.exists() and u_p.exists()):
            continue
        items.append({"name": fn, "mel_p": mel_p, "u_p": u_p,
                      "frame_len": a["frame_len"]})
        if len(items) >= n_max:
            break
    # load tensors
    for it in items:
        it["mel"] = torch.load(it["mel_p"])  # [T, 128]
        it["units"] = torch.load(it["u_p"])  # [T, 768]
        T = it["mel"].shape[0]
        it["f0"] = torch.rand(T) * 400 + 100  # real f0 not critical for throughput
        it["rms"] = torch.rand(T)
    return items


def make_batches(items, batch_size, seg_len, n_batches=50):
    """Pre-build n_batches GPU batches by cycling through items (filtering short)."""
    out = []
    rng = torch.Generator().manual_seed(42)
    valid = [it for it in items if it["mel"].shape[0] >= seg_len]
    if len(valid) < batch_size:
        raise RuntimeError(f"only {len(valid)} clips >= {seg_len} frames, need >= {batch_size}")
    idx = 0
    for _ in range(n_batches):
        mels, units, f0s, rmss = [], [], [], []
        for _b in range(batch_size):
            it = valid[idx % len(valid)]; idx += 1
            T = it["mel"].shape[0]
            s = torch.randint(0, max(1, T - seg_len), (1,), generator=rng).item()
            mels.append(it["mel"][s:s + seg_len])
            units.append(it["units"][s:s + seg_len])
            f0s.append(it["f0"][s:s + seg_len])
            rmss.append(it["rms"][s:s + seg_len])
        out.append({
            "mel": torch.stack(mels).cuda(),
            "units": torch.stack(units).cuda(),
            "f0": torch.stack(f0s).cuda(),
            "rms": torch.stack(rmss).cuda(),
            "sid": torch.zeros(batch_size, dtype=torch.long).cuda(),
        })
    return out


class Timer:
    def __init__(self):
        self.times = []
    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self.t0)


def _make_cfg(backend, opts):
    """Build a TrainConfig for GAN module construction."""
    from usvc.train.config import TrainConfig
    cfg = TrainConfig(backend=backend)
    cfg.backend_opts = opts
    return cfg


def _make_svcbatch(B, T, gen, spec_ch=1025, hop=512, rvc=False):
    """Build a synthetic SVCBatch on CUDA for GAN training."""
    from usvc.train.dataset import SVCBatch
    import dataclasses
    wave_len = T * hop
    fields = {
        "units": torch.randn(B, T, 768, generator=gen),
        "f0": torch.rand(B, T, generator=gen) * 400 + 100,
        "mel": torch.randn(B, spec_ch, T, generator=gen),  # linear spec [B, spec_ch, T]
        "wav": torch.randn(B, 1, wave_len, generator=gen),
        "rms": torch.rand(B, T, generator=gen),
        "spk_id": torch.zeros(B, dtype=torch.long),
        "frame_len": torch.full((B,), T),
        "uv": torch.rand(B, T, generator=gen),
        "aug_shift": None,
        "pitch_token": None,
        "key": None,
    }
    if rvc:
        fields["pitch_token"] = torch.randint(1, 256, (B, T), generator=gen)
    b = SVCBatch(**fields)
    return SVCBatch(**{f.name: (getattr(b, f.name).cuda() if torch.is_tensor(getattr(b, f.name)) else getattr(b, f.name)) for f in dataclasses.fields(b)})


def bench_backend(name, epochs, use_amp=False, batch_override=None, seg_override=None):
    """Run N epochs, return timing stats."""
    results = {"name": name, "epochs": epochs, "amp": use_amp}
    step_times = []
    epoch_times = []

    if name == "rift_dit":
        from usvc.backends.rift_dit.model import RIFTDiTModel
        B = batch_override or 64
        T = seg_override or 256
        torch.manual_seed(42)
        m = RIFTDiTModel(dim=512, depth=8, head_dim=64, n_mel_channels=128,
                         num_speaker=1, cvec_dim=768).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4, fused=True)  # flow: fused kernel
        data = load_real_dataset(200)
        batches = make_batches(data, B, T)
        def run_epoch():
            for b in batches:
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _ = m(b["mel"], b["sid"], b["f0"], b["rms"], b["units"])
                loss.backward(); opt.step()
        loss_fn = lambda b: m(b["mel"], b["sid"], b["f0"], b["rms"], b["units"])[0]

    elif name == "ddsp6":
        from usvc.backends.ddsp6.model import DDSP6Model
        B = batch_override or 48
        T = seg_override or 256
        torch.manual_seed(42)
        m = DDSP6Model(n_units=768, n_spk=1, n_aux_layers=6, n_aux_chans=512,
                       n_layers=6, n_chans=1024, velocity_net="LYNXNet2").cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4, fused=True)  # flow: fused kernel
        data = load_real_dataset(200)
        batches = make_batches(data, B, T)
        def run_epoch():
            for b in batches:
                # ddsp6 needs spec [B,T,128], use mel as spec proxy
                spec = b["mel"]  # [B,T,128]
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _ = m(b["units"], b["f0"], b["rms"], b["sid"], spec)
                loss.backward(); opt.step()
        loss_fn = lambda b: m(b["units"], b["f0"], b["rms"], b["sid"], b["mel"])[0]

    elif name == "diffusion":
        from usvc.backends.diffusion.model import DiffusionModel
        B = batch_override or 48
        T = seg_override or 256
        torch.manual_seed(42)
        m = DiffusionModel(n_units=768, n_spk=1, out_dims=128, n_hidden=256,
                           cn_layers=20, cn_chans=384).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4, fused=True)  # flow: fused kernel
        data = load_real_dataset(200)
        batches = make_batches(data, B, T)
        def run_epoch():
            for b in batches:
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _ = m(b["units"], b["f0"], b["rms"], b["sid"], b["mel"])
                loss.backward(); opt.step()
        loss_fn = lambda b: m(b["units"], b["f0"], b["rms"], b["sid"], b["mel"])[0]

    elif name == "reflow_vae":
        # Bi-RectifiedFlow + VAE (ReFlow-VAE-SVC, lynxnet backbone),
        # flow class: real dataset, same pattern as diffusion.
        from usvc.backends.reflow_vae.model import ReflowVAEModel
        B = batch_override or 48
        T = seg_override or 256
        torch.manual_seed(42)
        # lynxnet backbone (对应 reflow-vae-lynxnet.yaml, 主流配置)
        m = ReflowVAEModel(n_units=768, n_spk=1, win_length=2048, sampling_rate=44100,
                           block_size=512, use_pitch_aug=False, out_dims=128,
                           n_layers=6, n_chans=512, n_hidden=256,
                           back_bone="lynxnet", use_attention=True).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.0, fused=True)
        data = load_real_dataset(200)
        batches = make_batches(data, B, T)
        def run_epoch():
            for b in batches:
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    # _fit_dims auto-normalizes f0/rms [B,T]->[B,T,1], spk_id [B]->[B,1]
                    loss, _ = m(b["units"], b["f0"], b["rms"], b["sid"], gt_spec=b["mel"])
                loss.backward(); opt.step()
        loss_fn = lambda b: m(b["units"], b["f0"], b["rms"], b["sid"], gt_spec=b["mel"])[0]

    elif name == "sovits":
        # Full GAN training: SoVITSModule (generator_loss + discriminator_loss)
        # includes loss_mel (STFT re-extraction) + loss_lf0 + MPD, matching original.
        # NOTE: strict alternating order is REQUIRED (D fwd->bwd->step, then
        # G fwd->bwd->step). Building both loss graphs before any optimizer step
        # fails autograd version check on the later backward: AdamW's step()
        # in-place mutates shared weight_norm params (twice per param with
        # default weight_decay), so a graph built before that step sees
        # "modified by an inplace operation" RuntimeError. See BaseGANModule.
        from usvc.backends.sovits.module import SoVITSModule
        B = batch_override or 6
        T = seg_override or 64  # frames; segment_size must be < T
        seg = 32  # segment_size in frames (original: segment_size//hop)
        torch.manual_seed(42)
        cfg = _make_cfg("sovits", {"segment_size": seg, "num_speaker": 200,
                                    "gin_channels": 256, "n_units": 768, "spec_channels": 1025})
        m = SoVITSModule(cfg).cuda()
        opt_g = torch.optim.AdamW(m.generator.parameters(), lr=2e-4, betas=(0.8, 0.99), fused=True)
        opt_d = torch.optim.AdamW(m.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99), fused=True)
        n_steps = 30
        g = torch.Generator().manual_seed(42)
        def make_synth():
            return _make_svcbatch(B, T, g, spec_ch=1025, hop=512)
        batches = [make_synth() for _ in range(n_steps)]
        def run_epoch():
            for b in batches:
                # forward generator once (with grad), shared by D and G
                # (aligned with so-vits-svc / BaseGANModule.training_step)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = m.forward_generator(b)
                    loss_d, _ = m.discriminator_loss(b, out)
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss_g, _ = m.generator_loss(b, out)
                opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        loss_fn = lambda b: 0

    elif name == "rvc":
        # Full GAN training: RVCModule (generator_loss + discriminator_loss)
        # (same strict alternating-order constraint as sovits, see above).
        from usvc.backends.rvc.module import RVCModule
        B = batch_override or 4
        T = seg_override or 64
        seg = 36  # rvc segment_size in frames
        torch.manual_seed(42)
        cfg = _make_cfg("rvc", {"segment_size": seg, "num_speaker": 200,
                                 "spk_embed_dim": 200, "n_units": 768, "spec_channels": 1025,
                                 "sr": 48000, "hop": 480})
        m = RVCModule(cfg).cuda()
        opt_g = torch.optim.AdamW(m.generator.parameters(), lr=2e-4, betas=(0.8, 0.99), fused=True)
        opt_d = torch.optim.AdamW(m.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99), fused=True)
        n_steps = 30
        g = torch.Generator().manual_seed(42)
        def make_synth():
            return _make_svcbatch(B, T, g, spec_ch=1025, hop=480, rvc=True)
        batches = [make_synth() for _ in range(n_steps)]
        def run_epoch():
            for b in batches:
                # forward generator once (with grad), shared by D and G
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = m.forward_generator(b)
                    loss_d, _ = m.discriminator_loss(b, out)
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss_g, _ = m.generator_loss(b, out)
                opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        loss_fn = lambda b: 0
    else:
        raise ValueError(name)

    # warmup
    run_epoch()

    # timed epochs
    for ep in range(epochs):
        t = Timer()
        with t:
            run_epoch()
        epoch_times.append(t.times[0])

    steps_per_epoch = len(batches)
    results["batch_size"] = B
    results["seg_len"] = T
    results["steps_per_epoch"] = steps_per_epoch
    results["epoch_times_s"] = [round(x, 3) for x in epoch_times]
    results["mean_epoch_s"] = round(statistics.mean(epoch_times), 3)
    results["std_epoch_s"] = round(statistics.stdev(epoch_times) if len(epoch_times) > 1 else 0, 4)
    results["ms_per_step"] = round(statistics.mean(epoch_times) / steps_per_epoch * 1000, 1)
    results["gpu"] = gpu_power()
    return results


# Per-backend optimal config: (batch, amp) — determined empirically.
# ALL backends: AMP bf16 is near-lossless (divergence <= training's own
# posterior-sampling noise; rigorously verified bf16-error / fp32-noise ratio
# < 0.75x for sovits, < 0.49x for rvc). Large batch for GAN throughput.
OPTIMAL = {
    "rift_dit":  {"batch": 64,  "amp": True},   # GPU-bound 100% util
    "ddsp6":    {"batch": 48,  "amp": True},   # GPU-bound 94% util
    "diffusion": {"batch": 48, "amp": True},   # GPU-bound 97% util
    "sovits":   {"batch": 48,  "amp": True},  # 完整GAN(含mel STFT+MPD+lf0), B=48 peak 10.5GB
    "rvc":      {"batch": 32,  "amp": True},  # 完整GAN(含mel STFT+MPD), B=32 peak 10.4GB
    "reflow_vae": {"batch": 48, "amp": True},  # flow类, 同 diffusion
}
# Baseline (original-project defaults) for before/after comparison
BASELINE = {
    "rift_dit":  {"batch": 64,  "amp": False},
    "ddsp6":    {"batch": 48,  "amp": False},
    "diffusion": {"batch": 48, "amp": False},
    "sovits":   {"batch": 6,   "amp": False},
    "rvc":      {"batch": 4,   "amp": False},
    "reflow_vae": {"batch": 48, "amp": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=list(EPOCHS))
    ap.add_argument("--mode", choices=["optimal", "baseline", "custom"], default="optimal")
    ap.add_argument("--amp", action="store_true", default=None)
    ap.add_argument("--tf32", action="store_true", default=True)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default=str(OUT_DIR / "throughput_epochs.json"))
    args = ap.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[tf32] enabled (matmul + cudnn)")
    # cudnn.benchmark disabled: pathological exhaustive search for some shapes.
    torch.backends.cudnn.benchmark = False

    all_results = []
    for name in args.backends:
        ep = EPOCHS[name]
        if args.mode == "optimal":
            cfg = OPTIMAL[name]
            use_amp, B = cfg["amp"], cfg["batch"]
        elif args.mode == "baseline":
            cfg = BASELINE[name]
            use_amp, B = cfg["amp"], cfg["batch"]
        else:
            use_amp = bool(args.amp)
            B = args.batch or (OPTIMAL[name]["batch"] if args.mode == "optimal" else BASELINE[name]["batch"])
        print(f"\n>>> {name}: {ep} epochs (mode={args.mode}, amp={use_amp}, B={B})...")
        try:
            r = bench_backend(name, ep, use_amp=use_amp, batch_override=B)
            r["mode"] = args.mode
            print(f"  mean_epoch={r['mean_epoch_s']}s, ms/step={r['ms_per_step']}, "
                  f"std={r['std_epoch_s']}, steps/ep={r['steps_per_epoch']}")
            all_results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "mode": args.mode, "error": str(e)})
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
