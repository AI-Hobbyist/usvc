"""Real 1-epoch training loss comparison: usvc reflow_vae vs ReFlow-VAE-SVC.

Validates the reflow_vae backend at the same level as the other 5 backends
(rift_dit 0.0%, ddsp6 0.0%, diffusion 0.1%, sovits 0.2%, rvc 3.9% mean loss
diff vs original projects over 1 epoch).

Protocol (identical on both sides, mirroring ReFlow-VAE-SVC solver.py):
  - fp32, AdamW(lr=2e-4, weight_decay=0, betas=(0.9,0.999), eps=1e-8)
  - clip_grad_value_(1.0), no scheduler (StepLR decay_step=100000 never
    triggers within a 13-step epoch)
  - every step reseeds torch RNG with manual_seed(seed + step_idx) so the
    two stochastic sources (VAE randn_like noise in Unit2Wav_VAE.forward,
    t = torch.rand in Bi_RectifiedFlow.forward) consume identical random
    streams on both sides -> per-step losses must match to float error
  - identical data order: torch.randperm with manual_seed(seed)

The default comparison drives usvc's own ReflowVAEModel (the verbatim port)
in a manual loop, exactly like real_training_compare.py does for ddsp6 /
diffusion / sovits / rvc (train_*_usvc call the usvc model directly).  With
--lightning, an additional smoke run drives ReflowVAEModule through the full
pytorch-lightning stack to prove the training pipeline is usable end-to-end.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

REF_SVC_PATH = "/root/ReFlow-VAE-SVC"


def make_data(n=200, T=64, seed=777):
    g = torch.Generator().manual_seed(seed)
    return [{
        "mel": torch.randn(T, 128, generator=g) * 2,
        "f0": torch.rand(T, generator=g) * 400 + 100,
        "rms": torch.rand(T, generator=g),
        "units": torch.randn(T, 768, generator=g),
    } for _ in range(n)]


def _order(n, seed):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()


def _batch(samples, order, start, bs, device):
    idx = order[start:start + bs]
    return {
        "units": torch.stack([samples[i]["units"] for i in idx]).to(device),
        "f0": torch.stack([samples[i]["f0"] for i in idx]).to(device),      # [B,T]
        "rms": torch.stack([samples[i]["rms"] for i in idx]).to(device),    # [B,T]
        "mel": torch.stack([samples[i]["mel"] for i in idx]).to(device),    # [B,T,128]
    }


def _step_seed(seed, step):
    return seed + step


# ---------------------------------------------------------------------------
# usvc side (default: manual loop on the verbatim-ported model)
# ---------------------------------------------------------------------------
def train_reflow_vae_usvc(samples, seed=42, bs=16, device="cuda"):
    """usvc ReflowVAEModel in the original project's training protocol."""
    from usvc.backends.reflow_vae.model import ReflowVAEModel

    torch.manual_seed(seed)
    model = ReflowVAEModel(
        n_units=768, n_spk=1, win_length=2048, sampling_rate=44100,
        block_size=512, use_pitch_aug=False, out_dims=128, n_layers=6,
        n_chans=512, n_hidden=256, back_bone="lynxnet", use_attention=True,
        t_start=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    order = _order(len(samples), seed)
    losses = []
    model.train()
    for step, start in enumerate(range(0, len(order), bs)):
        torch.manual_seed(_step_seed(seed, step))  # align RNG with orig side
        b = _batch(samples, order, start, bs, device)
        spk = torch.zeros(bs, dtype=torch.long, device=device)
        loss, _sub = model(b["units"], b["f0"], b["rms"], spk, b["mel"],
                           aug_shift=None)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_reflow_vae_usvc_lightning(samples, seed=42, bs=16):
    """Full usvc training stack (ReflowVAEModule + pytorch-lightning)."""
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, Dataset

    from usvc.backends.reflow_vae.module import ReflowVAEModule
    from usvc.train.config import OptimizerConfig, TrainConfig
    from usvc.train.dataset import svc_collate

    class DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            d = samples[i]
            return {**d, "frame_len": d["mel"].shape[0], "spk_id": 0}

    pl.seed_everything(seed, workers=True)
    cfg = TrainConfig(backend="reflow_vae", batch_size=bs, precision="32-true")
    cfg.backend_opts = {
        "back_bone": "lynxnet", "n_layers": 6, "n_chans": 512, "n_hidden": 256,
        "use_attention": True, "use_pitch_aug": False, "out_dims": 128,
        "n_units": 768, "num_speaker": 1, "t_start": 0.0,
    }
    cfg.optimizer = OptimizerConfig(
        type="adamw", lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0)
    module = ReflowVAEModule(cfg)
    dl = DataLoader(DS(), batch_size=bs, shuffle=False,
                    collate_fn=svc_collate, num_workers=0)

    losses = []

    class Tracker(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            l = trainer.callback_metrics.get("train/loss")
            if l is not None:
                losses.append(float(l))

    trainer = pl.Trainer(
        max_epochs=1, accelerator="auto", devices=1, precision="32-true",
        logger=False, enable_progress_bar=False, callbacks=[Tracker()],
        num_sanity_val_steps=0,
    )
    trainer.fit(module, dl)
    return losses


# ---------------------------------------------------------------------------
# original project side
# ---------------------------------------------------------------------------
def train_reflow_vae_orig(samples, seed=42, bs=16, device="cuda"):
    """ReFlow-VAE-SVC Unit2Wav_VAE + solver.py training protocol."""
    if REF_SVC_PATH not in sys.path[:1]:
        sys.path.insert(0, REF_SVC_PATH)
    from reflow.vocoder import Unit2Wav_VAE

    torch.manual_seed(seed)
    model = Unit2Wav_VAE(
        44100, 512, 2048, 768, 1, False, 128, 6, 512, 256, "lynxnet", True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    order = _order(len(samples), seed)
    losses = []
    model.train()
    for step, start in enumerate(range(0, len(order), bs)):
        torch.manual_seed(_step_seed(seed, step))  # align RNG with usvc side
        b = _batch(samples, order, start, bs, device)
        spk = torch.zeros(bs, 1, dtype=torch.long, device=device)
        # original project expects [B,T,1] f0/volume; spk unused when n_spk=1
        loss = model(
            b["units"], b["f0"].unsqueeze(-1), b["rms"].unsqueeze(-1),
            spk_id=spk, aug_shift=None, vocoder=None,
            gt_spec=b["mel"], infer=False,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def compare(name, u, o):
    n = min(len(u), len(o))
    diffs = [abs(u[i] - o[i]) / max(abs(o[i]), 1e-9) * 100 for i in range(n)]
    mean_diff = float(np.mean(diffs))
    mu, mo = float(np.mean(u)), float(np.mean(o))
    md = abs(mu - mo) / abs(mo) * 100
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"  usvc: {len(u)} steps, first={u[0]:.6f}, last={u[-1]:.6f}, mean={mu:.6f}")
    print(f"  orig: {len(o)} steps, first={o[0]:.6f}, last={o[-1]:.6f}, mean={mo:.6f}")
    print(f"  per-step diff: mean={mean_diff:.4f}%, max={max(diffs):.4f}%")
    print(f"  mean loss diff: {md:.2f}%")
    ok = mean_diff < 1.0
    print(f"  -> {'PASS (per-step < 1%)' if ok else 'CHECK (>= 1%)'}")
    return {"per_step_mean_pct": mean_diff, "mean_diff_pct": md, "ok": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lightning", action="store_true",
                    help="also smoke-run usvc ReflowVAEModule via pl.Trainer")
    args = ap.parse_args()

    samples = make_data(n=200, T=64, seed=777)
    print("=== reflow_vae 1-epoch training comparison: usvc vs ReFlow-VAE-SVC ===")
    print(f"data: n={len(samples)}, T=64, bs={args.bs}, seed={args.seed}, fp32")

    usvc_losses = train_reflow_vae_usvc(samples, seed=args.seed, bs=args.bs)
    orig_losses = train_reflow_vae_orig(samples, seed=args.seed, bs=args.bs)
    r1 = compare("usvc ReflowVAEModel (manual loop) vs original Unit2Wav_VAE",
                 usvc_losses, orig_losses)

    if args.lightning:
        lit_losses = train_reflow_vae_usvc_lightning(samples, seed=args.seed,
                                                     bs=args.bs)
        r2 = compare("usvc ReflowVAEModule (pl.Trainer stack) vs original",
                     lit_losses, orig_losses)
        print(f"\nlightning stack: {len(lit_losses)} steps, "
              f"first={lit_losses[0]:.4f}, last={lit_losses[-1]:.4f} "
              f"(loss decreasing = training pipeline works)")

    print("\n=== summary ===")
    print(f"manual-loop per-step diff: {r1['per_step_mean_pct']:.4f}% "
          f"({'PASS' if r1['ok'] else 'FAIL'})")
    if args.lightning:
        print(f"lightning stack mean diff: {r2['mean_diff_pct']:.2f}%")
    return 0 if r1["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
