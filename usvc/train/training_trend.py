"""Realistic training loss trend comparison: 1 full epoch, compare moving averages.

This measures whether usvc and the original project produce *comparable* loss
trends (not exact overlap — flow/diffusion loss is stochastic due to random t).
Metric: moving-average loss curve, check if they track each other.

We compare the loss *distribution* and *trend direction*, not per-step diff.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset


def moving_avg(x, w=20):
    """Simple moving average."""
    if len(x) < w:
        return [float(np.mean(x))]
    return [float(np.mean(x[i:i+w])) for i in range(0, len(x)-w+1, max(1, w//4))]


class LossTracker(pl.Callback):
    def __init__(self):
        self.losses = []
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        m = trainer.callback_metrics
        loss = m.get("train/loss") or m.get("train/loss_g")
        if loss is not None:
            self.losses.append(float(loss))


def make_fixed_dataset(n=200, T=64, seed=777):
    """Deterministic dataset from real-ish distribution."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n):
        samples.append({
            "mel": torch.randn(T, 128, generator=g) * 2,
            "f0": torch.rand(T, generator=g) * 400 + 100,
            "rms": torch.rand(T, generator=g),
            "units": torch.randn(T, 768, generator=g),
            "spk_id": 0,
        })
    return samples


class FixedDS(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        d = self.s[i]
        return {**d, "frame_len": d["mel"].shape[0]}


def train_usvc_rift(samples, seed=42):
    from usvc.train.config import TrainConfig, OptimizerConfig
    from usvc.train.dataset import svc_collate
    from usvc.backends.rift_dit.module import RIFTModule

    torch.manual_seed(seed)
    pl.seed_everything(seed)
    cfg = TrainConfig(backend="rift_dit")
    cfg.backend_opts = {"dim": 256, "depth": 4, "head_dim": 64, "cvec_dim": 768,
                        "n_mel_channels": 128, "num_speaker": 1}
    cfg.optimizer = OptimizerConfig(type="adamw", lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    cfg.grad_clip = 1.0

    module = RIFTModule(cfg)
    dl = DataLoader(FixedDS(samples), batch_size=16, shuffle=True,
                    collate_fn=svc_collate, num_workers=0)
    tracker = LossTracker()
    trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=1, precision="32-true",
                        logger=False, enable_progress_bar=False, callbacks=[tracker],
                        num_sanity_val_steps=0)
    trainer.fit(module, dl)
    return tracker.losses


def train_orig_rift(samples, seed=42):
    import sys; sys.path.insert(0, "/root/RIFT-SVC")
    from rift_svc.rf import RF
    from rift_svc.dit import DiT

    torch.manual_seed(seed)
    dit = DiT(dim=256, depth=4, head_dim=64, dropout=0.0, ff_mult=4,
              n_mel_channels=128, num_speaker=1, cvec_dim=768, kernel_size=3, zero_null_spk=True)
    rf = RF(dit, time_schedule="lognorm")
    rf.mel_min, rf.mel_max = -12.0, 2.0
    rf = rf.cuda()
    opt = torch.optim.AdamW(rf.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)

    g = torch.Generator().manual_seed(seed)
    losses = []
    rf.train()
    # One epoch
    order = torch.randperm(len(samples), generator=g).tolist()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        mel = torch.stack([samples[i]["mel"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        rms = torch.stack([samples[i]["rms"] for i in idx]).cuda()
        cvec = torch.stack([samples[i]["units"] for i in idx]).cuda()
        fl = torch.full((len(idx),), mel.shape[1], device="cuda")
        sid = torch.zeros(len(idx), dtype=torch.long, device="cuda")
        loss, _ = rf(mel, spk_id=sid, f0=f0, rms=rms, cvec=cvec, frame_len=fl, drop_speaker=False)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(rf.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    return losses


def analyze(name, usvc_losses, orig_losses):
    """Compare loss trends statistically."""
    print(f"\n{'='*60}")
    print(f"{name}: 1-epoch training loss trend comparison")
    print(f"{'='*60}")
    print(f"  usvc: {len(usvc_losses)} steps, mean={np.mean(usvc_losses):.4f}, "
          f"first={np.mean(usvc_losses[:10]):.4f}, last={np.mean(usvc_losses[-10:]):.4f}")
    print(f"  orig: {len(orig_losses)} steps, mean={np.mean(orig_losses):.4f}, "
          f"first={np.mean(orig_losses[:10]):.4f}, last={np.mean(orig_losses[-10:]):.4f}")

    u_ma = moving_avg(usvc_losses)
    o_ma = moving_avg(orig_losses)

    # Trend comparison: both should decrease, similar rate
    u_drop = u_ma[0] - u_ma[-1] if len(u_ma) > 1 else 0
    o_drop = o_ma[0] - o_ma[-1] if len(o_ma) > 1 else 0
    print(f"\n  Loss drop (MA start→end):")
    print(f"    usvc: {u_ma[0]:.4f} → {u_ma[-1]:.4f} (drop {u_drop:.4f})")
    print(f"    orig: {o_ma[0]:.4f} → {o_ma[-1]:.4f} (drop {o_drop:.4f})")

    # Relative difference of means (the key metric)
    mean_diff = abs(np.mean(usvc_losses) - np.mean(orig_losses)) / np.mean(orig_losses)
    drop_ratio = min(u_drop, o_drop) / max(u_drop, o_drop) if max(u_drop, o_drop) > 0 else 1.0

    print(f"\n  Mean relative diff: {mean_diff*100:.1f}%")
    print(f"  Drop ratio (smaller/larger): {drop_ratio:.2f}")

    # Verdict
    if mean_diff < 0.15 and drop_ratio > 0.7:
        verdict = "✓ LOSS TREND MATCHES (趋势接近,量级合理)"
    elif mean_diff < 0.30:
        verdict = "△ LOSS TREND CLOSE (趋势大致接近)"
    else:
        verdict = "✗ LOSS TREND DIVERGES (趋势偏离)"
    print(f"  → {verdict}")
    return {"mean_rel_diff": mean_diff, "drop_ratio": drop_ratio, "verdict": verdict,
            "usvc_ma": u_ma, "orig_ma": o_ma}


def main():
    samples = make_fixed_dataset(n=200, T=64, seed=777)

    print("Training usvc rift_dit (1 epoch, 200 samples)...")
    usvc = train_usvc_rift(samples, seed=42)
    print("Training original RIFT-SVC (1 epoch, 200 samples)...")
    orig = train_orig_rift(samples, seed=42)

    result = analyze("RIFT-DiT", usvc, orig)

    out = Path("/tmp/usvc_validate/training_trend.json")
    with open(out, "w") as f:
        json.dump({"usvc": usvc, "orig": orig, "analysis": {k: v for k, v in result.items() if k != "verdict"}}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
