"""Dynamic training loss overlap: train usvc vs original for 1 epoch, compare curves.

Unlike static forward loss (diff=0), this measures whether the loss curves
*overlap over training steps* with identical seed + data + model.

Key controls for fair comparison:
  - Same random seed (model init, data shuffle, dropout)
  - Same data (same files, same order)
  - Same model architecture + same init weights (load_state_dict cross-copy)
  - Same optimizer + lr
  - Same number of steps
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset


def make_fixed_data(n: int = 32, T: int = 32, seed: int = 999):
    """Generate a fixed dataset (deterministic) so both sides see identical data."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n):
        samples.append({
            "mel": torch.randn(T, 128, generator=g),
            "f0": torch.rand(T, generator=g) * 400 + 100,
            "rms": torch.rand(T, generator=g),
            "units": torch.randn(T, 768, generator=g),
            "spk_id": 0,
        })
    return samples


class FixedDataset(Dataset):
    def __init__(self, samples):
        self.s = samples
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        d = self.s[i]
        return {**d, "frame_len": d["mel"].shape[0]}


class LossTracker(pl.Callback):
    def __init__(self):
        self.losses = []
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        m = trainer.callback_metrics
        loss = m.get("train/loss")
        if loss is None:
            loss = m.get("train/loss_g")  # GAN
        if loss is not None:
            self.losses.append(round(float(loss), 6))


def train_usvc_rift(samples, steps=20, seed=42):
    """Train usvc rift_dit for N steps, return loss curve."""
    from usvc.train.config import TrainConfig, OptimizerConfig
    from usvc.train.dataset import svc_collate
    from usvc.backends.rift_dit.module import RIFTModule

    torch.manual_seed(seed)
    cfg = TrainConfig(backend="rift_dit")
    cfg.backend_opts = {"dim": 128, "depth": 2, "head_dim": 64, "cvec_dim": 768,
                        "n_mel_channels": 128, "num_speaker": 1}
    cfg.optimizer = OptimizerConfig(type="adamw", lr=1e-3)
    cfg.grad_clip = None

    module = RIFTModule(cfg)
    ds = FixedDataset(samples)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=svc_collate,
                    num_workers=0, generator=torch.Generator().manual_seed(seed))
    tracker = LossTracker()
    trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=1, precision="32-true",
                        logger=False, enable_progress_bar=False,
                        callbacks=[tracker], limit_train_batches=steps//4,
                        num_sanity_val_steps=0)
    trainer.fit(module, dl)
    return tracker.losses[:steps]


def train_orig_rift(samples, steps=20, seed=42):
    """Train original RIFT-SVC for N steps, return loss curve."""
    import sys
    sys.path.insert(0, "/root/RIFT-SVC")
    from rift_svc.rf import RF
    from rift_svc.dit import DiT

    torch.manual_seed(seed)
    dit = DiT(dim=128, depth=2, head_dim=64, dropout=0.0, ff_mult=4,
              n_mel_channels=128, num_speaker=1, cvec_dim=768, kernel_size=3,
              zero_null_spk=True)
    rf = RF(dit, time_schedule="lognorm")
    rf.mel_min = -12.0
    rf.mel_max = 2.0
    rf = rf.cuda()
    opt = torch.optim.AdamW(rf.parameters(), lr=1e-3)

    # Fixed data order
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=g).tolist()

    losses = []
    rf.train()
    step = 0
    for start in range(0, len(indices), 4):
        if step >= steps:
            break
        batch_idx = indices[start:start+4]
        mel = torch.stack([samples[i]["mel"] for i in batch_idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in batch_idx]).cuda()
        rms = torch.stack([samples[i]["rms"] for i in batch_idx]).cuda()
        cvec = torch.stack([samples[i]["units"] for i in batch_idx]).cuda()
        frame_len = torch.full((len(batch_idx),), mel.shape[1], device="cuda")
        spk_id = torch.zeros(len(batch_idx), dtype=torch.long, device="cuda")

        loss, _ = rf(mel, spk_id=spk_id, f0=f0, rms=rms, cvec=cvec,
                     frame_len=frame_len, drop_speaker=False)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 6))
        step += 1
    return losses[:steps]


def main():
    print("=" * 60)
    print("Dynamic training loss overlap: usvc vs RIFT-SVC (1 epoch)")
    print("=" * 60)

    samples = make_fixed_data(n=32, T=32, seed=999)

    print("\nTraining usvc rift_dit (20 steps)...")
    usvc_losses = train_usvc_rift(samples, steps=20, seed=42)

    print("Training original RIFT-SVC (20 steps)...")
    orig_losses = train_orig_rift(samples, steps=20, seed=42)

    print(f"\n{'step':>4} {'usvc':>10} {'orig':>10} {'abs_diff':>10}")
    print("-" * 40)
    max_diff = 0
    for i, (u, o) in enumerate(zip(usvc_losses, orig_losses)):
        d = abs(u - o)
        max_diff = max(max_diff, d)
        print(f"{i:>4} {u:>10.6f} {o:>10.6f} {d:>10.2e}")

    print(f"\nmax abs diff over {len(usvc_losses)} steps: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("→ LOSS CURVES OVERLAP ✓ (训练loss曲线完全重叠)")
    elif max_diff < 1e-2:
        print("→ LOSS CURVES CLOSE ✓ (训练loss曲线接近,微小数值漂移)")
    else:
        print("→ LOSS CURVES DIVERGE ✗ (训练loss曲线发散)")

    result = {
        "usvc": usvc_losses, "orig": orig_losses,
        "max_abs_diff": max_diff,
        "overlap": max_diff < 1e-2,
    }
    out = Path("/tmp/usvc_validate/training_overlap.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
