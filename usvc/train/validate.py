"""Training validation: run each backend for 2 epochs, record loss curves.

This validates that all backends train correctly and produces loss curves
for overlap comparison with original projects.

Usage:
    python -m usvc.train.validate --dataset /tmp/u_opt --epochs 2
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ..backends import get_backend
from .config import OptimizerConfig, TrainConfig
from .dataset import svc_collate
from .callbacks import ScheduleFreeModeCallback

# Feature hashes for /tmp/u_opt test dataset
TEST_HASHES = {
    "mel": "512c8684abd5bce4",
    "f0": "e2396fffbd0bbea4",
    "rms": "f5352cab2a3e87a0",
    "units": "7b5f455ae0b6450d",
}


class LossLogger(pl.Callback):
    """Record per-step loss into a list."""

    def __init__(self):
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        # Get the loss from the logged metrics
        if trainer.callback_metrics:
            loss = trainer.callback_metrics.get("train/loss")
            if loss is None:
                # GAN backends log loss_g/loss_d
                loss = trainer.callback_metrics.get("train/loss_g")
            if loss is not None:
                self.losses.append(float(loss))


def run_backend(
    backend_name: str,
    dataset_dir: str,
    epochs: int = 2,
    max_frames: int = 64,
    batch_size: int = 4,
    use_synthetic: bool = False,
    encoder_kind: str = "contentvec",
) -> dict:
    """Run one backend for N epochs, return loss curve + metadata."""
    print(f"\n{'='*60}")
    print(f"Backend: {backend_name}")
    print(f"{'='*60}")

    backend_cls = get_backend(backend_name)
    backend = backend_cls()

    exp_dir = Path(f"/tmp/usvc_validate/{backend_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        backend=backend_name,
        dataset_dir=dataset_dir,
        exp_dir=str(exp_dir),
        batch_size=batch_size,
        max_frames=max_frames,
        max_epochs=epochs,
        max_steps=-1,
        num_workers=0,
        precision="32-true",
        grad_clip=None,
    )
    cfg.optimizer = OptimizerConfig(type="adamw", lr=1e-4)

    # Backend-specific small model config for fast validation
    if backend_name == "rift_dit":
        cfg.backend_opts = {"dim": 128, "depth": 2, "head_dim": 64, "cvec_dim": 768,
                           "n_mel_channels": 128, "num_speaker": 1}
    elif backend_name == "ddsp6":
        cfg.backend_opts = {"n_units": 768, "num_speaker": 1, "n_aux_layers": 3,
                           "n_aux_chans": 128, "n_layers": 2, "n_chans": 256}
    elif backend_name == "diffusion":
        cfg.backend_opts = {"n_units": 768, "num_speaker": 1, "n_hidden": 256}
    elif backend_name == "sovits":
        use_synthetic = True  # needs spec which test data lacks
        cfg.backend_opts = {"n_units": 256, "spec_channels": 1025, "inter_channels": 192,
                           "hidden_channels": 192, "n_heads": 2, "segment_size": 8192,
                           "num_speaker": 1, "hop": 512}
    elif backend_name == "rvc":
        use_synthetic = True
        cfg.backend_opts = {"n_units": 768, "inter_channels": 192, "hidden_channels": 192,
                           "filter_channels": 768, "num_speaker": 1, "segment_size": 20,
                           "sr": 48000, "hop": 512}

    module = backend.build_module(cfg)

    # Build dataset
    if use_synthetic:
        dl = _make_synthetic_loader(backend_name, batch_size)
    else:
        # Force encoder_kind to match test data (contentvec, not tta2x)
        ds = backend.build_dataset(dataset_dir, "train", TEST_HASHES)
        if hasattr(ds, 'encoder_kind'):
            ds.encoder_kind = 'contentvec'
            ds._units_dir = Path(dataset_dir) / 'units' / 'contentvec' / TEST_HASHES['units']
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                       collate_fn=svc_collate, num_workers=0)

    loss_logger = LossLogger()
    ckpt_cb = ModelCheckpoint(dirpath=str(exp_dir), save_last=True, save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="32-true",
        callbacks=[ckpt_cb, loss_logger, ScheduleFreeModeCallback()],
        logger=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_train_batches=20,  # cap for speed
    )

    t0 = time.perf_counter()
    trainer.fit(module, dl)
    elapsed = time.perf_counter() - t0

    result = {
        "backend": backend_name,
        "epochs": epochs,
        "global_step": trainer.global_step,
        "elapsed_s": round(elapsed, 1),
        "losses": loss_logger.losses,
        "final_loss": loss_logger.losses[-1] if loss_logger.losses else None,
        "ckpt_exists": ckpt_cb.last_model_path and Path(ckpt_cb.last_model_path).exists(),
    }
    return result


def _make_synthetic_loader(backend_name: str, batch_size: int):
    """Make synthetic data loader for GAN backends (need spec/wav)."""
    from torch.utils.data import Dataset

    class SyntheticDataset(Dataset):
        def __len__(self): return 40
        def __getitem__(self, i):
            T, L = 32, 16384
            n_units = 256 if backend_name == "sovits" else 768
            item = {
                "units": torch.randn(T, n_units),
                "f0": torch.rand(T)*400+100,
                "spk_id": 0,
            }
            if backend_name == "sovits":
                item["mel"] = torch.randn(1025, T)
                item["wav"] = torch.randn(L)
                item["uv"] = (torch.rand(T) > 0.5).float()
            elif backend_name == "rvc":
                item["mel"] = torch.randn(1025, T)
                item["wav"] = torch.randn(L)
                item["pitch_token"] = torch.randint(1, 256, (T,))
            return item

    return DataLoader(SyntheticDataset(), batch_size=batch_size,
                     shuffle=True, collate_fn=svc_collate, num_workers=0)


def main(dataset_dir: str = "/tmp/u_opt", epochs: int = 2):
    """Run all backends and save results."""
    results = {}
    for backend in ["rift_dit", "ddsp6", "diffusion", "sovits", "rvc"]:
        try:
            results[backend] = run_backend(backend, dataset_dir, epochs)
            print(f"✅ {backend}: step={results[backend]['global_step']}, "
                  f"final_loss={results[backend]['final_loss']:.4f}, "
                  f"{results[backend]['elapsed_s']}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[backend] = {"backend": backend, "error": str(e)}

    # Save results
    out_path = Path("/tmp/usvc_validate/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    dataset = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else "/tmp/u_opt"
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 2
    main(dataset, epochs)
