"""Training CLI entry point.

Usage:
    usvc train --config configs/rift_dit_train.yaml
    usvc train --config ... --resume experiments/rift_v1/last.ckpt
"""
from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import svc_collate
from .callbacks import ScheduleFreeModeCallback

__all__ = ["run_training"]


def run_training(cfg: TrainConfig, resume: str | None = None):
    """Run training with the given config."""
    # Enable TF32 on Ampere+ for near-lossless matmul/cudnn speedup.
    # (PyTorch defaults these to False for historical reasons.)
    import torch as _t
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True

    from ..backends import get_backend

    backend = get_backend(cfg.backend)()

    # Resolve feature hashes from dataset meta
    from ..data.meta import load_meta

    meta = load_meta(cfg.dataset_dir)
    cfg.feature_hashes = {fc.kind: h for h, fc in meta.iter_feature_configs()}

    # Build datasets
    train_ds = backend.build_dataset(cfg.dataset_dir, "train", cfg.feature_hashes)
    val_ds = backend.build_dataset(
        cfg.val_dir or cfg.dataset_dir, "val", cfg.feature_hashes
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=svc_collate,
        drop_last=True,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch_size // 4),
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=svc_collate,
        pin_memory=True,
    )

    # Build module
    module = backend.build_module(cfg)

    # Load pretrained weights (if specified, before resume)
    if cfg.pretrained:
        import torch

        ckpt = torch.load(cfg.pretrained, map_location="cpu", weights_only=True)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        if missing:
            print(f"[pretrained] missing keys: {len(missing)}")
        if unexpected:
            print(f"[pretrained] unexpected keys: {len(unexpected)}")

    # Trainer
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.exp_dir,
        every_n_train_steps=cfg.save_interval,
        save_last=True,
        save_top_k=-1,
        save_weights_only=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        accelerator="auto",
        devices="auto",
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip if cfg.grad_clip else None,
        callbacks=[checkpoint_cb, lr_monitor, ScheduleFreeModeCallback()],
        val_check_interval=cfg.val_interval,
        check_val_every_n_epoch=None if cfg.max_epochs else 1,
        log_every_n_steps=50,
    )

    trainer.fit(
        module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=resume or cfg.resume,
    )
