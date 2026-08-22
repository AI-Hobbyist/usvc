"""Training CLI entry point.

Usage:
    usvc train --config configs/rift_dit_train.yaml
    usvc train --config ... --resume experiments/rift_v1/last.ckpt
"""
from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from .config import TrainConfig


class _PruneCheckpoints(Callback):
    """Delete old epoch=*.ckpt files, keeping only the N most recent.

    Lightning's save_top_k requires a monitor metric (which we lack when
    validation is skipped). This callback manually prunes step checkpoints
    after each save, keeping disk bounded. last.ckpt is never deleted.
    """

    def __init__(self, dirpath, keep_last_n=3):
        self.dirpath = str(dirpath)
        self.keep_last_n = keep_last_n

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.keep_last_n < 0:
            return  # -1 = keep all
        import glob, os
        # Only prune right after a checkpoint was saved.
        ckpts = sorted(
            glob.glob(os.path.join(self.dirpath, "epoch=*.ckpt")),
            key=os.path.getmtime,
        )
        if len(ckpts) > self.keep_last_n:
            for old in ckpts[: len(ckpts) - self.keep_last_n]:
                try:
                    os.remove(old)
                except OSError:
                    pass
from .dataset import svc_collate
from .callbacks import ScheduleFreeModeCallback
from ._resolvers import resolve_feature_hashes, resolve_val_interval

__all__ = ["run_training"]


def run_training(cfg: TrainConfig, resume: str | None = None, device: str | None = None):
    """Run training with the given config."""
    # Enable TF32 on Ampere+ for near-lossless matmul/cudnn speedup.
    # (PyTorch defaults these to False for historical reasons.)
    import torch as _t
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True
    # cudnn.benchmark: 让 cuDNN 为每个 conv 形状 benchmark 选最优算法（含 TensorCore dgrad）。
    # 实测 rvc dgrad 290→111ms (62%↓)，总步 712→626ms (1.14×)；所有 conv 后端普遍受益。
    # 代价：首次 forward 慢（benchmark 各算法），但训练形状固定，后续都走最优算法。
    _t.backends.cudnn.benchmark = True

    from ..backends import get_backend

    backend = get_backend(cfg.backend)()

    # Resolve feature hashes from dataset meta (units nested per encoder kind)
    # — see _resolvers.resolve_feature_hashes for the match semantics.
    from ..data.meta import load_meta

    meta = load_meta(cfg.dataset_dir)
    spec_cfg = dict(cfg.backend_opts or {})
    want_spec = backend.feature_spec(spec_cfg)
    units_kind = (
        want_spec.units.params.get("kind", "contentvec")
        if want_spec.units else None
    )
    cfg.feature_hashes = resolve_feature_hashes(meta, want_spec, units_kind)

    # Build datasets
    train_ds = backend.build_dataset(cfg.dataset_dir, "train", cfg.feature_hashes)
    train_ds.max_frames = cfg.max_frames  # override backend default (790) with config
    # Keep dataset encoder_kind in sync with config overrides (e.g. sovits
    # units_kind: contentvec) — build_dataset uses feature_spec({}) which has
    # no access to the train config.
    spec_cfg = dict(cfg.backend_opts or {})
    want_units_kind = (
        backend.feature_spec(spec_cfg).units.params.get("kind") if backend.feature_spec(spec_cfg).units else None
    )
    if want_units_kind and hasattr(train_ds, "encoder_kind") and train_ds.encoder_kind != want_units_kind:
        train_ds.encoder_kind = want_units_kind
        train_ds._units_dir = train_ds.dataset_dir / "units" / want_units_kind / cfg.feature_hashes.get("units", "")
    val_ds = backend.build_dataset(
        cfg.val_dir or cfg.dataset_dir, "val", cfg.feature_hashes
    )
    val_ds.max_frames = cfg.max_frames
    if want_units_kind and hasattr(val_ds, "encoder_kind") and val_ds.encoder_kind != want_units_kind:
        val_ds.encoder_kind = want_units_kind
        val_ds._units_dir = val_ds.dataset_dir / "units" / want_units_kind / cfg.feature_hashes.get("units", "")

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
        from pathlib import Path

        # Collect all pretrained state_dicts.  so-vits-svc base models ship
        # G_0.pth (generator) and D_0.pth (discriminator) separately; RVC
        # ships f0G48k.pth / f0D48k.pth.  If the user points at the
        # generator file we also try to load the sibling discriminator file.
        pretrained_paths = [Path(cfg.pretrained)]
        p = Path(cfg.pretrained)
        # Auto-download base model if missing
        if not p.exists():
            from ..hub import ensure_pretrained
            downloaded = ensure_pretrained(cfg.pretrained)
            p = Path(downloaded)
            pretrained_paths = [p]
        if p.name == "G_0.pth":
            d0 = p.with_name("D_0.pth")
            if d0.exists():
                pretrained_paths.append(d0)
        elif "G" in p.stem and "D" not in p.stem:
            d0 = p.with_name(p.name.replace("G", "D", 1))
            if d0.exists():
                pretrained_paths.append(d0)

        merged_sd: dict = {}
        for pp in pretrained_paths:
            _c = torch.load(pp, map_location="cpu", weights_only=True)
            merged_sd.update(_c.get("state_dict", _c.get("model", _c)))
        sd = merged_sd
        module_sd = module.state_dict()

        def _safe_load(remapped_sd):
            """load_state_dict that skips shape-mismatched keys (e.g. emb_g
            when finetuning a multi-speaker base down to 1 speaker)."""
            filtered = {
                k: v for k, v in remapped_sd.items()
                if k in module_sd and v.shape == module_sd[k].shape
            }
            missing, unexpected = module.load_state_dict(filtered, strict=False)
            matched = len(module_sd) - len(missing)
            return matched, missing, unexpected

        # Remap keys: base models (sovits/rvc) store raw G/D weights without
        # the ``generator.``/``discriminator.`` wrapper prefix.  Match each key
        # individually against the module state_dict, trying (no prefix,
        # generator., discriminator., model.) in order.  This lets a single
        # checkpoint that mixes G and D keys (e.g. merged G_0+D_0) load both
        # subtrees in one pass.
        module_keys = set(module_sd.keys())
        remapped = {}
        for k, v in sd.items():
            if k in module_keys:
                remapped[k] = v
                continue
            for prefix in ("generator.", "discriminator.", "model."):
                cand = prefix + k
                if cand in module_keys:
                    remapped[cand] = v
                    break
            else:
                remapped[k] = v  # leave as-is (will be "unexpected")
        matched, missing, unexpected = _safe_load(remapped)
        if missing:
            print(f"[pretrained] missing keys: {len(missing)}")
        if unexpected:
            print(f"[pretrained] unexpected keys: {len(unexpected)}")
        print(f"[pretrained] matched: {matched}/{len(module_sd)}")

    # Trainer
    # Checkpoint strategy: keep complete checkpoints (weights + optimizer)
    # for perfect resume from ANY checkpoint. Disk bounded via keep_last_n.
    # save_weights_only=False preserves Adam momentum/variance so resume
    # continues training exactly as if never interrupted.
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.exp_dir,
        every_n_train_steps=cfg.save_interval,
        save_last=True,
        save_top_k=-1,  # keep all; cleanup via _PruneCheckpoints
        save_weights_only=False,  # full state: weights + optimizer (perfect resume)
    )
    # Prune old step checkpoints to keep disk bounded (save_top_k needs a
    # monitor metric which we don't have without validation).
    keep_n = int(getattr(cfg, "keep_last_n", 5))
    prune_cb = _PruneCheckpoints(
        dirpath=cfg.exp_dir,
        keep_last_n=keep_n,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # gradient_clip_val only for automatic optimization; GAN backends use
    # manual_optimization and clip internally (gan_module.py).
    from ..train.gan_module import BaseGANModule
    use_grad_clip = cfg.grad_clip if not isinstance(module, BaseGANModule) else None

    # Resolve device: 'cuda' / 'cuda:N' -> gpu + specific device; else auto.
    if device and ":" in device:
        accelerator = "gpu"
        devices = [int(device.split(":", 1)[1])]
    elif device == "cuda":
        accelerator, devices = "gpu", "auto"
    elif device == "cpu":
        accelerator, devices = "cpu", "auto"
    else:
        accelerator, devices = "auto", "auto"

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        accelerator=accelerator,
        devices=devices,
        precision=cfg.precision,
        gradient_clip_val=use_grad_clip,
        callbacks=[checkpoint_cb, prune_cb, lr_monitor, ScheduleFreeModeCallback()],
        # Explicit logger so metrics/checkpoints land under exp_dir predictably
        # (Lightning's default TensorBoardLogger sometimes defers dir creation).
        logger=pl.loggers.TensorBoardLogger(save_dir=str(cfg.exp_dir), name="lightning_logs"),
        # val_check_interval: >1 = step count (if <= batches/epoch),
        # <=1 = epoch fraction. Clamp to per-epoch (1.0) when the requested
        # step interval exceeds the batches-per-epoch, to avoid
        # "interval > num_batches" error on small datasets.
        val_check_interval=resolve_val_interval(cfg.val_interval, len(train_dl)),
        # Step-based training (max_steps, no max_epochs) MUST use
        # check_val_every_n_epoch=None so val_check_interval (an int step count
        # possibly larger than batches/epoch) is honored across epochs. Epoch-
        # based training (max_epochs set) validates every epoch. (Lightning 2.x
        # raises MisconfigurationException if both an int val_check_interval and
        # a non-None check_val_every_n_epoch are given.)
        check_val_every_n_epoch=1 if cfg.max_epochs else None,
        num_sanity_val_steps=0,
        limit_val_batches=cfg.limit_val_batches if cfg.limit_val_batches is not None else 1.0,
        log_every_n_steps=10,
    )

    trainer.fit(
        module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=resume or cfg.resume,
    )
