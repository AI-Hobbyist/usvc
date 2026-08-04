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


def _resolve_val_interval(val_interval: float | int, n_batches: int) -> float:
    """Map cfg.val_interval to Lightning's val_check_interval.

    >1 = step count; <=1 = epoch fraction. Values larger than the number of
    batches per epoch are clamped to 1.0 (validate at the end of every epoch)
    so small datasets don't crash with "interval > num_batches".
    """
    if val_interval <= 1:
        return max(float(val_interval), 1e-6)
    if n_batches > 0 and val_interval > n_batches:
        return 1.0
    return float(val_interval)


def run_training(cfg: TrainConfig, resume: str | None = None):
    """Run training with the given config."""
    # Enable TF32 on Ampere+ for near-lossless matmul/cudnn speedup.
    # (PyTorch defaults these to False for historical reasons.)
    import torch as _t
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True

    from ..backends import get_backend

    backend = get_backend(cfg.backend)()

    # Resolve feature hashes from dataset meta.
    # meta.features structure: {"mel":[{hash,...}], "f0":[...], "units":{enc:[{hash,...}]}}
    # Build flat {kind: hash} for dataset construction.
    from ..data.meta import load_meta

    meta = load_meta(cfg.dataset_dir)
    # Determine which units encoder this backend needs (contentvec_tta2x /
    # contentvec / hubertsoft) so the correct nested hash is selected.
    # feature_spec reads config overrides (units_kind, f0_extractor, mel bins).
    spec_cfg = dict(cfg.backend_opts or {})
    units_kind = (
        backend.feature_spec(spec_cfg).units.params.get("kind", "contentvec")
        if backend.feature_spec(spec_cfg).units
        else None
    )
    feature_hashes: dict[str, str] = {}
    for kind, entries in meta.features.items():
        if kind == "units":
            # nested: {encoder_kind: [{hash,...}]}
            if units_kind is not None and units_kind in (entries or {}):
                enc_entries = entries[units_kind]
                if enc_entries:
                    feature_hashes["units"] = enc_entries[0]["hash"]
                    continue
            for enc_kind, enc_entries in (entries or {}).items():
                if enc_entries:
                    feature_hashes["units"] = enc_entries[0]["hash"]
                    break
        elif isinstance(entries, list) and entries:
            feature_hashes[kind] = entries[0]["hash"]
    cfg.feature_hashes = feature_hashes

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

        ckpt = torch.load(cfg.pretrained, map_location="cpu", weights_only=True)
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # Try direct load first; if too few keys match, attempt prefix remapping
        # (e.g. base model has 'enc_p.*' but module expects 'generator.enc_p.*')
        missing, unexpected = module.load_state_dict(sd, strict=False)
        matched = len(module.state_dict()) - len(missing)
        if matched < len(module.state_dict()) * 0.5:
            # Heuristic: GAN modules wrap generator/discriminator; base models
            # (sovits/rvc) often store raw generator weights without the prefix.
            module_keys = set(module.state_dict().keys())
            # Detect candidate prefix: longest common suffix alignment
            for prefix in ("generator.", "discriminator."):
                remapped = {}
                for k, v in sd.items():
                    candidate = prefix + k
                    if candidate in module_keys:
                        remapped[candidate] = v
                    else:
                        remapped[k] = v
                new_missing, new_unexpected = module.load_state_dict(
                    remapped, strict=False
                )
                new_matched = len(module.state_dict()) - len(new_missing)
                if new_matched > matched:
                    sd, missing, unexpected = remapped, new_missing, new_unexpected
                    matched = new_matched
        if missing:
            print(f"[pretrained] missing keys: {len(missing)}")
        if unexpected:
            print(f"[pretrained] unexpected keys: {len(unexpected)}")
        print(f"[pretrained] matched: {matched}/{len(module.state_dict())}")

    # Trainer
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.exp_dir,
        every_n_train_steps=cfg.save_interval,
        save_last=True,
        save_top_k=-1,
        save_weights_only=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # gradient_clip_val only for automatic optimization; GAN backends use
    # manual_optimization and clip internally (gan_module.py).
    from ..train.gan_module import BaseGANModule
    use_grad_clip = cfg.grad_clip if not isinstance(module, BaseGANModule) else None

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        accelerator="auto",
        devices="auto",
        precision=cfg.precision,
        gradient_clip_val=use_grad_clip,
        callbacks=[checkpoint_cb, lr_monitor, ScheduleFreeModeCallback()],
        # val_check_interval: >1 = step count (if <= batches/epoch),
        # <=1 = epoch fraction. Clamp to per-epoch (1.0) when the requested
        # step interval exceeds the batches-per-epoch, to avoid
        # "interval > num_batches" error on small datasets.
        val_check_interval=_resolve_val_interval(cfg.val_interval, len(train_dl)),
        check_val_every_n_epoch=None if cfg.max_epochs else 1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )

    trainer.fit(
        module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=resume or cfg.resume,
    )
