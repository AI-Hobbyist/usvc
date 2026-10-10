"""Training CLI entry point: usvc train --config configs/*.yaml [--resume] [--device]."""
import glob
import os
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from ._resolvers import resolve_feature_hashes, resolve_val_interval
from .callbacks import ScheduleFreeModeCallback
from .config import TrainConfig
from .dataset import svc_collate


class _PruneCheckpoints(Callback):
    # keep the N most recent epoch=*.ckpt (save_top_k needs a monitor metric we lack without validation); last.ckpt never deleted
    def __init__(self, dirpath, keep_last_n=3):
        self.dirpath, self.keep_last_n = str(dirpath), keep_last_n

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.keep_last_n < 0:
            return  # -1 = keep all
        ckpts = sorted(glob.glob(os.path.join(self.dirpath, "epoch=*.ckpt")), key=os.path.getmtime)
        for old in ckpts[:max(0, len(ckpts) - self.keep_last_n)]:
            try:
                os.remove(old)
            except OSError:
                pass


def run_training(cfg: TrainConfig, resume=None, device=None):
    import torch as _t
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True
    # cudnn.benchmark: 让 cuDNN 为每个 conv 形状 benchmark 选最优算法（含 TensorCore dgrad）。
    # 实测 rvc dgrad 290→111ms (62%↓)，总步 712→626ms；代价是首次 forward 慢，形状固定后都走最优算法
    _t.backends.cudnn.benchmark = True

    from ..backends import get_backend
    from ..data.meta import load_meta
    backend = get_backend(cfg.backend)()

    # Resolve feature hashes from dataset meta (units nested per encoder kind) — see _resolvers for match semantics
    meta = load_meta(cfg.dataset_dir)
    from ..speakers import TARGET_BACKENDS, resolve_training_speakers, bind_dataset_speakers
    if cfg.backend in TARGET_BACKENDS:
        if resume or cfg.resume:
            from .speaker_checkpoint import read_speaker_metadata
            saved = _t.load(resume or cfg.resume, map_location="cpu", weights_only=False)
            metadata = read_speaker_metadata(saved)
            if metadata is None:
                raise ValueError("full resume requires trusted checkpoint speaker metadata")
            if cfg.spk is None:
                cfg.spk = metadata["spk"]
        resolve_training_speakers(cfg, meta)
    want_spec = backend.feature_spec(dict(cfg.backend_opts or {}))
    units_kind = want_spec.units.params.get("kind", "contentvec") if want_spec.units else None
    cfg.feature_hashes = resolve_feature_hashes(meta, want_spec, units_kind)

    def _ds(dir, split):
        ds = backend.build_dataset(dir, split, cfg.feature_hashes)
        ds.max_frames = cfg.max_frames  # override backend default with config
        if cfg.backend in TARGET_BACKENDS:
            bind_dataset_speakers(ds, cfg.spk, training=split == "train")
        # keep dataset encoder_kind in sync with config overrides (build_dataset sees feature_spec({}), no train config)
        want_units_kind = want_spec.units.params.get("kind") if want_spec.units else None
        if want_units_kind and hasattr(ds, "encoder_kind") and ds.encoder_kind != want_units_kind:
            ds.encoder_kind = want_units_kind
            ds._units_dir = ds.dataset_dir / "units" / want_units_kind / cfg.feature_hashes.get("units", "")
        return ds

    train_ds, val_ds = _ds(cfg.dataset_dir, "train"), _ds(cfg.val_dir or cfg.dataset_dir, "val")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                          persistent_workers=cfg.num_workers > 0, collate_fn=svc_collate, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=max(1, cfg.batch_size // 4), shuffle=False, num_workers=cfg.num_workers,
                        persistent_workers=cfg.num_workers > 0, collate_fn=svc_collate, pin_memory=True)

    module = backend.build_module(cfg)
    if cfg.backend in TARGET_BACKENDS and (resume or cfg.resume):
        module.on_load_checkpoint(saved)
    if cfg.backend in TARGET_BACKENDS:
        from .speaker_checkpoint import config_payload
        import yaml
        Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.exp_dir) / "config.yaml").write_text(
            yaml.safe_dump(config_payload(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ---- pretrained: so-vits base ships G_0.pth/D_0.pth separately, RVC f0G48k.pth/f0D48k.pth; when pointed at
    # the generator file we also try the sibling discriminator; keys get (no prefix, generator., discriminator.,
    # model.) remap; shape-mismatched keys are skipped (e.g. emb_g finetuning multi-speaker base down to 1) ----
    if cfg.pretrained:
        p = Path(cfg.pretrained)
        paths = [p]
        if not p.exists():
            from ..hub import ensure_pretrained
            p = Path(ensure_pretrained(cfg.pretrained))
            paths = [p]
        if p.name == "G_0.pth":
            d0 = p.with_name("D_0.pth")
            if d0.exists(): paths.append(d0)
        elif "G" in p.stem and "D" not in p.stem:
            d0 = p.with_name(p.name.replace("G", "D", 1))
            if d0.exists(): paths.append(d0)
        sd = {}
        for pp in paths:
            _c = _t.load(pp, map_location="cpu", weights_only=True)
            sd.update(_c.get("state_dict", _c.get("model", _c)))
        msd = module.state_dict()
        keys = set(msd)
        remapped = {}
        for k, v in sd.items():
            for prefix in ("", "generator.", "discriminator.", "model."):
                if prefix + k in keys:
                    remapped[prefix + k] = v
                    break
            else:
                remapped[k] = v  # leave as-is (will be "unexpected")
        filtered = {k: v for k, v in remapped.items() if k in keys and v.shape == msd[k].shape}
        for k, v in remapped.items():
            if cfg.backend in TARGET_BACKENDS and k in keys and v.shape != msd[k].shape and ("spk_embed" in k or "emb_g" in k):
                print(f"[pretrained] reinitializing speaker embedding {k}: {tuple(v.shape)} -> {tuple(msd[k].shape)}")
        missing, unexpected = module.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[pretrained] missing keys: {len(missing)}")
        if unexpected:
            print(f"[pretrained] unexpected keys: {len(unexpected)}")
        print(f"[pretrained] matched: {len(msd) - len(missing)}/{len(msd)}")

    # ---- trainer: checkpoint strategy = complete checkpoints (weights + optimizer) for perfect resume,
    # disk bounded via _PruneCheckpoints; gradient_clip_val only for automatic optimization (GAN backends
    # use manual_optimization and clip internally in gan_module.py) ----
    from ..train.gan_module import BaseGANModule
    use_grad_clip = cfg.grad_clip if not isinstance(module, BaseGANModule) else None
    checkpoint_cb = ModelCheckpoint(dirpath=cfg.exp_dir, every_n_train_steps=cfg.save_interval, save_last=True,
                                    save_top_k=-1,  # keep all; cleanup via _PruneCheckpoints
                                    save_weights_only=False)  # full state: weights + optimizer (perfect resume)
    prune_cb = _PruneCheckpoints(dirpath=cfg.exp_dir, keep_last_n=int(getattr(cfg, "keep_last_n", 5)))
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if device and ":" in device:
        accelerator, devices = "gpu", [int(device.split(":", 1)[1])]
    elif device == "cuda":
        accelerator, devices = "gpu", "auto"
    elif device == "cpu":
        accelerator, devices = "cpu", "auto"
    else:
        accelerator, devices = "auto", "auto"

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs, max_steps=cfg.max_steps, accelerator=accelerator, devices=devices,
        precision=cfg.precision, gradient_clip_val=use_grad_clip,
        callbacks=[checkpoint_cb, prune_cb, lr_monitor, ScheduleFreeModeCallback()],
        # explicit logger so metrics/checkpoints land under exp_dir predictably
        logger=pl.loggers.TensorBoardLogger(save_dir=str(cfg.exp_dir), name="lightning_logs"),
        # val_check_interval: >1 = step count, <=1 = epoch fraction (resolve_val_interval clamps step counts
        # over batches/epoch to per-epoch). Step-based training (max_steps, no max_epochs) MUST use
        # check_val_every_n_epoch=None so an int val_check_interval (possibly larger than batches/epoch) is
        # honored across epochs (lightning raises if both an int interval and a non-None epoch count are given)
        val_check_interval=resolve_val_interval(cfg.val_interval, len(train_dl)),
        check_val_every_n_epoch=1 if cfg.max_epochs else None,
        num_sanity_val_steps=0,
        limit_val_batches=cfg.limit_val_batches if cfg.limit_val_batches is not None else 1.0,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=resume or cfg.resume)
