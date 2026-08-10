"""usvc-train-mlx CLI:全 MLX 训练入口(所有 backend)。

用法:
    usvc-train-mlx -c configs/rift_dit_pretrain.yaml
    usvc-train-mlx -c configs/sovits.yaml --resume experiments/mlx/last.npz

流程:
    1. 加载 TrainConfig
    2. 构建 dataset(复用 torch backend.build_dataset + DataLoader 搬运)
    3. 构建 backend mlx 模型 + 训练 module(forward_loss/sample)
    4. 构建优化器 + trainer
    5. fit 循环(step 训练 / val / save)

backend 训练 module 注册表:各 backend 在 usvc/train/mlx/modules/ 提供
build_mlx_module(cfg) -> dict(model_g?, model_d?, loss_fn, ...)。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import TrainConfig


def _resolve_val_interval(val_interval: float | int, n_batches: int) -> int | float:
    if val_interval <= 1:
        return max(float(val_interval), 1e-6)
    return int(val_interval)


def run_training_mlx(
    cfg: TrainConfig,
    resume: str | None = None,
    device: str | None = None,
    max_steps_override: int | None = None,
):
    """全 MLX 训练入口。"""
    import mlx.core as mx
    device = device or "gpu"
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    from ...backends import get_backend
    backend = get_backend(cfg.backend)()

    # ---- dataset(复用 torch 构建逻辑)----
    from ...data.meta import load_meta
    from ...data.feature_config import FeatureConfig

    meta = load_meta(cfg.dataset_dir)
    spec_cfg = dict(cfg.backend_opts or {})
    want_spec = backend.feature_spec(spec_cfg)
    units_kind = (
        want_spec.units.params.get("kind", "contentvec") if want_spec.units else None
    )

    def _match_hash(kind, fc, entries):
        if fc is None or not entries:
            return entries[0]["hash"] if entries else ""
        want = fc.hash()
        for e in entries:
            if e.get("hash") == want:
                return e["hash"]
        for e in entries:
            if e.get("config", {}).get("n_mels") == fc.params.get("n_mels"):
                return e["hash"]
        return entries[0]["hash"]

    feature_hashes: dict[str, str] = {}
    for kind, entries in meta.features.items():
        if kind == "units":
            enc_map = entries or {}
            if units_kind is not None and units_kind in enc_map:
                feature_hashes["units"] = _match_hash(kind, want_spec.units, enc_map[units_kind])
            else:
                for enc_kind, enc_entries in enc_map.items():
                    if enc_entries:
                        feature_hashes["units"] = enc_entries[0]["hash"]
                        break
        elif isinstance(entries, list) and entries:
            fc = getattr(want_spec, kind, None)
            feature_hashes[kind] = _match_hash(kind, fc, entries)
    cfg.feature_hashes = feature_hashes

    train_ds = backend.build_dataset(cfg.dataset_dir, "train", cfg.feature_hashes)
    train_ds.max_frames = cfg.max_frames
    if units_kind and hasattr(train_ds, "encoder_kind") and train_ds.encoder_kind != units_kind:
        train_ds.encoder_kind = units_kind
        train_ds._units_dir = train_ds.dataset_dir / "units" / units_kind / feature_hashes.get("units", "")
    val_ds = backend.build_dataset(cfg.val_dir or cfg.dataset_dir, "val", cfg.feature_hashes)
    val_ds.max_frames = cfg.max_frames
    if units_kind and hasattr(val_ds, "encoder_kind") and val_ds.encoder_kind != units_kind:
        val_ds.encoder_kind = units_kind
        val_ds._units_dir = val_ds.dataset_dir / "units" / units_kind / feature_hashes.get("units", "")

    from .dataset import mlx_dataloader
    train_dl = mlx_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
    )
    val_dl = mlx_dataloader(
        val_ds, batch_size=max(1, cfg.batch_size // 4), shuffle=False,
        num_workers=cfg.num_workers, drop_last=False,
    )

    # ---- 构建 mlx 训练 module(backend 注册)----
    from . import modules
    build = modules.build_mlx_module(cfg)
    model_g, model_d, loss_fns = build

    # pretrained/resume
    if cfg.pretrained:
        from .checkpoint import load_torch_weights
        load_torch_weights(model_g, cfg.pretrained)
        print(f"[mlx-train] loaded pretrained {cfg.pretrained}")

    # ---- 优化器 ----
    from .optimizer import build_optimizer, build_optimizer_gan
    max_steps = max_steps_override or cfg.max_steps

    from ..gan_module import BaseGANModule as _unused  # noqa
    is_gan = model_d is not None

    if is_gan:
        opt_g, opt_d, lr_fn = build_optimizer_gan(model_g, model_d, cfg.optimizer, cfg.schedule, max_steps)
        from .gan import MlxGANTrainer
        trainer = MlxGANTrainer(
            model_g, model_d, opt_g, opt_d,
            g_loss_fn=loss_fns["g"], d_loss_fn=loss_fns["d"],
            grad_clip=cfg.grad_clip, device=device,
            exp_dir=cfg.exp_dir, lr_fn=lr_fn,
        )
    else:
        opt, lr_fn = build_optimizer(model_g, cfg.optimizer, cfg.schedule, max_steps)
        from .trainer import MlxTrainer
        trainer = MlxTrainer(
            model_g, opt, loss_fns["loss"],
            grad_clip=cfg.grad_clip, device=device,
            exp_dir=cfg.exp_dir, lr_fn=lr_fn,
        )

    if resume:
        trainer.load_checkpoint(resume)
    elif cfg.resume:
        trainer.load_checkpoint(cfg.resume)

    # ---- fit ----
    val_interval = _resolve_val_interval(cfg.val_interval, len(train_ds) // max(1, cfg.batch_size))
    trainer.fit(
        train_dl,
        max_steps=max_steps,
        val_loader=val_dl if cfg.limit_val_batches != 0 else None,
        val_interval=int(val_interval) if val_interval > 1 else 1,
        log_interval=max(1, cfg.save_interval // 20),
        save_interval=cfg.save_interval,
        keep_last_n=cfg.keep_last_n,
    )
    return trainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="usvc-train-mlx")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=__import__("sys").stderr)
        return 1
    try:
        cfg = TrainConfig.from_yaml(cfg_path)
    except Exception as exc:
        print(f"error: cannot load config {cfg_path}: {exc}")
        return 1
    try:
        run_training_mlx(
            cfg, resume=args.resume, device=args.device,
            max_steps_override=args.max_steps,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"error: training failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
