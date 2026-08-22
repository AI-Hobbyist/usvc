"""usvc-train-mlx CLI:全 MLX 训练入口(所有 backend)。
    usvc-train-mlx -c configs/rift_dit_pretrain.yaml [--resume experiments/mlx/last.npz]
流程:TrainConfig → 复用 torch backend.build_dataset + DataLoader 搬运 →
backend mlx 模型 + 训练 module(build_mlx_module)→ 优化器 + trainer → fit。"""
from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path
from ..config import TrainConfig
from .._resolvers import resolve_feature_hashes, resolve_val_interval
def _make_ds(backend, split, base_dir, cfg, units_kind, feature_hashes):
    ds = backend.build_dataset(base_dir, split, feature_hashes)
    ds.max_frames = cfg.max_frames
    if units_kind and hasattr(ds, "encoder_kind") and ds.encoder_kind != units_kind:
        ds.encoder_kind = units_kind
        ds._units_dir = ds.dataset_dir / "units" / units_kind / feature_hashes.get("units", "")
    return ds
def run_training_mlx(cfg: TrainConfig, resume: str | None = None,
                     device: str | None = None,
                     max_steps_override: int | None = None):
    """全 MLX 训练入口。"""
    import mlx.core as mx
    device = device or "gpu"
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    from ...backends import get_backend
    backend = get_backend(cfg.backend)()
    want_spec = backend.feature_spec(dict(cfg.backend_opts or {}))
    units_kind = (want_spec.units.params.get("kind", "contentvec")
                  if want_spec.units else None)
    from ...data.meta import load_meta
    cfg.feature_hashes = feature_hashes = resolve_feature_hashes(
        load_meta(cfg.dataset_dir), want_spec, units_kind)
    from .dataset import mlx_dataloader
    train_ds = _make_ds(backend, "train", cfg.dataset_dir, cfg, units_kind, feature_hashes)
    val_ds = _make_ds(backend, "val", cfg.val_dir or cfg.dataset_dir, cfg, units_kind, feature_hashes)
    train_dl = mlx_dataloader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    val_dl = mlx_dataloader(val_ds, batch_size=max(1, cfg.batch_size // 4),
                            shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    from . import modules
    model_g, model_d, loss_fns = modules.build_mlx_module(cfg)
    if cfg.pretrained:
        from .checkpoint import load_torch_weights
        load_torch_weights(model_g, cfg.pretrained)
        print(f"[mlx-train] loaded pretrained {cfg.pretrained}")
    from .optimizer import build_optimizer, build_optimizer_gan
    from ..gan_module import BaseGANModule as _unused  # noqa
    max_steps = max_steps_override or cfg.max_steps
    if model_d is not None:
        opt_g, opt_d, lr_fn = build_optimizer_gan(model_g, model_d, cfg.optimizer,
                                                  cfg.schedule, max_steps)
        from .gan import MlxGANTrainer
        trainer = MlxGANTrainer(model_g, model_d, opt_g, opt_d,
                                g_loss_fn=loss_fns["g"], d_loss_fn=loss_fns["d"],
                                grad_clip=cfg.grad_clip, device=device,
                                exp_dir=cfg.exp_dir, lr_fn=lr_fn)
    else:
        opt, lr_fn = build_optimizer(model_g, cfg.optimizer, cfg.schedule, max_steps)
        from .trainer import MlxTrainer
        trainer = MlxTrainer(model_g, opt, loss_fns["loss"],
                             grad_clip=cfg.grad_clip, device=device,
                             exp_dir=cfg.exp_dir, lr_fn=lr_fn)
    if resume:
        trainer.load_checkpoint(resume)
    elif cfg.resume:
        trainer.load_checkpoint(cfg.resume)
    val_interval = resolve_val_interval(cfg.val_interval,
                                        len(train_ds) // max(1, cfg.batch_size))
    trainer.fit(train_dl, max_steps=max_steps,
                val_loader=val_dl if cfg.limit_val_batches != 0 else None,
                val_interval=int(val_interval) if val_interval > 1 else 1,
                log_interval=max(1, cfg.save_interval // 20),
                save_interval=cfg.save_interval, keep_last_n=cfg.keep_last_n)
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
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    try:
        cfg = TrainConfig.from_yaml(cfg_path)
    except Exception as exc:
        print(f"error: cannot load config {cfg_path}: {exc}")
        return 1
    try:
        run_training_mlx(cfg, resume=args.resume, device=args.device,
                         max_steps_override=args.max_steps)
    except Exception as exc:
        traceback.print_exc()
        print(f"error: training failed: {exc}")
        return 1
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
