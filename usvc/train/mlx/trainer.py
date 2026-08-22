"""MLX 训练器:函数式训练循环(非 GAN)。
    loss, grads = mx.value_and_grad(loss_fn)(model, batch)
    grads = clip_grad_norm(grads); optimizer.update(model, grads)
    (schedulefree: 先 train_transform 再 forward,update 后恢复)
支持 grad clip、lr schedule、checkpoint 保存/恢复、AMP(bf16/fp16 模型,
mlx 无 autocast,set_dtype 语义等价)、验证循环。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable
import mlx.core as mx
import mlx.optimizers as mopt
__all__ = ["MlxTrainer"]
class MlxTrainer:
    def __init__(self, model, optimizer, loss_fn: Callable, cfg: Any = None,
                 lr_fn: Callable | None = None, grad_clip: float | None = 1.0,
                 device: str = "gpu", exp_dir: str | Path = "experiments/mlx",
                 amp_dtype: str | None = None):  # None | 'float16' | 'bfloat16'
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.model, self.optimizer, self.loss_fn = model, optimizer, loss_fn
        self.cfg, self.lr_fn, self.grad_clip = cfg, lr_fn, grad_clip
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.epoch = 0
        self.history: list[dict] = []
        self._amp_dtype = amp_dtype
        if amp_dtype:
            self.model.set_dtype(getattr(mx, amp_dtype))
    def train_step(self, batch) -> dict:  # value_and_grad(model, fn) 自动处理参数树
        import mlx.nn as nn
        if isinstance(batch, dict):  # 过滤非数组字段(破坏 tree_flatten)
            batch = type(batch)({k: v for k, v in batch.items() if hasattr(v, "shape")})
        loss, grads = nn.value_and_grad(
            self.model, lambda b: self.loss_fn(self.model, b))(batch)
        if self.grad_clip is not None:
            grads, _ = mopt.clip_grad_norm(grads, self.grad_clip)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.step += 1
        log = {"step": self.step, "loss": float(mx.array(loss).item())}
        if self.lr_fn is not None:
            log["lr"] = self.lr_fn(self.step)
        self.history.append(log)
        return log
    def eval_loss(self, batch) -> float:  # no_grad 语义:mlx 用 stop_gradient 包住
        return float(mx.array(mx.stop_gradient(self.loss_fn(self.model, batch))).item())
    def fit(self, dataloader, max_steps: int = 1000, val_loader=None,
            val_interval: int = 100, log_interval: int = 10,
            save_interval: int = 200, keep_last_n: int = 5, callbacks=None):
        callbacks = callbacks or []
        while self.step < max_steps:
            for batch in dataloader:
                if self.step >= max_steps:
                    break
                log = self.train_step(batch)
                for cb in callbacks:
                    cb(self, log)
                if log["step"] % log_interval == 0:
                    print(f"[mlx-train] step {log['step']}/{max_steps} "
                          f"loss {log['loss']:.4f} "
                          + (f"lr {log['lr']:.2e}" if "lr" in log else ""))
                if log["step"] % val_interval == 0 and val_loader is not None:
                    self.validate(val_loader)
                if log["step"] % save_interval == 0:
                    self.save_checkpoint(f"step_{log['step']}")
                    self._prune_ckpts(keep_last_n)
        if save_interval:
            self.save_checkpoint("last")
        return self.history
    def validate(self, val_loader):  # 遍历 val_loader 算平均 loss
        total, n = 0.0, 0
        for batch in val_loader:
            total += self.eval_loss(batch)
            n += 1
        if n:
            avg = total / n
            print(f"[mlx-train] val loss {avg:.4f} @ step {self.step}")
            self.history.append({"step": self.step, "val_loss": avg})
    def save_checkpoint(self, name: str):  # exp_dir/{name}.npz + .json
        import numpy as np
        path = self.exp_dir / f"{name}.npz"
        np.savez(str(path), **{k: np.array(v) for k, v in
                               mopt.tree_flatten(self.model.trainable_parameters())})
        meta = {"step": self.step, "epoch": self.epoch,
                "opt_type": type(self.optimizer).__name__,
                "history": self.history[-50:]}
        with open(self.exp_dir / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=1, default=str)
        print(f"[mlx-train] saved checkpoint {name} @ step {self.step}")
    def load_checkpoint(self, name: str | Path):  # 恢复参数 + step/epoch 计数
        import numpy as np
        name = Path(name)
        path = name if name.suffix == ".npz" else self.exp_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        data = np.load(str(path))
        self.model.update(mopt.tree_unflatten([(k, mx.array(data[k])) for k in data.files]))
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.step = int(meta.get("step", 0))
            self.epoch = int(meta.get("epoch", 0))
        print(f"[mlx-train] loaded checkpoint {path} @ step {self.step}")
        return self
    def _prune_ckpts(self, keep_last_n: int):
        if keep_last_n < 0:
            return
        import glob
        ckpts = sorted(glob.glob(str(self.exp_dir / "step_*.npz")))
        for old in ckpts[: max(0, len(ckpts) - keep_last_n)]:
            Path(old).unlink(missing_ok=True)
            Path(old).with_suffix(".json").unlink(missing_ok=True)
def load_pretrained_mlx(model, ckpt_path: str | Path, strict: bool = True):
    """torch ckpt(.pth/.pt/.ckpt)或 mlx npz → mlx 权重(优先 backend from_ckpt)。"""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".npz":
        import numpy as np
        data = np.load(str(ckpt_path))
        model.update(mopt.tree_unflatten([(k, mx.array(data[k])) for k in data.files]))
        return model
    if hasattr(model, "from_ckpt"):
        return model.from_ckpt(str(ckpt_path))
    raise ValueError(f"model has no from_ckpt and ckpt is not npz: {ckpt_path}")
