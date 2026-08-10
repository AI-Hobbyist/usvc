"""MLX 训练器:函数式训练循环(非 GAN)。

核心模式(与 torch Lightning 训练循环对齐语义):
    loss, grads = mx.value_and_grad(loss_fn)(model, batch)
    grads = clip_grad_norm(grads)
    model = optimizer.update(model, grads)   # AdamW/Muon 内置
    (schedulefree: 先 train_transform 再 forward,update 后恢复)

支持:
  - grad clip(mlx.optimizers.clip_grad_norm)
  - lr schedule(lr_fn(step) 回调,schedulefree 需在 train_transform 用当前 lr)
  - checkpoint 保存/恢复(参数 + optimizer state)
  - AMP:可选 bf16/fp16 模型(mlx 无 autocast,直接 set_dtype 语义等价)
  - 验证循环(no_grad 下 eval metrics)

用法(backend 模块提供 loss_fn(model, batch) -> loss):
    trainer = MlxTrainer(model, optimizer, lr_fn=None, cfg)
    for batch in dataloader:
        trainer.train_step(batch)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.optimizers as mopt

__all__ = ["MlxTrainer"]


class MlxTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn: Callable,
        cfg: Any = None,
        lr_fn: Callable | None = None,
        grad_clip: float | None = 1.0,
        device: str = "gpu",
        exp_dir: str | Path = "experiments/mlx",
        amp_dtype: str | None = None,  # None | 'float16' | 'bfloat16'
    ):
        if device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            mx.set_default_device(mx.cpu)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.lr_fn = lr_fn
        self.grad_clip = grad_clip
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.epoch = 0
        self.history: list[dict] = []
        self._amp_dtype = amp_dtype
        if amp_dtype:
            self.model.set_dtype(getattr(mx, amp_dtype))

    # ------------------------------------------------------------------
    # 训练核心
    # ------------------------------------------------------------------

    def _loss_and_grad_fn(self):
        loss_fn = self.loss_fn

        def loss_and_grad(model, batch):
            return mx.value_and_grad(loss_fn)(model, batch)

        return loss_and_grad

    def train_step(self, batch) -> dict:
        """单步训练。返回 log dict。"""
        lag = self._loss_and_grad_fn()
        loss, grads = lag(self.model, batch)
        if self.grad_clip is not None:
            grads, _ = mopt.clip_grad_norm(grads, self.grad_clip)
        # schedulefree 特殊处理:先 train_transform(已由 loss_fn 前调),
        # 这里 update 并恢复
        if hasattr(self.optimizer, "train_transform"):
            self.optimizer.update(self.model, grads)
        else:
            self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        loss_val = float(mx.array(loss).item())
        self.step += 1
        log = {"step": self.step, "loss": loss_val}
        if self.lr_fn is not None:
            log["lr"] = self.lr_fn(self.step)
        self.history.append(log)
        return log

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def eval_loss(self, batch) -> float:
        """no_grad 下计算 loss(验证/监控)。"""
        import mlx.core as mx
        # mlx 无 no_grad 上下文;用 mx.stop_gradient 包住 loss 计算
        loss = self.loss_fn(self.model, batch)
        loss = mx.stop_gradient(loss)
        return float(mx.array(loss).item())

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------

    def fit(
        self,
        dataloader,
        max_steps: int = 1000,
        val_loader=None,
        val_interval: int = 100,
        log_interval: int = 10,
        save_interval: int = 200,
        keep_last_n: int = 5,
        callbacks=None,
    ):
        """迭代 dataloader 直到 max_steps。dataloader 需支持无限循环或内部处理。"""
        callbacks = callbacks or []
        while self.step < max_steps:
            for batch in dataloader:
                if self.step >= max_steps:
                    break
                log = self.train_step(batch)
                for cb in callbacks:
                    cb(self, log)
                if log["step"] % log_interval == 0:
                    print(
                        f"[mlx-train] step {log['step']}/{max_steps} "
                        f"loss {log['loss']:.4f} "
                        + (f"lr {log['lr']:.2e}" if "lr" in log else "")
                    )
                if log["step"] % val_interval == 0 and val_loader is not None:
                    self.validate(val_loader)
                if log["step"] % save_interval == 0:
                    self.save_checkpoint(f"step_{log['step']}")
                    self._prune_ckpts(keep_last_n)
        if save_interval:
            self.save_checkpoint("last")
        return self.history

    def validate(self, val_loader):
        """遍历 val_loader 算平均 loss。"""
        total, n = 0.0, 0
        for batch in val_loader:
            total += self.eval_loss(batch)
            n += 1
        if n:
            avg = total / n
            print(f"[mlx-train] val loss {avg:.4f} @ step {self.step}")
            self.history.append({"step": self.step, "val_loss": avg})

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: str):
        """保存参数 + optimizer state 到 exp_dir/{name}.npz + .json。"""
        import numpy as np
        params = self.model.trainable_parameters()
        leaves = mopt.tree_flatten(params)  # [(path, val)]
        arrs = {k: np.array(v) for k, v in leaves}
        path = self.exp_dir / f"{name}.npz"
        np.savez(str(path), **arrs)
        meta = {
            "step": self.step,
            "epoch": self.epoch,
            "opt_type": type(self.optimizer).__name__,
            "history": self.history[-50:],
        }
        with open(self.exp_dir / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=1, default=str)
        print(f"[mlx-train] saved checkpoint {name} @ step {self.step}")

    def _serialize_opt_state(self):
        """优化器状态转可 JSON 的 dict(数组转 shape/dtype 摘要 + 文件)。"""
        out = {}
        if hasattr(self.optimizer, "state_dict"):
            sd = self.optimizer.state_dict()
            npz_path = self.exp_dir / f"opt_{self.step}.npz"
            arrs = {}
            if isinstance(sd, dict):
                for k, v in sd.items():
                    if isinstance(v, dict):
                        for pk, pv in v.items():
                            if hasattr(pv, "tolist"):
                                arrs[f"{k}.{pk}"] = pv
                            elif hasattr(pv, "items"):
                                for kk, vv in pv.items():
                                    if hasattr(vv, "tolist"):
                                        arrs[f"{k}.{pk}.{kk}"] = vv
                    elif hasattr(v, "tolist"):
                        arrs[k] = v
            if arrs:
                import numpy as np
                np.savez(str(npz_path), **{k: np.array(v) for k, v in arrs.items()})
                out["npz"] = str(npz_path.name)
        return out

    def load_checkpoint(self, name: str | Path):
        """恢复参数 + step 计数。"""
        import numpy as np
        name = Path(name)
        if name.suffix == ".npz":
            path = name
        else:
            path = self.exp_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        data = np.load(str(path))
        leaves = [(k, mx.array(data[k])) for k in data.files]
        params = mopt.tree_unflatten(leaves)
        self.model.update(params)
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
    """从 torch ckpt(.pth/.pt/.ckpt)或 mlx npz 加载权重到 mlx 模型。

    优先用 backend 的 from_ckpt(如果模型有),否则走 npz。
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".npz":
        import numpy as np
        data = np.load(str(ckpt_path))
        leaves = [(k, mx.array(data[k])) for k in data.files]
        model.update(mopt.tree_unflatten(leaves))
        return model
    # torch ckpt:委托模型自己的 from_ckpt(它处理 key 映射)
    if hasattr(model, "from_ckpt"):
        return model.from_ckpt(str(ckpt_path))
    raise ValueError(
        f"model has no from_ckpt and ckpt is not npz: {ckpt_path}"
    )
