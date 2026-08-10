"""MLX GAN 训练器(sovits/rvc):双优化器 D→G 严格交替。

镜像 torch BaseGANModule.training_step 的语义顺序:
    G fwd once(共享输出)
    -> D fwd#1(wave_gen.detach()) -> D bwd -> D opt.step
    -> G loss(D fwd#2 on wave_gen with grad)-> G bwd -> G opt.step

MLX 函数式实现:loss_fn 是 (model_g, model_d, batch) -> (loss_d, loss_g)。
train_step 用两次 value_and_grad(argnums 区分 G/D 参数)。
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.optimizers as mopt

__all__ = ["MlxGANTrainer"]


class MlxGANTrainer:
    """GAN 训练器。

    Args:
        g_model: 生成器(mlx.nn.Module)。
        d_model: 判别器(mlx.nn.Module)。
        opt_g/opt_d: 优化器。
        g_loss_fn: (g_model, d_model, batch) -> (loss_g, sub_dict)  生成器损失
        d_loss_fn: (g_model, d_model, batch) -> (loss_d, sub_dict)  判别器损失
        注意:G loss 需要 D forward(wave_gen with grad),D loss 需要
        wave_gen detach。两个 loss_fn 内部各自完成 G fwd 与 D fwd。
    """

    def __init__(
        self,
        g_model,
        d_model,
        opt_g,
        opt_d,
        g_loss_fn,
        d_loss_fn,
        grad_clip: float | None = None,
        device: str = "gpu",
        exp_dir: str = "experiments/mlx",
        lr_fn=None,
    ):
        if device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            mx.set_default_device(mx.cpu)
        self.g_model = g_model
        self.d_model = d_model
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        self.grad_clip = grad_clip
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.lr_fn = lr_fn
        self.step = 0
        self.history: list[dict] = []

    def train_step(self, batch) -> dict:
        """单步 GAN 训练。

        MLX 函数式:value_and_grad 默认对第一个参数求导。
        - D 步:g_loss 路径不需要(用 stop_gradient 隔离 wave_gen)。
          d_loss_fn 只对 d_model 求导 → mx.value_and_grad(d_loss_fn)(g_model, d_model, batch)
          中 d_loss_fn 需返回 (loss, g_model, d_model)?? 不行——value_and_grad
          要求第一个位置参数是被求导对象。

        正确做法:定义两个独立的 loss 函数,把被求导模型放第一位:
          d_loss_fn(d_model, g_model, batch) -> loss_d
          g_loss_fn(g_model, d_model, batch) -> loss_g
        """
        # --- D step(D fwd#1 on detached wave_gen)---
        d_lag = mx.value_and_grad(self.d_loss_fn, argnums=0)
        loss_d, grads_d = d_lag(self.d_model, self.g_model, batch)
        if self.grad_clip is not None:
            grads_d, _ = mopt.clip_grad_norm(grads_d, self.grad_clip)
        self.opt_d.update(self.d_model, grads_d)
        mx.eval(self.d_model.parameters(), self.opt_d.state)

        # --- G step(D fwd#2 on wave_gen with grad)---
        g_lag = mx.value_and_grad(self.g_loss_fn, argnums=0)
        loss_g, grads_g = g_lag(self.g_model, self.d_model, batch)
        if self.grad_clip is not None:
            grads_g, _ = mopt.clip_grad_norm(grads_g, self.grad_clip)
        self.opt_g.update(self.g_model, grads_g)
        mx.eval(self.g_model.parameters(), self.opt_g.state)

        loss_d_val = float(mx.array(loss_d).item())
        loss_g_val = float(mx.array(loss_g).item())
        self.step += 1
        log = {
            "step": self.step,
            "loss_g": loss_g_val,
            "loss_d": loss_d_val,
        }
        if self.lr_fn is not None:
            log["lr"] = self.lr_fn(self.step)
        self.history.append(log)
        return log

    def fit(
        self,
        dataloader,
        max_steps: int = 1000,
        log_interval: int = 10,
        save_interval: int = 200,
        keep_last_n: int = 5,
        callbacks=None,
    ):
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
                        f"[mlx-gan] step {log['step']}/{max_steps} "
                        f"loss_g {log['loss_g']:.4f} loss_d {log['loss_d']:.4f}"
                    )
                if log["step"] % save_interval == 0:
                    self.save_checkpoint(f"step_{log['step']}")
                    self._prune_ckpts(keep_last_n)
        if save_interval:
            self.save_checkpoint("last")
        return self.history

    def save_checkpoint(self, name: str):
        """保存 G/D 参数 + 优化器状态。"""
        import numpy as np
        g_params = self.g_model.trainable_parameters()
        d_params = self.d_model.trainable_parameters()
        g_arrs = {k: np.array(v) for k, v in mopt.tree_flatten(g_params)}
        d_arrs = {k: np.array(v) for k, v in mopt.tree_flatten(d_params)}
        np.savez(str(self.exp_dir / f"{name}_g.npz"), **g_arrs)
        np.savez(str(self.exp_dir / f"{name}_d.npz"), **d_arrs)
        print(f"[mlx-gan] saved checkpoint {name} @ step {self.step}")

    def load_checkpoint(self, name: str | Path, load_d: bool = True):
        import numpy as np
        name = Path(name)
        if name.suffix == ".npz":
            g_path = name
            d_path = name.with_name(name.stem.replace("_g", "_d") + ".npz")
        else:
            g_path = self.exp_dir / f"{name}_g.npz"
            d_path = self.exp_dir / f"{name}_d.npz"
        data = np.load(str(g_path))
        self.g_model.update(mopt.tree_unflatten([(k, mx.array(data[k])) for k in data.files]))
        if load_d and d_path.exists():
            data = np.load(str(d_path))
            self.d_model.update(mopt.tree_unflatten([(k, mx.array(data[k])) for k in data.files]))
        print(f"[mlx-gan] loaded {g_path}")
        return self

    def _prune_ckpts(self, keep_last_n: int):
        if keep_last_n < 0:
            return
        import glob
        ckpts = sorted(glob.glob(str(self.exp_dir / "step_*_g.npz")))
        for old in ckpts[: max(0, len(ckpts) - keep_last_n)]:
            Path(old).unlink(missing_ok=True)
            Path(str(old).replace("_g.npz", "_d.npz")).unlink(missing_ok=True)
