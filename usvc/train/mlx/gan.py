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
import mlx.nn as nn
import mlx.optimizers as mopt
__all__ = ["MlxGANTrainer"]
def _arrays(batch):  # 过滤非数组字段(破坏 value_and_grad tree_flatten)
    return type(batch)({k: v for k, v in batch.items() if hasattr(v, "shape")})
def _load(path, m):  # npz → mlx 模型权重
    import numpy as np
    d = np.load(str(path))
    m.update(mopt.tree_unflatten([(k, mx.array(d[k])) for k in d.files]))
class MlxGANTrainer:
    """g_loss_fn/d_loss_fn: (被求导模型放第一位)(model, other, batch) -> loss。
    G loss 需要 D forward(wave_gen with grad),D loss 需要 wave_gen detach;
    两个 loss_fn 内部各自完成 G fwd 与 D fwd。"""
    def __init__(self, g_model, d_model, opt_g, opt_d, g_loss_fn, d_loss_fn,
                 grad_clip: float | None = None, device: str = "gpu",
                 exp_dir: str = "experiments/mlx", lr_fn=None):
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.g_model, self.d_model = g_model, d_model
        self.opt_g, self.opt_d = opt_g, opt_d
        self.g_loss_fn, self.d_loss_fn = g_loss_fn, d_loss_fn
        self.grad_clip = grad_clip
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.lr_fn = lr_fn
        self.step = 0
        self.history: list[dict] = []
    def train_step(self, batch) -> dict:
        batch = _arrays(batch)
        losses = []
        # D step 先行(D fwd#1 on detached wave_gen),再 G step(fwd#2 with grad)
        for model, other, lf, opt in ((self.d_model, self.g_model, self.d_loss_fn, self.opt_d),
                                      (self.g_model, self.d_model, self.g_loss_fn, self.opt_g)):
            lag = nn.value_and_grad(model, lambda b, m=model, o=other, f=lf: f(m, o, b))
            loss, grads = lag(batch)
            if self.grad_clip is not None:
                grads, _ = mopt.clip_grad_norm(grads, self.grad_clip)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(loss)
        self.step += 1
        loss_d, loss_g = losses
        log = {"step": self.step,
               "loss_g": float(mx.array(loss_g).item()),
               "loss_d": float(mx.array(loss_d).item())}
        if self.lr_fn is not None:
            log["lr"] = self.lr_fn(self.step)
        self.history.append(log)
        return log
    def losses(self, batch) -> dict:  # 无梯度前向 G/D loss(验证用)
        batch = _arrays(batch)
        return {"loss_g": float(mx.array(self.g_loss_fn(self.g_model, self.d_model, batch)).item()),
                "loss_d": float(mx.array(self.d_loss_fn(self.d_model, self.g_model, batch)).item())}
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
                    print(f"[mlx-gan] step {log['step']}/{max_steps} "
                          f"loss_g {log['loss_g']:.4f} loss_d {log['loss_d']:.4f}")
                if val_loader is not None and log["step"] % val_interval == 0:
                    self.validate(val_loader)
                if log["step"] % save_interval == 0:
                    self.save_checkpoint(f"step_{log['step']}")
                    self._prune_ckpts(keep_last_n)
        if save_interval:
            self.save_checkpoint("last")
        return self.history
    def validate(self, val_loader):  # 前向 G/D loss 均值(不更新参数)
        g_sum = d_sum = 0.0
        n = 0
        for batch in val_loader:
            log = self.losses(batch)
            g_sum += float(log["loss_g"])
            d_sum += float(log["loss_d"])
            n += 1
        if n:
            print(f"[mlx-gan] val loss_g {g_sum / n:.4f} loss_d {d_sum / n:.4f}")
        return {"val_loss_g": g_sum / max(n, 1), "val_loss_d": d_sum / max(n, 1)}
    def save_checkpoint(self, name: str):  # exp_dir/{name}_g.npz + _d.npz
        import numpy as np
        for kind, m in (("g", self.g_model), ("d", self.d_model)):
            np.savez(str(self.exp_dir / f"{name}_{kind}.npz"),
                     **{k: np.array(v) for k, v in mopt.tree_flatten(m.trainable_parameters())})
        print(f"[mlx-gan] saved checkpoint {name} @ step {self.step}")
    def load_checkpoint(self, name: str | Path, load_d: bool = True):
        name = Path(name)
        if name.suffix == ".npz":
            g_path, d_path = name, name.with_name(name.stem.replace("_g", "_d") + ".npz")
        else:
            g_path, d_path = self.exp_dir / f"{name}_g.npz", self.exp_dir / f"{name}_d.npz"
        _load(g_path, self.g_model)
        if load_d and d_path.exists():
            _load(d_path, self.d_model)
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
