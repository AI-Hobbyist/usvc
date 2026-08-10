"""MLX 优化器工厂。

镜像 usvc/train/optimizer.py 的配置语义:
  - adamw:       mlx.optimizers.AdamW
  - muon_adamw:  mlx.optimizers.Muon + AdamW chained (MultiOptimizer)
  - schedulefree: MLX 无官方实现,提供 AdamWScheduleFree 的 MLX 移植
    (训练/评估模式切换语义与 schedulefree 包一致:train() 时参数带
    lr*(y-z) 修正项,eval() 时恢复 y)。

返回 mlx.optimizers.Optimizer(或其组合)。调度器用 lr_schedule 回调。
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.optimizers as mopt

__all__ = [
    "build_optimizer",
    "build_optimizer_gan",
    "build_lr_schedule",
    "AdamWScheduleFree",
    "MuonAdamW",
]


def build_optimizer(
    model,
    opt_cfg: Any,
    sched_cfg: Any = None,
    max_steps: int | None = None,
):
    """构建 MLX 优化器(含 lr schedule 回调)。

    Args:
        model: mlx.nn.Module(用于 trainable_parameters)。
        opt_cfg: OptimizerConfig-like(有 .type/.lr/.betas/.eps/.weight_decay)。
        sched_cfg: ScheduleConfig-like(可选)。
        max_steps: 用于 linear_warmup_decay 调度。

    Returns:
        (optimizer, lr_fn) — lr_fn(step)->lr,可为 None。
    """
    opt = _build_single(opt_cfg)
    lr_fn = build_lr_schedule(sched_cfg, max_steps, base_lr=opt_cfg.lr)
    return opt, lr_fn


def build_optimizer_gan(
    g_model,
    d_model,
    opt_cfg: Any,
    sched_cfg: Any = None,
    max_steps: int | None = None,
):
    """GAN:返回 (opt_g, opt_d, lr_fn)。"""
    opt_g = _build_single(opt_cfg)
    opt_d = _build_single(opt_cfg)
    lr_fn = build_lr_schedule(sched_cfg, max_steps, base_lr=opt_cfg.lr)
    return opt_g, opt_d, lr_fn


def _build_single(cfg: Any):
    if cfg.type == "schedulefree":
        return AdamWScheduleFree(
            learning_rate=cfg.lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay,
            eps=cfg.eps,
        )
    elif cfg.type == "muon_adamw":
        return MuonAdamW(
            lr=cfg.lr,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            muon_momentum=cfg.muon_momentum,
            muon_weight_decay=cfg.muon_weight_decay,
        )
    elif cfg.type == "adamw":
        return mopt.AdamW(
            learning_rate=cfg.lr,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.type}")


def build_lr_schedule(sched_cfg: Any, max_steps: int | None, base_lr: float):
    """构建 lr 调度函数 lr(step)->float。None 表示恒 lr。"""
    if sched_cfg is None or sched_cfg.type is None:
        return None
    t = sched_cfg.type
    if t == "step":
        step_size = sched_cfg.decay_step
        gamma = sched_cfg.gamma

        def lr_fn(step):
            return base_lr * (gamma ** (step // step_size))
    elif t == "exponential":
        gamma = sched_cfg.gamma

        def lr_fn(step):
            return base_lr * (gamma ** step)
    elif t == "linear_warmup_decay":
        warmup = sched_cfg.warmup_steps
        decay = max_steps if max_steps else sched_cfg.decay_step
        min_lr = getattr(sched_cfg, "min_lr", 0.0)

        def lr_fn(step):
            if step < warmup:
                return base_lr * (step / max(1, warmup))
            progress = (step - warmup) / max(1, decay - warmup)
            return max(min_lr, base_lr * (1.0 - progress))
    else:
        raise ValueError(f"Unknown scheduler type: {t}")
    return lr_fn


class AdamWScheduleFree(mopt.Optimizer):
    """AdamWScheduleFree 的 MLX 移植(schedulefree 论文算法)。

    与 schedulefree 包语义对齐:
      - 维护参数 y(实际优化变量)与 z(y 的指数移动平均)。
      - train 模式:前向用 x = y + lr*(y - z),step 更新 y(Adam 规则),
        z = z + beta*(y_new - z)。
      - eval 模式:前向用 y(不再加修正项)。

    state 结构:
      - m/v: Adam 动量(按参数树)
      - z: y 的 EMA(按参数树)
    训练循环需调用 train_transform() 在 forward 前变换参数,
    eval 时用 eval_restore() 恢复。update() 返回 (new_params, new_z)。
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self._learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._z = None  # 参数树
        self._m = None  # 参数树
        self._v = None  # 参数树
        self._step = 0
        self._initialized = False
        self._tm = mopt.tree_map

    @property
    def learning_rate(self):
        return self._learning_rate

    def init(self, params: dict):
        """初始化 z=m=v=params。"""
        self._z = self._tm(lambda p: mx.array(p, dtype=mx.float32), params)
        self._m = self._tm(lambda p: mx.zeros(p.shape, dtype=mx.float32), params)
        self._v = self._tm(lambda p: mx.zeros(p.shape, dtype=mx.float32), params)
        self._initialized = True

    def set_learning_rate(self, lr: float):
        self._learning_rate = lr

    def train_transform(self, params: dict, lr: float | None = None) -> dict:
        """x = y + lr*(y - z)。训练前对参数做变换,返回 x 树。"""
        lr = lr if lr is not None else self._learning_rate
        if self._z is None:
            return params
        return self._tm(
            lambda y, z: y + lr * (y - z), params, self._z
        )

    def eval_restore(self, params: dict) -> dict:
        """eval 模式:直接返回 y(不加修正项)。"""
        return params

    def state_dict(self):
        return {"z": self._z, "m": self._m, "v": self._v, "step": self._step}

    def load_state_dict(self, sd):
        self._z = sd["z"]
        self._m = sd["m"]
        self._v = sd["v"]
        self._step = sd["step"]

    def update(self, params_or_model, grads):
        """Adam 步更新 y 与 z。返回 (new_params, new_z)。"""
        params = (
            params_or_model.trainable_parameters()
            if hasattr(params_or_model, "trainable_parameters")
            else params_or_model
        )
        if not self._initialized or self._z is None:
            self.init(params)
        self._step += 1
        beta1, beta2 = self.betas
        lr = self._learning_rate
        # 偏差修正
        bc1 = 1 - beta1 ** self._step
        bc2 = 1 - beta2 ** self._step

        def step_one(p, g, m, v, z):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            mhat = m / bc1
            vhat = v / bc2
            # AdamW:weight_decay 直接在参数上衰减
            p_new = p * (1 - lr * self.weight_decay) - lr * mhat / (mx.sqrt(vhat) + self.eps)
            z_new = z + beta1 * (p_new - z)
            return p_new, m, v, z_new

        new_params = {}
        new_m = {}
        new_v = {}
        new_z = {}
        p_leaves = mopt.tree_flatten(params)          # [(path, val)]
        g_leaves = mopt.tree_flatten(grads)
        m_leaves = mopt.tree_flatten(self._m)
        v_leaves = mopt.tree_flatten(self._v)
        z_leaves = mopt.tree_flatten(self._z)
        paths = [p[0] for p in p_leaves]
        p_vals = [p[1] for p in p_leaves]
        for i, p in enumerate(p_vals):
            p_new, m_new, v_new, z_new = step_one(
                p, g_leaves[i][1], m_leaves[i][1], v_leaves[i][1], z_leaves[i][1]
            )
            new_params[paths[i]] = p_new
            new_m[paths[i]] = m_new
            new_v[paths[i]] = v_new
            new_z[paths[i]] = z_new
        self._m = mopt.tree_unflatten(list(new_m.items()))
        self._v = mopt.tree_unflatten(list(new_v.items()))
        self._z = mopt.tree_unflatten(list(new_z.items()))
        new_params_tree = mopt.tree_unflatten(list(new_params.items()))
        if hasattr(params_or_model, "update"):
            params_or_model.update(new_params_tree)
            return params_or_model, self._z
        return new_params_tree, self._z


class MuonAdamW(mopt.MultiOptimizer):
    """Muon + AdamW chained(DDSP-SVC 6.3 style)。

    2D+ 权重 → Muon,1D 参数(bias/norm/embedding)→ AdamW。
    """

    def __init__(
        self,
        lr: float = 1e-4,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.1,
        nesterov: bool = True,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.muon_momentum = muon_momentum
        self.muon_weight_decay = muon_weight_decay
        self.nesterov = nesterov
        self._adamw = mopt.AdamW(
            learning_rate=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay
        )
        self._muon = mopt.Muon(
            learning_rate=lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
            nesterov=nesterov,
        )

    def _split(self, params: dict, grads: dict):
        """按参数 ndim 分组:2D+ → Muon,1D → AdamW。"""
        p2, g2, p1, g1 = {}, {}, {}, {}
        for k in params.keys():
            if params[k].ndim >= 2:
                p2[k], g2[k] = params[k], grads[k]
            else:
                p1[k], g1[k] = params[k], grads[k]
        return p2, g2, p1, g1

    def update(self, model_or_params, grads):
        if hasattr(model_or_params, "trainable_parameters"):
            params = model_or_params.trainable_parameters()
        else:
            params = model_or_params
        p2, g2, p1, g1 = self._split(params, grads)
        new2 = self._muon.update(p2, g2) if p2 else p2
        new1 = self._adamw.update(p1, g1) if p1 else p1
        new = {}
        for k in params.keys():
            new[k] = new2.get(k, new1.get(k))
        if hasattr(model_or_params, "update"):
            model_or_params.update(new)
            return model_or_params
        return new
