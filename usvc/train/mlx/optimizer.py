"""MLX 优化器工厂(镜像 usvc/train/optimizer.py):adamw=mlx.optimizers.AdamW,
muon_adamw=Muon+AdamW chained,schedulefree=AdamWScheduleFree 的 MLX 移植。
返回 mlx.optimizers.Optimizer;调度器用 lr_fn(step)->lr 回调。"""
from __future__ import annotations
from typing import Any
import mlx.core as mx
import mlx.optimizers as mopt
__all__ = ["build_optimizer", "build_optimizer_gan", "build_lr_schedule",
           "AdamWScheduleFree", "MuonAdamW"]
def build_optimizer(model, opt_cfg: Any, sched_cfg: Any = None,
                    max_steps: int | None = None):  # → (optimizer, lr_fn)
    opt = _build_single(opt_cfg)
    return opt, build_lr_schedule(sched_cfg, max_steps, base_lr=opt_cfg.lr)
def build_optimizer_gan(g_model, d_model, opt_cfg: Any, sched_cfg: Any = None,
                        max_steps: int | None = None):  # → (opt_g, opt_d, lr_fn)
    lr_fn = build_lr_schedule(sched_cfg, max_steps, base_lr=opt_cfg.lr)
    return _build_single(opt_cfg), _build_single(opt_cfg), lr_fn
def _build_single(cfg: Any):
    if cfg.type == "schedulefree":
        return AdamWScheduleFree(learning_rate=cfg.lr, betas=tuple(cfg.betas),
                                 weight_decay=cfg.weight_decay, eps=cfg.eps)
    if cfg.type == "muon_adamw":
        return MuonAdamW(lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps,
                         weight_decay=cfg.weight_decay,
                         muon_momentum=cfg.muon_momentum,
                         muon_weight_decay=cfg.muon_weight_decay)
    if cfg.type == "adamw":
        return mopt.AdamW(learning_rate=cfg.lr, betas=tuple(cfg.betas),
                          eps=cfg.eps, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer type: {cfg.type}")
def build_lr_schedule(sched_cfg: Any, max_steps: int | None, base_lr: float):
    """lr(step)->float;None=恒 lr。"""
    if sched_cfg is None or sched_cfg.type is None:
        return None
    t = sched_cfg.type
    if t == "step":
        step_size, gamma = sched_cfg.decay_step, sched_cfg.gamma
        return lambda step: base_lr * (gamma ** (step // step_size))
    if t == "exponential":
        gamma = sched_cfg.gamma
        return lambda step: base_lr * (gamma ** step)
    if t == "linear_warmup_decay":
        warmup = sched_cfg.warmup_steps
        decay = max_steps if max_steps else sched_cfg.decay_step
        min_lr = getattr(sched_cfg, "min_lr", 0.0)
        def lr_fn(step):
            if step < warmup:
                return base_lr * (step / max(1, warmup))
            return max(min_lr, base_lr * (1.0 - (step - warmup) / max(1, decay - warmup)))
        return lr_fn
    raise ValueError(f"Unknown scheduler type: {t}")
class AdamWScheduleFree(mopt.Optimizer):
    """schedulefree 算法 MLX 移植:train 前向用 x=y+lr*(y-z)(train_transform),
    step 更新 y(Adam)+z=z+beta1*(y_new-z);eval 用 y(eval_restore)。"""
    def __init__(self, learning_rate: float = 1e-3, betas=(0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__()
        self._learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._z = self._m = self._v = None  # 参数树
        self._step = 0
        self._initialized = False
        self._tm = mopt.tree_map
    @property
    def learning_rate(self):
        return self._learning_rate
    def init(self, params: dict):
        self._z = self._tm(lambda p: mx.array(p, dtype=mx.float32), params)
        self._m = self._tm(lambda p: mx.zeros(p.shape, dtype=mx.float32), params)
        self._v = self._tm(lambda p: mx.zeros(p.shape, dtype=mx.float32), params)
        self._initialized = True
    def set_learning_rate(self, lr: float):
        self._learning_rate = lr
    def train_transform(self, params: dict, lr: float | None = None) -> dict:
        lr = lr if lr is not None else self._learning_rate
        return params if self._z is None else self._tm(
            lambda y, z: y + lr * (y - z), params, self._z)
    def eval_restore(self, params: dict) -> dict:
        return params
    def state_dict(self):
        return {"z": self._z, "m": self._m, "v": self._v, "step": self._step}
    def load_state_dict(self, sd):
        self._z, self._m, self._v, self._step = sd["z"], sd["m"], sd["v"], sd["step"]
    def update(self, params_or_model, grads):
        params = (params_or_model.trainable_parameters()
                  if hasattr(params_or_model, "trainable_parameters")
                  else params_or_model)
        if not self._initialized or self._z is None:
            self.init(params)
        self._step += 1
        beta1, beta2 = self.betas
        lr = self._learning_rate
        bc1 = 1 - beta1 ** self._step  # 偏差修正
        bc2 = 1 - beta2 ** self._step
        def step_one(p, g, m, v, z):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            # AdamW:weight_decay 直接在参数上衰减
            p_new = p * (1 - lr * self.weight_decay) - lr * (m / bc1) / (
                mx.sqrt(v / bc2) + self.eps)
            z_new = z + beta1 * (p_new - z)
            return p_new, m, v, z_new
        new_params, new_m, new_v, new_z = {}, {}, {}, {}
        for (path, p), (_, g), (_, m), (_, v), (_, z) in zip(
            mopt.tree_flatten(params), mopt.tree_flatten(grads),
            mopt.tree_flatten(self._m), mopt.tree_flatten(self._v),
            mopt.tree_flatten(self._z),
        ):
            p_new, m_new, v_new, z_new = step_one(p, g, m, v, z)
            new_params[path], new_m[path], new_v[path], new_z[path] = p_new, m_new, v_new, z_new
        self._m = mopt.tree_unflatten(list(new_m.items()))
        self._v = mopt.tree_unflatten(list(new_v.items()))
        self._z = mopt.tree_unflatten(list(new_z.items()))
        new_params_tree = mopt.tree_unflatten(list(new_params.items()))
        if hasattr(params_or_model, "update"):
            params_or_model.update(new_params_tree)
            return params_or_model, self._z
        return new_params_tree, self._z
class MuonAdamW(mopt.MultiOptimizer):
    """Muon + AdamW chained(DDSP-SVC 6.3 style):2D+ 权重→Muon,1D→AdamW。"""
    def __init__(self, lr: float = 1e-4, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0, muon_momentum: float = 0.95,
                 muon_weight_decay: float = 0.1, nesterov: bool = True):
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.muon_momentum, self.muon_weight_decay, self.nesterov = (
            muon_momentum, muon_weight_decay, nesterov)
        self._adamw = mopt.AdamW(learning_rate=lr, betas=tuple(betas),
                                 eps=eps, weight_decay=weight_decay)
        self._muon = mopt.Muon(learning_rate=lr, momentum=muon_momentum,
                               weight_decay=muon_weight_decay, nesterov=nesterov)
    def _split(self, params: dict, grads: dict):  # 2D+ → Muon,1D → AdamW
        hi = {k: p for k, p in params.items() if p.ndim >= 2}
        lo = {k: p for k, p in params.items() if p.ndim < 2}
        return hi, {k: grads[k] for k in hi}, lo, {k: grads[k] for k in lo}
    def update(self, model_or_params, grads):
        params = (model_or_params.trainable_parameters()
                  if hasattr(model_or_params, "trainable_parameters")
                  else model_or_params)
        p2, g2, p1, g1 = self._split(params, grads)
        new2 = self._muon.update(p2, g2) if p2 else p2
        new1 = self._adamw.update(p1, g1) if p1 else p1
        new = {k: new2.get(k, new1.get(k)) for k in params.keys()}
        if hasattr(model_or_params, "update"):
            model_or_params.update(new)
            return model_or_params
        return new
