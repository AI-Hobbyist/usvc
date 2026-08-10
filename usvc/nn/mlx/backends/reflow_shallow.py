"""MLX 版 reflow_shallow backend(mel-domain 浅扩散增强器)。

镜像 ``usvc/backends/reflow_shallow/model.py``:condition = mel + units/f0/vol
嵌入,velocity 网络做 "dirty → clean" 定向修正(而非盲重建)。

结构与 torch 对照:
  - ``ReflowShallowModel``:unit/f0/volume/spk 嵌入 + LYNXNet2 velocity + RectifiedFlow
  - ``LYNXNet2`` / ``LYNXNet2Block`` / ``SwiGLU``:DDSP-SVC-6.3 velocity 网络
    (ddsp6/model.py;ddsp6 mlx backend 未就绪前独立实现,键名与 torch 一致)
  - ``RectifiedFlow``:euler/rk4 ODE 采样(ddsp6/model.py,spec 归一化同款)

布局约定(见 MLX 映射手册):
  - 全程 NLC:spec [B, T, M],cond [B, T, H],velocity_fn 输出 [B, T, M]。
    torch 的 [B,1,M,T] 4-D 布局在 mlx 侧直接省去(squeeze/unsqueeze 恒等)。
  - LayerNorm/Linear 天然 NLC;Conv1d(dim, dim, k, groups=dim) 深度分离卷积
    在 NLC 上直接等价 torch 的 Transpose+Conv1d(groups=dim)(weight 转置见
    _convert 启发式 (O,C,K)->(O,K,C))。
  - 采样全程 fp32:ODE 迭代数值敏感。
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "SwiGLU",
    "SinusoidalPosEmb",
    "Transpose",
    "LYNXNet2Block",
    "LYNXNet2",
    "RectifiedFlow",
    "ReflowShallowModel",
]


# ---------------------------------------------------------------------------
# LYNXNet2 velocity 网络(DDSP-SVC-6.3,ddsp6/model.py verbatim,独立实现)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """正弦 step 嵌入(镜像 ddsp6/model.py::SinusoidalPosEmb)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.half_dim = half_dim
        emb = math.log(10000.0) / (half_dim - 1)
        self.freqs = mx.exp(mx.arange(half_dim) * -emb)  # [half_dim] fp32

    def __call__(self, x):
        """x: [B] step → [B, dim] 正弦嵌入(fp32)。"""
        x = x.astype(mx.float32)
        x = x[:, None] * self.freqs[None, :]
        return mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)


class SwiGLU(nn.Module):
    """SwiGLU(LYNXNet2 版:silu gate,无 fp16 clamp)。"""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        # torch: out, gate = torch.split(x, x.size(dim)//2, dim=dim)
        out, gate = mx.split(x, 2, axis=self.dim)
        return out * nn.silu(gate)


class Transpose(nn.Module):
    """占位 Transpose(无参数)。torch 的 [B,C,T]<->[B,T,C] 布局切换在 NLC
    约定下恒等;保留该模块仅为 state_dict 键名与 torch Sequential 对齐
    (net.0=LayerNorm, net.1=Transpose, net.2=Conv1d, ...)。"""

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims must be a tuple of two dimensions"
        self.dims = dims

    def __call__(self, x):
        return x


class LYNXNet2Block(nn.Module):
    """LYNXNet2 残差块(NLC 版,镜像 ddsp6/model.py::LYNXNet2Block)。

    net 用普通 list 保持与 torch Sequential 相同的键位:
      net.0=LayerNorm, net.2=Conv1d(depthwise), net.4/6/8=Linear;
    Transpose/SwiGLU/Dropout 无参数仅占位。
    """

    def __init__(self, dim: int, expansion_factor: int, kernel_size: int = 31,
                 dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        _dropout = nn.Dropout(dropout) if float(dropout) > 0.0 else _Noop()
        self.net = [
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            _dropout,
        ]

    def __call__(self, x):
        y = x
        for m in self.net:
            y = m(y)
        return x + y


class _Noop(nn.Module):
    """零参数占位(dropout=0 时替代 nn.Dropout,无运行时开销)。"""

    def __call__(self, x):
        return x


class LYNXNet2(nn.Module):
    """LYNXNet2 velocity 网络(NLC 版)。

    forward(spec [B,T,M], diffusion_step [B], cond [B,T,dim_cond]) -> [B,T,M]。
    镜像 ddsp6/model.py::LYNXNet2,去掉 4-D [B,1,M,T] 兼容分支。
    """

    def __init__(self, in_dims: int, dim_cond: int, n_layers: int = 6,
                 n_chans: int = 512, n_dilates: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)
        # [SinusoidalPosEmb, Linear, GELU, Linear] —— 键名与 torch Sequential 一致
        self.diffusion_embedding = [
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans),
        ]
        self.residual_layers = [
            LYNXNet2Block(
                dim=n_chans,
                expansion_factor=n_dilates,
                kernel_size=31,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
        # torch: nn.init.zeros_(output_projection.weight)
        self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond):
        """spec [B,T,M];diffusion_step [B];cond [B,T,dim_cond] → [B,T,M]。"""
        x = self.input_projection(spec)                       # [B,T,n_chans]
        x = x + self.conditioner_projection(cond)             # [B,T,n_chans]
        step_emb = self.diffusion_embedding[0](diffusion_step)   # [B,n_chans]
        step_emb = self.diffusion_embedding[1](step_emb)
        step_emb = self.diffusion_embedding[2](step_emb)
        step_emb = self.diffusion_embedding[3](step_emb)      # [B,n_chans]
        x = x + step_emb[:, None, :]                          # [B,1,n_chans] 广播
        for layer in self.residual_layers:
            x = layer(x)
        x = self.norm(x)
        return self.output_projection(x)                      # [B,T,M]


# ---------------------------------------------------------------------------
# RectifiedFlow(ddsp6/model.py verbatim;NLC 版,euler/rk4)
# ---------------------------------------------------------------------------

class RectifiedFlow(nn.Module):
    """RectifiedFlow ODE 采样(镜像 ddsp6/model.py::RectifiedFlow)。

    forward(condition [B,T,H], gt_spec [B,T,M] 或 None, infer=True,...)
      - infer=False:reflow_loss(l2_lognorm)
      - infer=True:从 gt_spec(混合噪声)或纯噪声出发 euler/rk4 积分
    返回 [B,T,M](infer)或标量 loss(训练)。
    """

    def __init__(self, velocity_fn, out_dims: int = 128,
                 spec_min: float = -12, spec_max: float = 2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max

    # -- ODE 单步 ----------------------------------------------------------

    def sample_euler(self, x, t, dt, cond):
        x = x + self.velocity_fn(x, 1000 * t, cond) * dt
        t = t + dt
        return x, t

    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, 1000 * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, 1000 * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, 1000 * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, 1000 * (t + dt), cond)
        x = x + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t = t + dt
        return x, t

    # -- 训练 loss(l2_lognorm,与 ddsp6 逐位一致)----------------------------

    def reflow_loss(self, x_1, t, cond, loss_type="l2_lognorm", mask=None):
        x_0 = mx.random.normal(shape=x_1.shape, dtype=mx.float32)
        x_t = x_0 + t[:, None, None] * (x_1 - x_0)          # [B,T,M]
        v_pred = self.velocity_fn(x_t, 1000 * t, cond)
        if mask is not None:
            # mask [B, T] → [B, T, 1] 广播到 [B,T,M](NLC 3D layout)
            m = mask.astype(x_1.dtype)[:, :, None]
            elem = mx.square(x_1 - x_0 - v_pred)
            if loss_type == "l2_lognorm":
                weights = 0.398942 / t / (1 - t) * mx.exp(
                    -0.5 * mx.square(mx.log(t / (1 - t)))
                )
                elem = weights[:, None, None] * elem
            loss = (elem * m).sum() / (m.sum() * x_1.shape[2] + 1e-8)
        elif loss_type == "l1":
            loss = mx.abs(x_1 - x_0 - v_pred).mean()
        elif loss_type == "l2":
            loss = mx.mean(mx.square(x_1 - x_0 - v_pred))
        elif loss_type == "l2_lognorm":
            weights = 0.398942 / t / (1 - t) * mx.exp(
                -0.5 * mx.square(mx.log(t / (1 - t)))
            )
            loss = mx.mean(weights[:, None, None] * mx.square(x_1 - x_0 - v_pred))
        else:
            raise NotImplementedError(loss_type)
        return loss

    # -- 主入口 ------------------------------------------------------------

    def __call__(self, condition, gt_spec=None, infer: bool = True,
                 infer_step: int = 10, method: str = "euler",
                 t_start: float = 0.0, mask=None):
        """condition [B,T,H];gt_spec [B,T,M] 或 None。

        infer=True → [B,T,M] denormalized mel;infer=False → 标量 loss。
        """
        cond = condition                                  # [B,T,H](NLC)
        b = condition.shape[0]
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec)                 # [B,T,M]
            t = t_start + (1.0 - t_start) * mx.random.uniform(shape=(b,))
            t = mx.clip(t, 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, t, cond=cond, mask=mask)
        else:
            shape = (b, condition.shape[1], self.out_dims)  # [B,T,M]
            if gt_spec is None:
                x = mx.random.normal(shape=shape, dtype=mx.float32)
                t = mx.zeros((b,))
                dt = 1.0 / infer_step
            else:
                norm_spec = self.norm_spec(gt_spec)       # [B,T,M]
                x = t_start * norm_spec + (1 - t_start) * mx.random.normal(
                    shape=shape, dtype=mx.float32
                )
                t = mx.full((b,), t_start)
                dt = (1.0 - t_start) / infer_step
            for _ in range(infer_step):
                if method == "euler":
                    x, t = self.sample_euler(x, t, dt, cond)
                elif method == "rk4":
                    x, t = self.sample_rk4(x, t, dt, cond)
                else:
                    raise NotImplementedError(method)
            # Clamp ODE 输出到 [-1,1] 再 denorm(与 torch 一致,防 vocoder 过冲)
            x = mx.clip(x, -1.0, 1.0)
            return self.denorm_spec(x)

    def norm_spec(self, x):
        x = x.astype(mx.float32)
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# ---------------------------------------------------------------------------
# ReflowShallowModel(镜像 reflow_shallow/model.py)
# ---------------------------------------------------------------------------

class ReflowShallowModel(nn.Module):
    """mel-conditioned 浅扩散增强器(NLC 版)。

    sample(units, f0, volume, sovits_mel, ...) -> [B, T, out_dims] mel。
    condition = norm(mel) + units/f0/vol/spk 嵌入(见 embed_condition)。
    """

    def __init__(
        self,
        out_dims: int = 128,
        n_layers: int = 6,
        n_chans: int = 768,
        n_units: int = 768,
        n_spk: int = 1,
        n_hidden: int = 256,
    ):
        super().__init__()
        self.out_dims = out_dims
        self.n_spk = n_spk

        # ---- auxiliary condition embeddings ----
        self.unit_embed = nn.Linear(n_units, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        else:
            self.spk_embed = None

        # ---- reflow(LYNXNet2 velocity + RectifiedFlow)----
        velocity_fn = LYNXNet2(
            in_dims=out_dims,
            dim_cond=out_dims + n_hidden,   # mel + aux embeddings
            n_layers=n_layers,
            n_chans=n_chans,
        )
        self.reflow_model = RectifiedFlow(
            velocity_fn, out_dims=out_dims, spec_min=-12, spec_max=2
        )

    def embed_condition(self, units, f0, volume, mel, spk_id=None):
        """→ [B, T, out_dims + n_hidden] condition:norm(mel) + embeddings。"""
        if f0.ndim == 2:
            f0 = f0[:, :, None]
        f0 = f0.astype(mx.float32)
        emb = self.unit_embed(units) + self.f0_embed(mx.log(1 + f0 / 700))
        if volume is not None:
            if volume.ndim == 2:
                volume = volume[:, :, None]
            emb = emb + self.volume_embed(volume.astype(mx.float32))
        if self.spk_embed is not None and spk_id is not None:
            spk = spk_id if spk_id.ndim > 1 else spk_id[:, None]
            emb = emb + self.spk_embed(spk.astype(mx.int32))
        norm_mel = self.reflow_model.norm_spec(mel)          # [B,T,M]
        return mx.concatenate([norm_mel, emb], axis=-1)      # [B,T,M+H]

    # -- degrade(no-grad,与 torch _degrade 一致)---------------------------

    @staticmethod
    def _sample_degrade_params():
        """随机采样 degrade 参数(在标定范围内均匀分布)。"""
        import random
        return {
            "bias": random.uniform(0.4, 0.8),
            "noise_amp": random.uniform(0.35, 0.75),
            "smooth_win": 5,
        }

    @staticmethod
    def _degrade(mel, bias, noise_amp, smooth_win):
        """模拟 RVC 伪影的 degrade 操作(纯 mlx,可被 value_and_grad 追踪)。

        mel: [B, T, M](dB 域)→ 同 shape。镜像 torch reflow_shallow
        _degrade:近平坦偏置 + 时域平滑噪声(频率衰减 exp(-bin/62),
        lag1 自相关 ~0.7-0.8)。输出 stop_gradient(训练时条件不用梯度)。

        grouped conv1d(torch F.conv1d groups=M)用 as_strided 窗口 + einsum
        向量化实现(无 python 循环,纯 mlx 算子)。
        """
        B, T, M = mel.shape
        dtype = mel.dtype

        # 近平坦偏置;极低频 bin 只有 40%
        bias_curve = mx.full((M,), bias, dtype=dtype)
        idx_m = mx.arange(M, dtype=mx.float32)
        bias_curve = mx.where(
            idx_m < 16, mx.full((M,), bias * 0.4, dtype=dtype), bias_curve
        )

        # 时域平滑噪声,频率衰减幅度(exp(-bin/62))
        taper = mx.exp(-idx_m / 62.0)
        noise_raw = mx.random.normal(shape=(B, M, T), dtype=mx.float32) * taper[None, :, None]
        win = max(3, int(smooth_win))
        # torch F.conv1d(groups=M, padding=win//2): pad 后 valid 卷积(同长)
        pad = win // 2
        # 每通道 pad [T] -> [T+2pad] 再窗口化 [T, win] -> matmul
        nw = mx.pad(
            noise_raw.transpose(0, 2, 1),  # [B, T, M]
            [(0, 0), (pad, pad), (0, 0)],
        )  # [B, T+2pad, M]
        # as_strided 窗口:[B, T, win, M](行主序 strides)
        Bc, Tc, Mc = nw.shape
        s2 = 1
        s1 = s2 * Mc
        s0 = s1 * Tc
        w = mx.as_strided(
            nw,
            shape=(Bc, T, win, Mc),
            strides=(s0, s1, s1, s2),
        )
        # kernel [win, M] -> einsum: [B, T, win, M] * [win, M] -> [B, T, M]
        kernel = mx.full((win, M), 1.0 / win, dtype=mx.float32)
        noise = mx.einsum("btwm,wm->btm", w, kernel)
        noise = noise * (win ** 0.5) * noise_amp  # 撤销移动平均收缩

        degraded = mel + noise + bias_curve[None, None, :]
        return mx.stop_gradient(degraded)

    def forward_loss(self, units, f0, volume, real_mel, spk_id=None,
                     t_start: float = 0.0, frame_len=None):
        """训练 loss:degrade(clean_mel) 条件 + clean mel 目标。

        镜像 torch forward_loss(reflow_shallow/model.py:153)。
        """
        dp = self._sample_degrade_params()
        degraded_mel = self._degrade(real_mel, **dp)
        cond = self.embed_condition(units, f0, volume, degraded_mel, spk_id)
        from ....train.mlx.objective import lens_to_mask
        mask = None
        if frame_len is not None:
            mask = lens_to_mask(frame_len, length=real_mel.shape[1])
        loss = self.reflow_model(
            condition=cond,
            gt_spec=real_mel,  # 目标是 clean mel
            infer=False,
            t_start=t_start,
            mask=mask,
        )
        return loss

    def sample(self, units, f0, volume, sovits_mel, t_start: float = 0.5,
               infer_step: int = 10, method: str = "euler", spk_id=None):
        """浅扩散 sovits_mel → 干净 mel [B, T, out_dims](dB 域)。

        Args:
            sovits_mel: [B, T, M] sovits/rvc 输出 mel(dB 域)。
            t_start: ODE 起点(浅扩散 0.3–0.7)。
            infer_step: ODE 步数。
            method: 'euler' | 'rk4'。
        """
        cond = self.embed_condition(units, f0, volume, sovits_mel, spk_id)
        return self.reflow_model(
            condition=cond,
            gt_spec=sovits_mel,
            infer=True,
            infer_step=infer_step,
            method=method,
            t_start=t_start,
        )

    # -- ckpt 加载 ----------------------------------------------------------

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, key_prefix: str | None = "model",
                  strict: bool = False, dtype=None):
        """从 torch reflow_shallow ckpt 加载(推理)。

        键形如 ``model.reflow_model.velocity_fn.*`` / ``model.unit_embed.*``
        (lightning wrapper);也兼容 ``module.`` 前缀或无前缀。
        """
        from .._convert import load_torch_ckpt_to_mlx

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype or mx.float32)
        if key_prefix:
            prefix = key_prefix + "."
            stripped = {k[len(prefix):]: v for k, v in sd.items()
                        if k.startswith(prefix)}
            if not stripped:
                stripped = sd
            sd = stripped
        cfg = cls._infer_config(sd)
        obj = cls(**cfg)
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        """从权重 shape 反推构造参数。"""
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731
        uw = shape("unit_embed.weight")
        if uw is None:
            raise ValueError("reflow_shallow ckpt missing unit_embed.weight")
        n_hidden, n_units = uw[0], uw[1]
        out_dims = int(shape("reflow_model.velocity_fn.input_projection.weight")[1])
        n_chans = int(shape("reflow_model.velocity_fn.input_projection.weight")[0])
        n_layers = 0
        while shape(f"reflow_model.velocity_fn.residual_layers.{n_layers}.net.0.weight") is not None:
            n_layers += 1
        sw = shape("spk_embed.weight")
        n_spk = int(sw[0]) if sw is not None else 1
        return dict(out_dims=out_dims, n_layers=n_layers, n_chans=n_chans,
                    n_units=n_units, n_spk=n_spk, n_hidden=n_hidden)
