"""MLX reflow_shallow backend(mel-domain 浅扩散增强器,镜像 reflow_shallow/model.py)。

NLC 全程 spec [B,T,M] / cond [B,T,H](torch 4-D [B,1,M,T] 的 squeeze 恒等省去);
LayerNorm/Linear 天然 NLC,Conv1d(dim,dim,k,groups=dim) 深度分离卷积直接等价
torch Transpose+Conv1d(权重转置见 _convert 启发式);采样全程 fp32(ODE 敏感)。
LYNXNet2/SwiGLU/Transpose/`_Noop` 键名与 ddsp6 一致(独立实现,勿合并)。
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .._ops import (
    denorm_spec as _denorm_spec,
    norm_spec as _norm_spec,
    ode_euler as _ode_euler,
    ode_rk4 as _ode_rk4,
    reflow_loss as _reflow_loss,
)

__all__ = [
    "SwiGLU",
    "SinusoidalPosEmb",
    "Transpose",
    "LYNXNet2Block",
    "LYNXNet2",
    "RectifiedFlow",
    "ReflowShallowModel",
]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.half_dim = half_dim
        emb = math.log(10000.0) / (half_dim - 1)
        self.freqs = mx.exp(mx.arange(half_dim) * -emb)  # [half_dim]

    def __call__(self, x):
        x = x.astype(mx.float32)
        x = x[:, None] * self.freqs[None, :]
        return mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        out, gate = mx.split(x, 2, axis=self.dim)  # torch split(x, dim//2, dim)
        return out * nn.silu(gate)


class Transpose(nn.Module):
    """零参数占位:NLC 下布局切换恒等,仅对齐 torch Sequential 键位
    (net.0=LayerNorm, net.1=Transpose, net.2=Conv1d, ...)。"""

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims must be a tuple of two dimensions"
        self.dims = dims

    def __call__(self, x):
        return x


class LYNXNet2Block(nn.Module):
    """net 列表键位与 torch Sequential 一致;Transpose/SwiGLU/`_Noop` 无参数占位。"""

    def __init__(self, dim: int, expansion_factor: int, kernel_size: int = 31,
                 dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = [
            nn.LayerNorm(dim), Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2), SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2), SwiGLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout) if float(dropout) > 0.0 else _Noop(),
        ]

    def __call__(self, x):
        y = x
        for m in self.net:
            y = m(y)
        return x + y


class _Noop(nn.Module):
    def __call__(self, x):
        return x


class LYNXNet2(nn.Module):
    def __init__(self, in_dims: int, dim_cond: int, n_layers: int = 6,
                 n_chans: int = 512, n_dilates: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)
        self.diffusion_embedding = [  # 键名与 torch Sequential 一致
            SinusoidalPosEmb(n_chans), nn.Linear(n_chans, n_chans * 4),
            nn.GELU(), nn.Linear(n_chans * 4, n_chans),
        ]
        self.residual_layers = [
            LYNXNet2Block(dim=n_chans, expansion_factor=n_dilates,
                          kernel_size=31, dropout=dropout)
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
        # torch nn.init.zeros_(output_projection.weight)
        self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond):
        x = self.input_projection(spec) + self.conditioner_projection(cond)
        step_emb = diffusion_step
        for m in self.diffusion_embedding:
            step_emb = m(step_emb)  # [B] → [B, n_chans]
        x = x + step_emb[:, None, :]
        for layer in self.residual_layers:
            x = layer(x)
        return self.output_projection(self.norm(x))


class RectifiedFlow(nn.Module):
    """ODE 采样 + 训练 loss(euler/rk4/reflow_loss 共享 kernel 见 _ops)。"""

    def __init__(self, velocity_fn, out_dims: int = 128,
                 spec_min: float = -12, spec_max: float = 2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max

    def sample_euler(self, x, t, dt, cond):
        return _ode_euler(self.velocity_fn, x, t, dt, cond)

    def sample_rk4(self, x, t, dt, cond):
        return _ode_rk4(self.velocity_fn, x, t, dt, cond)

    def reflow_loss(self, x_1, t, cond, loss_type="l2_lognorm", mask=None):
        """layout [B,T,M](与 ddsp6 仅布局/分母不同,见 _ops.reflow_loss)。"""
        return _reflow_loss(self.velocity_fn, x_1, t, cond, loss_type, mask,
                            layout="btm")

    def __call__(self, condition, gt_spec=None, infer: bool = True,
                 infer_step: int = 10, method: str = "euler",
                 t_start: float = 0.0, mask=None):
        """infer=True → [B,T,M] denormalized mel;False → 标量 loss。"""
        b = condition.shape[0]
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec)
            t = mx.clip(t_start + (1.0 - t_start) * mx.random.uniform(shape=(b,)), 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, t, cond=condition, mask=mask)
        shape = (b, condition.shape[1], self.out_dims)
        if gt_spec is None:
            x = mx.random.normal(shape=shape, dtype=mx.float32)
            t = mx.zeros((b,))
            dt = 1.0 / infer_step
        else:
            x = (t_start * self.norm_spec(gt_spec)
                 + (1 - t_start) * mx.random.normal(shape=shape, dtype=mx.float32))
            t = mx.full((b,), t_start)
            dt = (1.0 - t_start) / infer_step
        for _ in range(infer_step):
            if method == "euler":
                x, t = self.sample_euler(x, t, dt, condition)
            elif method == "rk4":
                x, t = self.sample_rk4(x, t, dt, condition)
            else:
                raise NotImplementedError(method)
        # clamp 到 [-1,1] 再 denorm(与 torch 一致,防 vocoder 过冲)
        return self.denorm_spec(mx.clip(x, -1.0, 1.0))

    def norm_spec(self, x):
        return _norm_spec(x, self.spec_min, self.spec_max, cast=True)

    def denorm_spec(self, x):
        return _denorm_spec(x, self.spec_min, self.spec_max)


class ReflowShallowModel(nn.Module):
    """mel-conditioned 浅扩散增强器:condition = norm(mel) + 嵌入。"""

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

        self.unit_embed = nn.Linear(n_units, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.spk_embed = nn.Embedding(n_spk, n_hidden) if (n_spk is not None and n_spk > 1) else None

        velocity_fn = LYNXNet2(in_dims=out_dims, dim_cond=out_dims + n_hidden,
                               n_layers=n_layers, n_chans=n_chans)
        self.reflow_model = RectifiedFlow(velocity_fn, out_dims=out_dims,
                                          spec_min=-12, spec_max=2)

    def embed_condition(self, units, f0, volume, mel, spk_id=None):
        """→ [B, T, out_dims + n_hidden]:norm(mel) + embeddings。"""
        if f0.ndim == 2:
            f0 = f0[:, :, None]
        emb = self.unit_embed(units) + self.f0_embed(mx.log(1 + f0.astype(mx.float32) / 700))
        if volume is not None:
            if volume.ndim == 2:
                volume = volume[:, :, None]
            emb = emb + self.volume_embed(volume.astype(mx.float32))
        if self.spk_embed is not None and spk_id is not None:
            spk = spk_id if spk_id.ndim > 1 else spk_id[:, None]
            emb = emb + self.spk_embed(spk.astype(mx.int32))
        return mx.concatenate([self.reflow_model.norm_spec(mel), emb], axis=-1)

    @staticmethod
    def _sample_degrade_params():
        import random
        return {"bias": random.uniform(0.4, 0.8),
                "noise_amp": random.uniform(0.35, 0.75), "smooth_win": 5}

    @staticmethod
    def _degrade(mel, bias, noise_amp, smooth_win):
        """模拟 RVC 伪影:近平坦偏置 + 时域平滑噪声(exp(-bin/62) 衰减)。
        输出 stop_gradient(训练时条件不用梯度)。grouped conv 用
        as_strided 窗口 + einsum 向量化(无 python 循环)。"""
        B, T, M = mel.shape
        dtype = mel.dtype

        idx_m = mx.arange(M, dtype=mx.float32)
        bias_curve = mx.where(idx_m < 16, mx.full((M,), bias * 0.4, dtype=dtype),
                              mx.full((M,), bias, dtype=dtype))

        taper = mx.exp(-idx_m / 62.0)
        noise_raw = mx.random.normal(shape=(B, M, T), dtype=mx.float32) * taper[None, :, None]
        win = max(3, int(smooth_win))
        pad = win // 2
        nw = mx.pad(noise_raw.transpose(0, 2, 1), [(0, 0), (pad, pad), (0, 0)])  # [B, T+2pad, M]
        Bc, Tc, Mc = nw.shape
        w = mx.as_strided(nw, shape=(Bc, T, win, Mc),
                          strides=(Tc * Mc, Mc, Mc, 1))
        kernel = mx.full((win, M), 1.0 / win, dtype=mx.float32)
        noise = mx.einsum("btwm,wm->btm", w, kernel)
        noise = noise * (win ** 0.5) * noise_amp  # 撤销移动平均收缩

        return mx.stop_gradient(mel + noise + bias_curve[None, None, :])

    def forward_loss(self, units, f0, volume, real_mel, spk_id=None,
                     t_start: float = 0.0, frame_len=None):
        """degrade(clean_mel) 作条件,clean mel 作目标(镜像 model.py:153)。"""
        dp = self._sample_degrade_params()
        degraded_mel = self._degrade(real_mel, **dp)
        cond = self.embed_condition(units, f0, volume, degraded_mel, spk_id)
        from ....train.mlx.objective import lens_to_mask
        mask = (lens_to_mask(frame_len, length=real_mel.shape[1])
                if frame_len is not None else None)
        return self.reflow_model(condition=cond, gt_spec=real_mel, infer=False,
                                 t_start=t_start, mask=mask)

    def sample(self, units, f0, volume, sovits_mel, t_start: float = 0.5,
               infer_step: int = 10, method: str = "euler", spk_id=None):
        """浅扩散 sovits_mel → 干净 mel [B, T, out_dims](dB 域)。"""
        cond = self.embed_condition(units, f0, volume, sovits_mel, spk_id)
        return self.reflow_model(condition=cond, gt_spec=sovits_mel, infer=True,
                                 infer_step=infer_step, method=method, t_start=t_start)

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, key_prefix: str | None = "model",
                  strict: bool = False, dtype=None):
        """键形如 model.reflow_model.velocity_fn.*;支持 module./无前缀(回退保留)。"""
        from .._convert import load_torch_ckpt_to_mlx, strip_prefix

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype or mx.float32)
        if key_prefix:
            stripped = strip_prefix(sd, key_prefix + ".")
            sd = stripped if stripped else sd  # 裸 sd 回退(顺序保留)
        cfg = cls._infer_config(sd)
        obj = cls(**cfg)
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731
        uw = shape("unit_embed.weight")
        if uw is None:
            raise ValueError("reflow_shallow ckpt missing unit_embed.weight")
        n_hidden, n_units = uw[0], uw[1]
        iw = shape("reflow_model.velocity_fn.input_projection.weight")
        n_layers = 0
        while shape(f"reflow_model.velocity_fn.residual_layers.{n_layers}.net.0.weight") is not None:
            n_layers += 1
        sw = shape("spk_embed.weight")
        return dict(out_dims=int(iw[1]), n_layers=n_layers, n_chans=int(iw[0]),
                    n_units=n_units, n_spk=int(sw[0]) if sw is not None else 1,
                    n_hidden=n_hidden)