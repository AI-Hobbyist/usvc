"""MLX 版 reflow_vae backend(Bi_RectifiedFlow + VAE,ReFlow-VAE-SVC),镜像
backends/reflow_vae/model.py。与 ddsp6 RectifiedFlow 不同:x_start(VAE mean)与
x_end(gt_spec)二选一作 ODE 起点,dt 可负。布局全程 NLC [B,T,C];采样全程
fp32(ODE 迭代数值敏感)。NaiveV2Diff 的 GLU 是 **sigmoid-gated**(torch nn.GLU),
与 LYNXNet2 的 SwiGLU(silu)不同 —— 不要混用。
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
from .diffusion import DiffusionEmbedding, SinusoidalPosEmb, WaveNet

__all__ = ["Bi_RectifiedFlow", "NaiveV2DiffLayer", "NaiveV2Diff",
           "ConformerConvModule", "Transpose", "Unit2Wav_VAE", "ReflowVAEModel"]


class Bi_RectifiedFlow(nn.Module):
    """velocity_fn(x [B,T,M], 1000*t [B], cond [B,T,H]);infer=False → loss
    (x_0=x_start, x_1=norm(x_end)),infer=True → 从 x_start(t_start) 或
    norm(x_end)(反向,t_end)积分;dt>0 → denorm_spec,dt<0 → 原始值。"""

    def __init__(self, velocity_fn, spec_min: float = -12, spec_max: float = 2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.spec_min = spec_min
        self.spec_max = spec_max

    def sample_euler(self, x, t, dt, cond=None):
        return _ode_euler(self.velocity_fn, x, t, dt, cond)

    def sample_rk4(self, x, t, dt, cond=None):
        return _ode_rk4(self.velocity_fn, x, t, dt, cond)

    def sample_heun(self, x, t, dt, cond=None):
        k1 = self.velocity_fn(x, 1000 * t, cond=cond)
        k2 = self.velocity_fn(x + k1 * dt, 1000 * (t + dt), cond=cond)
        x = x + (k1 + k2) / 2 * dt
        return x, t + dt

    def sample_PECECE(self, x, t, dt, cond=None):
        k1 = self.velocity_fn(x, 1000 * t, cond=cond)
        k2 = self.velocity_fn(x + k1 * dt, 1000 * (t + dt), cond=cond)
        x_corr1 = x + (k1 + k2) / 2 * dt
        k3 = self.velocity_fn(x_corr1, 1000 * (t + dt), cond=cond)
        k4 = self.velocity_fn(x_corr1 + k3 * dt, 1000 * (t + 2 * dt), cond=cond)
        x = x + (k3 + k4) / 2 * dt
        return x, t + dt

    def reflow_loss(self, x_1, x_0, t, cond=None, loss_type="l2_lognorm", mask=None):
        return _reflow_loss(self.velocity_fn, x_1, t, cond, loss_type, mask,
                            x_0=x_0, layout="btm")  # [B,T,M] 布局见 _ops.reflow_loss

    def __call__(self, infer: bool = True, x_start=None, x_end=None, cond=None,
                 t_start: float = 0.0, t_end: float = 1.0, infer_step: int = 10,
                 method: str = "euler", mask=None):  # x_start/x_end/cond NLC
        b = x_start.shape[0] if x_start is not None else x_end.shape[0]
        if not infer:
            t = mx.clip(mx.random.uniform(shape=(b,)), 1e-7, 1 - 1e-7)
            return self.reflow_loss(self.norm_spec(x_end), x_start, t, cond=cond, mask=mask)
        t_start, t_end = min(max(t_start, 0.0), 1.0), min(max(t_end, 0.0), 1.0)
        assert t_start < t_end, "Bi_RectifiedFlow: need t_start < t_end"
        if x_start is not None and x_end is None:
            x, t, dt = x_start, mx.full((b,), t_start), (t_end - t_start) / infer_step
        elif x_start is None and x_end is not None:
            x, t, dt = self.norm_spec(x_end), mx.full((b,), t_end), -(t_end - t_start) / infer_step
        else:
            raise ValueError("Bi_RectifiedFlow infer: give exactly one of x_start/x_end")
        step = {"euler": self.sample_euler, "rk4": self.sample_rk4,
                "heun": self.sample_heun, "PECECE": self.sample_PECECE}.get(method)
        if step is None: raise NotImplementedError(method)
        for _ in range(infer_step):
            x, t = step(x, t, dt, cond=cond)
        return self.denorm_spec(x) if dt > 0 else x

    def norm_spec(self, x):
        return _norm_spec(x, self.spec_min, self.spec_max, cast=True)

    def denorm_spec(self, x):
        return _denorm_spec(x, self.spec_min, self.spec_max)


class Transpose(nn.Module):
    """占位 Transpose(无参数):NLC 恒等,仅保留 torch Sequential 键位。"""

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims must be a tuple of two dimensions"
        self.dims = dims

    def __call__(self, x): return x


class ConformerConvModule(nn.Module):
    """mode1;net 键位与 torch Sequential 对齐(1/3/5/7/8 为无参数占位)。
    GLU 是 **sigmoid-gated**(torch nn.GLU),非 SwiGLU。"""

    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0,
                 use_norm=False, conv_model_type="mode1"):
        super().__init__()
        inner_dim = dim * expansion_factor
        pad = dim and kernel_size // 2  # calc_same_padding[0]
        if conv_model_type != "mode1":
            raise NotImplementedError(f"{conv_model_type} not implemented")
        self.net = [
            nn.LayerNorm(dim) if use_norm else nn.Identity(),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=pad, groups=inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout),
        ]

    def __call__(self, x):
        for m in self.net: x = m(x)
        return x


class NaiveV2DiffLayer(nn.Module):
    """残差层:non-wavenet → x+res;wavenet_like → glu,返回 ((x+res)/√2, res)。"""

    def __init__(self, dim_model: int, dim_cond: int, num_heads: int = 4,
                 use_norm: bool = False, conv_only: bool = True,
                 conv_dropout: float = 0.0, atten_dropout: float = 0.1,
                 use_mlp: bool = True, expansion_factor: int = 2,
                 kernel_size: int = 31, wavenet_like: bool = False,
                 conv_model_type: str = "mode1"):
        super().__init__()
        self.conformer = ConformerConvModule(
            dim_model, expansion_factor=expansion_factor, kernel_size=kernel_size,
            dropout=conv_dropout, use_norm=use_norm, conv_model_type=conv_model_type,
        )
        self.wavenet_like_proj = nn.Conv1d(dim_model, 2 * dim_model, 1) if wavenet_like else None
        self.diffusion_step_projection = nn.Conv1d(dim_model, dim_model, 1)
        self.condition_projection = nn.Conv1d(dim_cond, dim_model, 1) if dim_cond > 0 else nn.Identity()
        self.attn = None  # conv_only 模式无 attention(torch 默认;attn 分支保持死代码)
        self.norm = nn.LayerNorm(dim_model) if not conv_only else None

    def __call__(self, x, condition=None, diffusion_step=None):
        """x [B,T,dim];condition [B,T,dim_cond];diffusion_step [B,1,dim]。"""
        res_x = x
        x = x + self.diffusion_step_projection(diffusion_step) + self.condition_projection(condition)
        if self.attn is not None:
            x = self.attn(self.norm(x))
        x = self.conformer(x)
        if self.wavenet_like_proj is not None:
            x = nn.glu(self.wavenet_like_proj(x), axis=-1)
            return (x + res_x) / math.sqrt(2.0), res_x
        return x + res_x


class NaiveV2Diff(nn.Module):
    """denoiser:spec [B,T,M], step [B], cond [B,T,condition_dim] → [B,T,M]。
    use_mlp=False 时 conditioner_projection=Identity(layer 内仍 condition_dim→dim)。"""

    def __init__(self, mel_channels: int = 128, dim: int = 512, use_mlp: bool = True,
                 mlp_factor: int = 4, condition_dim: int = 256, num_layers: int = 20,
                 expansion_factor: int = 2, kernel_size: int = 31,
                 conv_only: bool = True, wavenet_like: bool = False,
                 use_norm: bool = False, conv_model_type: str = "mode1",
                 conv_dropout: float = 0.0, atten_dropout: float = 0.1):
        super().__init__()
        self.wavenet_like = wavenet_like
        self.mask_cond_ratio = None
        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = [DiffusionEmbedding(dim), nn.Linear(dim, dim * mlp_factor),
                                    nn.GELU(), nn.Linear(dim * mlp_factor, dim)]  # 键名 0/1/2/3
        self.conditioner_projection = [
            nn.Conv1d(condition_dim, dim * mlp_factor, 1), nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        ] if use_mlp and condition_dim > 0 else None
        self.residual_layers = [
            NaiveV2DiffLayer(
                dim_model=dim, dim_cond=dim if use_mlp else condition_dim,
                num_heads=8, use_norm=use_norm, conv_only=conv_only,
                conv_dropout=conv_dropout, atten_dropout=atten_dropout,
                use_mlp=use_mlp, expansion_factor=expansion_factor,
                kernel_size=kernel_size, wavenet_like=wavenet_like,
                conv_model_type=conv_model_type,
            )
            for _ in range(num_layers)
        ]
        if use_mlp:
            _out = nn.Conv1d(dim * mlp_factor, mel_channels, kernel_size=1)
            _out.weight = mx.zeros_like(_out.weight)
            self.output_projection = [
                nn.Conv1d(dim, dim * mlp_factor, kernel_size=1), nn.GELU(), _out,
            ]
        else:
            self.output_projection = nn.Conv1d(dim, mel_channels, kernel_size=1)
            self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond=None):
        x = nn.gelu(self.input_projection(spec))
        step_emb = diffusion_step
        for m in self.diffusion_embedding:
            step_emb = m(step_emb)
        step_emb = step_emb[:, None, :]  # [B,1,dim]
        if self.conditioner_projection is not None:
            condition = cond
            for m in self.conditioner_projection:
                condition = m(condition)
        else:
            condition = cond
        if self.wavenet_like:
            sk = []
            for layer in self.residual_layers:
                x, s = layer(x, cond, step_emb)  # ⚠ wavenet 分支用原始 cond
                sk.append(s)
            x = mx.sum(mx.stack(sk), axis=0) / math.sqrt(len(self.residual_layers))
        else:
            for layer in self.residual_layers:
                x = layer(x, condition, step_emb)
        proj = self.output_projection
        for m in (proj if isinstance(proj, list) else [proj]):
            x = m(x)
        return x


class Unit2Wav_VAE(nn.Module):
    """VAE 条件 + Bi_RectifiedFlow(镜像 reflow_vae/model.py::Unit2Wav_VAE)。"""

    def __init__(self, sampling_rate, block_size, win_length, n_unit, n_spk,
                 use_pitch_aug=False, out_dims=128, n_layers=6, n_chans=512,
                 n_hidden=256, back_bone="lynxnet", use_attention=False):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.f0_embed = nn.Linear(1, n_hidden)
        self.use_attention = use_attention
        if use_attention:
            self.unit_embed = nn.Linear(n_unit, n_hidden)
            self.volume_embed = nn.Linear(1, n_hidden)
            self.attention = [  # [TransformerEncoderLayer, Linear(n_hidden, out_dims)]
                nn.TransformerEncoderLayer(
                    dims=n_hidden, num_heads=8, mlp_dims=n_hidden * 4,
                    dropout=0.1, activation=nn.gelu, norm_first=False,
                ),
                nn.Linear(n_hidden, out_dims),
            ]
        else:
            self.unit_embed = nn.Linear(n_unit, out_dims)
            self.volume_embed = nn.Linear(1, out_dims)
            self.attention = None
        self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False) if use_pitch_aug else None
        self.n_spk = n_spk
        self.spk_embed = nn.Embedding(n_spk, n_hidden) if n_spk is not None and n_spk > 1 else None

        if back_bone is None or back_bone == "lynxnet":
            self.reflow_model = Bi_RectifiedFlow(
                NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers,
                            condition_dim=n_hidden, use_mlp=False)
            )
        elif back_bone == "wavenet":
            self.reflow_model = Bi_RectifiedFlow(
                WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans,
                        n_hidden=n_hidden)
            )
        else:
            raise ValueError(f"Unknown Backbone: {back_bone}")

    def _fit_dims(self, f0, volume, spk_id):
        """SVCBatch 形状 → 模型约定([B,T,1] / [B,1],0-based spk)。"""
        if f0 is not None and getattr(f0, "ndim", 0) == 2:
            f0 = f0[:, :, None]
        if volume is not None and getattr(volume, "ndim", 0) == 2:
            volume = volume[:, :, None]
        if spk_id is not None:
            if not hasattr(spk_id, "ndim"):
                spk_id = mx.array([int(spk_id)])
            if spk_id.ndim == 1:
                spk_id = spk_id[:, None]
        return f0, volume, spk_id

    def __call__(self, units, f0, volume, spk_id=None, aug_shift=None,
                 gt_spec=None, infer: bool = True, infer_step: int = 10,
                 method: str = "euler", t_start: float = 0.0):
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        f0 = f0.astype(mx.float32)
        cond = self.f0_embed(mx.log(1 + f0 / 700))  # [B,T,n_hidden]
        if self.spk_embed is not None and spk_id is not None:
            cond = cond + self.spk_embed(spk_id.astype(mx.int32))  # [B,1,H] 广播
        if self.aug_shift_embed is not None and aug_shift is not None:
            cond = cond + self.aug_shift_embed(aug_shift.astype(mx.float32) / 5)
        x = self.unit_embed(units) + self.volume_embed(volume.astype(mx.float32))
        if self.use_attention:
            x = self.attention[1](self.attention[0](x, mask=None))
        x = x + mx.random.normal(shape=x.shape, dtype=mx.float32)  # vae noise
        return self.reflow_model(
            infer=infer, x_start=x, x_end=gt_spec, cond=cond,
            infer_step=infer_step, method=method, t_start=t_start,
        )


class ReflowVAEModel(Unit2Wav_VAE):
    """usvc 包装:继承 Unit2Wav_VAE 保持 ckpt 键名,提供 sample。"""

    def __init__(self, n_units=768, n_spk=1, win_length=2048, sampling_rate=44100,
                 block_size=512, use_pitch_aug=False, out_dims=128, n_layers=6,
                 n_chans=512, n_hidden=256, back_bone="lynxnet",
                 use_attention=True, t_start=0.0):
        super().__init__(
            sampling_rate, block_size, win_length, n_units, n_spk,
            use_pitch_aug=use_pitch_aug, out_dims=out_dims, n_layers=n_layers,
            n_chans=n_chans, n_hidden=n_hidden, back_bone=back_bone,
            use_attention=use_attention,
        )
        self.t_start = t_start

    def forward_loss(self, units, f0, volume, spk_id, gt_spec, aug_shift=None,
                     frame_len=None):
        mel = self(units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
                   gt_spec=gt_spec, infer=False)
        return mel, {"reflow_loss": float(mel.item())}

    def sample(self, units, f0, volume, spk_id, aug_shift=None, frame_len=None,
               infer_step=10, method="euler", t_start=None, use_tqdm=False):
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        t_start = self.t_start if t_start is None else t_start
        return self(units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
                    gt_spec=None, infer=True, infer_step=infer_step,
                    method=method, t_start=t_start)

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, key_prefix: str | None = "model",
                  strict: bool = False, dtype=None):
        """torch ckpt → 推理模型。⚠ use_attention=True 时 attention 内键名
        torch(self_attn.in_proj_weight)≠mlx(query_proj 等),strict=False 跳过
        attention 权重(结构可用,键映射待 MLX_PITFALLS)。"""
        from .._convert import load_torch_ckpt_to_mlx, strip_prefix

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype or mx.float32)
        if key_prefix:
            stripped = strip_prefix(sd, key_prefix + ".")
            if not stripped:  # 裸 sd 回退(顺序保留)
                stripped = sd
            sd = stripped
        obj = cls(**cls._infer_config(sd))
        obj.load_weights(list(sd.items()), strict=strict)
        obj.eval()
        return obj

    @staticmethod
    def _infer_config(sd: dict) -> dict:
        shape = lambda k: sd[k].shape if k in sd else None  # noqa: E731
        uw = shape("unit_embed.weight")
        if uw is None:
            raise ValueError("reflow_vae ckpt missing unit_embed.weight")
        n_hidden = int(shape("f0_embed.weight")[0])
        n_units = int(uw[1])
        # use_attention:unit_embed 输出 = n_hidden 还是 out_dims(直连)
        use_attention = uw[0] == n_hidden
        out_dims = int(uw[0]) if not use_attention else int(shape("attention.1.weight")[0])
        sw = shape("spk_embed.weight")
        n_spk = int(sw[0]) if sw is not None else 1
        use_pitch_aug = shape("aug_shift_embed.weight") is not None
        wn0 = shape("reflow_model.velocity_fn.residual_layers.0.dilated_conv.weight")
        if wn0 is not None:  # wavenet denoiser
            n_chans = int(shape("reflow_model.velocity_fn.input_projection.weight")[0])
            n = 0
            while shape(f"reflow_model.velocity_fn.residual_layers.{n}.dilated_conv.weight") is not None:
                n += 1
            return dict(n_units=n_units, n_spk=n_spk, out_dims=out_dims,
                        n_layers=n, n_chans=n_chans, n_hidden=n_hidden,
                        back_bone="wavenet", use_attention=use_attention,
                        use_pitch_aug=use_pitch_aug)
        ip = shape("reflow_model.velocity_fn.input_projection.weight")  # (dim,1,M)
        if ip is None:
            raise ValueError("reflow_vae ckpt has neither NaiveV2Diff nor WaveNet denoiser keys")
        n = 0
        while shape(f"reflow_model.velocity_fn.residual_layers.{n}.conformer.net.2.weight") is not None:
            n += 1
        return dict(n_units=n_units, n_spk=n_spk, out_dims=int(ip[2]),
                    n_layers=n, n_chans=int(ip[0]), n_hidden=n_hidden,
                    back_bone="lynxnet", use_attention=use_attention,
                    use_pitch_aug=use_pitch_aug)