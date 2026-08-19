"""MLX 版 reflow_vae backend(Bi_RectifiedFlow + VAE,ReFlow-VAE-SVC)。

镜像 ``usvc/backends/reflow_vae/model.py``:
  - ``Bi_RectifiedFlow``:双向(forward/reverse)rectified-flow ODE
    (euler / rk4 / heun / PECECE),与 ddsp6 的 RectifiedFlow 不同点:
    x_start(VAE mean)与 x_end(gt_spec)二选一作为 ODE 起点,dt 可负。
  - ``NaiveV2Diff`` / ``NaiveV2DiffLayer`` / ``ConformerConvModule``:
    back_bone='lynxnet' 的 denoiser。注意其 GLU 是 **sigmoid-gated**
    (torch nn.GLU),与 LYNXNet2 的 SwiGLU(silu)不同。
  - ``WaveNet``:back_bone='wavenet' denoiser —— 复用 ``.diffusion`` 的
    WaveNet(结构/键名一致,输出投影零初始化)。
  - ``Unit2Wav_VAE`` / ``ReflowVAEModel``:f0/unit/volume 嵌入 + 可选
    attention + VAE 高斯噪声 → Bi_RectifiedFlow 采样。

布局约定(见 MLX 映射手册):
  - 全程 NLC:units/f0/volume/cond [B, T, C],mel [B, T, M]。
  - ConformerConvModule 的 Transpose 占位无参数,保留 torch Sequential 键位。
  - 采样全程 fp32(ODE 迭代数值敏感)。
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

__all__ = [
    "Bi_RectifiedFlow",
    "NaiveV2DiffLayer",
    "NaiveV2Diff",
    "ConformerConvModule",
    "Transpose",
    "Unit2Wav_VAE",
    "ReflowVAEModel",
]


# ===========================================================================
# Bi_RectifiedFlow(reflow/reflow.py verbatim;NLC 版)
# ===========================================================================

class Bi_RectifiedFlow(nn.Module):
    """双向 rectified flow。velocity_fn(x [B,T,M], 1000*t [B], cond [B,T,H])。

    forward(infer, x_start, x_end, cond, t_start, t_end, infer_step, method):
      - infer=False:reflow_loss(l2_lognorm),x_0=x_start, x_1=norm(x_end)
      - infer=True:ODE 从 x_start(VAE mean,t_start)或 norm(x_end)(反向,t_end)
        出发积分;dt>0 → denorm_spec,dt<0 → 原始值(与 torch 一致)。
    """

    def __init__(self, velocity_fn, spec_min: float = -12, spec_max: float = 2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.spec_min = spec_min
        self.spec_max = spec_max

    # -- ODE 单步 / 训练 loss(euler/rk4 共享 kernel 见 _ops) -------------

    def sample_euler(self, x, t, dt, cond=None):
        return _ode_euler(self.velocity_fn, x, t, dt, cond)

    def sample_rk4(self, x, t, dt, cond=None):
        return _ode_rk4(self.velocity_fn, x, t, dt, cond)

    def sample_heun(self, x, t, dt, cond=None):
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        x_pred = x + k_1 * dt
        t_pred = t + dt
        k_2 = self.velocity_fn(x_pred, 1000 * t_pred, cond=cond)
        x = x + (k_1 + k_2) / 2 * dt
        t = t + dt
        return x, t

    def sample_PECECE(self, x, t, dt, cond=None):
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        x_pred1 = x + k_1 * dt
        t_pred1 = t + dt
        k_2 = self.velocity_fn(x_pred1, 1000 * t_pred1, cond=cond)
        x_corr1 = x + (k_1 + k_2) / 2 * dt
        k_3 = self.velocity_fn(x_corr1, 1000 * (t + dt), cond=cond)
        x_pred2 = x_corr1 + k_3 * dt
        k_4 = self.velocity_fn(x_pred2, 1000 * (t + 2 * dt), cond=cond)
        x = x + (k_3 + k_4) / 2 * dt
        t = t + dt
        return x, t

    # -- 训练 loss ---------------------------------------------------------

    def reflow_loss(self, x_1, x_0, t, cond=None, loss_type="l2_lognorm", mask=None):
        """训练 loss,layout [B,T,M] + 外部 x_0 见 _ops.reflow_loss。"""
        return _reflow_loss(self.velocity_fn, x_1, t, cond, loss_type, mask,
                            x_0=x_0, layout="btm")

    # -- 主入口 ------------------------------------------------------------

    def __call__(self, infer: bool = True, x_start=None, x_end=None, cond=None,
                 t_start: float = 0.0, t_end: float = 1.0, infer_step: int = 10,
                 method: str = "euler", mask=None):
        """x_start/x_end/cond: NLC [B, T, *]。infer=False → loss;True → mel。"""
        if cond is not None:
            cond = cond                                    # [B,T,H](NLC)
        b = x_start.shape[0] if x_start is not None else x_end.shape[0]
        if not infer:
            x_0 = x_start                                  # [B,T,M]
            x_1 = self.norm_spec(x_end)                    # [B,T,M]
            t = mx.random.uniform(shape=(b,))
            t = mx.clip(t, 1e-7, 1 - 1e-7)
            return self.reflow_loss(x_1, x_0, t, cond=cond, mask=mask)
        else:
            if t_start < 0.0:
                t_start = 0.0
            elif t_start > 1.0:
                t_start = 1.0
            if t_end < 0.0:
                t_end = 0.0
            elif t_end > 1.0:
                t_end = 1.0
            assert t_start < t_end, "Bi_RectifiedFlow: need t_start < t_end"

            if x_start is not None and x_end is None:
                x = x_start                                # [B,T,M]
                t = mx.full((b,), t_start)
                dt = (t_end - t_start) / infer_step
            elif x_start is None and x_end is not None:
                x = self.norm_spec(x_end)                  # [B,T,M]
                t = mx.full((b,), t_end)
                dt = -(t_end - t_start) / infer_step
            else:
                raise ValueError(
                    "Bi_RectifiedFlow infer: give exactly one of x_start/x_end"
                )

            for _ in range(infer_step):
                if method == "euler":
                    x, t = self.sample_euler(x, t, dt, cond=cond)
                elif method == "rk4":
                    x, t = self.sample_rk4(x, t, dt, cond=cond)
                elif method == "heun":
                    x, t = self.sample_heun(x, t, dt, cond=cond)
                elif method == "PECECE":
                    x, t = self.sample_PECECE(x, t, dt, cond=cond)
                else:
                    raise NotImplementedError(method)

            if dt > 0:
                return self.denorm_spec(x)                 # [B,T,M]
            else:
                return x

    def norm_spec(self, x):
        return _norm_spec(x, self.spec_min, self.spec_max, cast=True)

    def denorm_spec(self, x):
        return _denorm_spec(x, self.spec_min, self.spec_max)


# ===========================================================================
# ConformerConvModule(reflow/model_conformer_naive.py verbatim;GLU=sigmoid)
# ===========================================================================

class Transpose(nn.Module):
    """占位 Transpose(无参数)。NLC 布局下恒等,仅保留 torch Sequential 键位。"""

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims must be a tuple of two dimensions"
        self.dims = dims

    def __call__(self, x):
        return x


class ConformerConvModule(nn.Module):
    """Conformer 卷积模块(mode1,NLC 版)。

    net 键位与 torch Sequential 对齐:0=LayerNorm/Identity, 2=Conv1d,
    4=depthwise Conv1d, 6=Conv1d;1/3/5/7/8 为无参数占位(Transpose/GLU/
    SiLU/Dropout)。GLU 是 **sigmoid-gated**(torch nn.GLU),非 SwiGLU。
    """

    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0,
                 use_norm=False, conv_model_type="mode1"):
        super().__init__()
        inner_dim = dim * expansion_factor
        pad = dim and kernel_size // 2                     # calc_same_padding[0]
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
        for m in self.net:
            x = m(x)
        return x


# ===========================================================================
# NaiveV2Diff(reflow/naive_v2_diff.py verbatim;NLC 版)
# ===========================================================================

class NaiveV2DiffLayer(nn.Module):
    """NaiveV2Diff 残差层(NLC 版)。

    forward(x [B,T,dim], condition [B,T,dim_cond], diffusion_step [B,1,dim]):
      - non-wavenet:x + diffusion_step_proj(step) + condition_proj(cond),
        再过 conformer;返回 x + res_x
      - wavenet_like:再经 wavenet_like_proj + glu,返回 ((x+res)/√2, res)
    """

    def __init__(self, dim_model: int, dim_cond: int, num_heads: int = 4,
                 use_norm: bool = False, conv_only: bool = True,
                 conv_dropout: float = 0.0, atten_dropout: float = 0.1,
                 use_mlp: bool = True, expansion_factor: int = 2,
                 kernel_size: int = 31, wavenet_like: bool = False,
                 conv_model_type: str = "mode1"):
        super().__init__()
        self.conformer = ConformerConvModule(
            dim_model,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            dropout=conv_dropout,
            use_norm=use_norm,
            conv_model_type=conv_model_type,
        )
        if wavenet_like:
            self.wavenet_like_proj = nn.Conv1d(dim_model, 2 * dim_model, 1)
        else:
            self.wavenet_like_proj = None
        self.diffusion_step_projection = nn.Conv1d(dim_model, dim_model, 1)
        if dim_cond > 0:
            self.condition_projection = nn.Conv1d(dim_cond, dim_model, 1)
        else:
            self.condition_projection = nn.Identity()
        # conv_only=True 时无 attention(torch 默认 conv_only=True)
        self.attn = None
        self.norm = nn.LayerNorm(dim_model) if not conv_only else None

    def __call__(self, x, condition=None, diffusion_step=None):
        """x: [B,T,dim];condition: [B,T,dim_cond];diffusion_step: [B,1,dim]。"""
        res_x = x
        x = (
            x
            + self.diffusion_step_projection(diffusion_step)
            + self.condition_projection(condition)
        )
        if self.attn is not None:
            x = self.attn(self.norm(x))
        x = self.conformer(x)                              # [B,T,dim]
        if self.wavenet_like_proj is not None:
            x = self.wavenet_like_proj(x)                  # [B,T,2*dim]
            x = nn.glu(x, axis=-1)                         # [B,T,dim]
            return (x + res_x) / math.sqrt(2.0), res_x
        else:
            return x + res_x


class NaiveV2Diff(nn.Module):
    """NaiveV2Diff denoiser(NLC 版)。

    __call__(spec [B,T,M], diffusion_step [B], cond [B,T,condition_dim]) -> [B,T,M]。
    镜像 torch:use_mlp=False 时 conditioner_projection=Identity(layer 内
    condition_projection 仍把 condition_dim → dim)。
    """

    def __init__(
        self,
        mel_channels: int = 128,
        dim: int = 512,
        use_mlp: bool = True,
        mlp_factor: int = 4,
        condition_dim: int = 256,
        num_layers: int = 20,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        conv_only: bool = True,
        wavenet_like: bool = False,
        use_norm: bool = False,
        conv_model_type: str = "mode1",
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.1,
    ):
        super().__init__()
        self.wavenet_like = wavenet_like
        self.mask_cond_ratio = None

        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        # [DiffusionEmbedding, Linear, GELU, Linear](键名 0/1/2/3)
        self.diffusion_embedding = [
            DiffusionEmbedding(dim),
            nn.Linear(dim, dim * mlp_factor),
            nn.GELU(),
            nn.Linear(dim * mlp_factor, dim),
        ]
        if use_mlp and condition_dim > 0:
            self.conditioner_projection = [
                nn.Conv1d(condition_dim, dim * mlp_factor, 1),
                nn.GELU(),
                nn.Conv1d(dim * mlp_factor, dim, 1),
            ]
        else:
            self.conditioner_projection = None
        self.residual_layers = [
            NaiveV2DiffLayer(
                dim_model=dim,
                dim_cond=dim if use_mlp else condition_dim,
                num_heads=8,
                use_norm=use_norm,
                conv_only=conv_only,
                conv_dropout=conv_dropout,
                atten_dropout=atten_dropout,
                use_mlp=use_mlp,
                expansion_factor=expansion_factor,
                kernel_size=kernel_size,
                wavenet_like=wavenet_like,
                conv_model_type=conv_model_type,
            )
            for _ in range(num_layers)
        ]
        if use_mlp:
            _out = nn.Conv1d(dim * mlp_factor, mel_channels, kernel_size=1)
            _out.weight = mx.zeros_like(_out.weight)
            self.output_projection = [
                nn.Conv1d(dim, dim * mlp_factor, kernel_size=1),
                nn.GELU(),
                _out,
            ]
        else:
            self.output_projection = nn.Conv1d(dim, mel_channels, kernel_size=1)
            self.output_projection.weight = mx.zeros_like(self.output_projection.weight)

    def __call__(self, spec, diffusion_step, cond=None):
        """spec: [B,T,M];diffusion_step: [B];cond: [B,T,condition_dim]。"""
        x = self.input_projection(spec)                    # [B,T,dim]
        x = nn.gelu(x)
        step_emb = self.diffusion_embedding[0](diffusion_step)   # [B,dim]
        step_emb = self.diffusion_embedding[1](step_emb)
        step_emb = self.diffusion_embedding[2](step_emb)
        step_emb = self.diffusion_embedding[3](step_emb)
        diffusion_step = step_emb[:, None, :]              # [B,1,dim]

        if self.conditioner_projection is not None:
            condition = self.conditioner_projection[0](cond)
            condition = self.conditioner_projection[1](condition)
            condition = self.conditioner_projection[2](condition)  # [B,T,dim]
        else:
            condition = cond                               # [B,T,condition_dim]

        if self.wavenet_like:
            _sk = []
            for layer in self.residual_layers:
                x, sk = layer(x, cond, diffusion_step)
                _sk.append(sk)
            x = mx.sum(mx.stack(_sk), axis=0) / math.sqrt(len(self.residual_layers))
        else:
            for layer in self.residual_layers:
                x = layer(x, condition, diffusion_step)

        if isinstance(self.output_projection, list):
            x = self.output_projection[0](x)
            x = self.output_projection[1](x)
            x = self.output_projection[2](x)
        else:
            x = self.output_projection(x)
        return x                                           # [B,T,M]


# ===========================================================================
# Unit2Wav_VAE + ReflowVAEModel(reflow/vocoder.py verbatim;NLC 版)
# ===========================================================================

class Unit2Wav_VAE(nn.Module):
    """VAE 条件 + Bi_RectifiedFlow(镜像 reflow_vae/model.py::Unit2Wav_VAE)。

    __call__(units, f0, volume, spk_id, aug_shift, gt_spec, infer,
             infer_step, method, t_start) -> [B, T, out_dims]。
    """

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
            # [TransformerEncoderLayer, Linear(n_hidden, out_dims)]
            self.attention = [
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
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        else:
            self.spk_embed = None

        if back_bone is None or back_bone == "lynxnet":
            self.reflow_model = Bi_RectifiedFlow(
                NaiveV2Diff(
                    mel_channels=out_dims, dim=n_chans, num_layers=n_layers,
                    condition_dim=n_hidden, use_mlp=False,
                )
            )
        elif back_bone == "wavenet":
            self.reflow_model = Bi_RectifiedFlow(
                WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans,
                        n_hidden=n_hidden)
            )
        else:
            raise ValueError(f"Unknown Backbone: {back_bone}")

    def _fit_dims(self, f0, volume, spk_id):
        """usvc SVCBatch 形状 → 模型约定([B,T,1] / [B,1],0-based spk)。"""
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
        """units [B,T,n_unit];f0/volume [B,T] 或 [B,T,1];spk_id [B] 或 [B,1]。"""
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        f0 = f0.astype(mx.float32)
        # condition
        cond = self.f0_embed(mx.log(1 + f0 / 700))         # [B,T,n_hidden]
        if self.spk_embed is not None and spk_id is not None:
            cond = cond + self.spk_embed(spk_id.astype(mx.int32))  # [B,1,H] 广播
        if self.aug_shift_embed is not None and aug_shift is not None:
            cond = cond + self.aug_shift_embed(aug_shift.astype(mx.float32) / 5)

        # vae mean
        x = self.unit_embed(units) + self.volume_embed(volume.astype(mx.float32))
        if self.use_attention:
            x = self.attention[0](x, mask=None)
            x = self.attention[1](x)

        # vae noise
        x = x + mx.random.normal(shape=x.shape, dtype=mx.float32)

        x = self.reflow_model(
            infer=infer, x_start=x, x_end=gt_spec, cond=cond,
            infer_step=infer_step, method=method, t_start=t_start,
        )
        return x                                          # [B,T,out_dims]


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
        """训练 forward → (reflow_loss, {"reflow_loss": float})。"""
        mel = self(
            units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
            gt_spec=gt_spec, infer=False,
        )
        return mel, {"reflow_loss": float(mel.item())}

    def sample(self, units, f0, volume, spk_id, aug_shift=None, frame_len=None,
               infer_step=10, method="euler", t_start=None, use_tqdm=False):
        """推理:VAE latent → ODE 采样 → mel [B, T, out_dims]。"""
        f0, volume, spk_id = self._fit_dims(f0, volume, spk_id)
        if t_start is None:
            t_start = self.t_start
        return self(
            units, f0, volume, spk_id=spk_id, aug_shift=aug_shift,
            gt_spec=None, infer=True, infer_step=infer_step,
            method=method, t_start=t_start,
        )

    # -- ckpt 加载 ----------------------------------------------------------

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path, *, key_prefix: str | None = "model",
                  strict: bool = False, dtype=None):
        """从 torch reflow_vae ckpt 加载(推理)。

        键形如 ``model.unit_embed.weight`` / ``model.reflow_model.velocity_fn.*``。
        注意:use_attention=True 时 attention 的 TransformerEncoderLayer 内部
        键名 torch(``self_attn.in_proj_weight``)与 mlx(``query_proj`` 等)不同,
        strict=False 会跳过 attention 权重(结构可用,权重需键映射,见
        MLX_PITFALLS)。
        """
        from .._convert import load_torch_ckpt_to_mlx, strip_prefix

        sd = load_torch_ckpt_to_mlx(ckpt_path, dtype=dtype or mx.float32)
        if key_prefix:
            stripped = strip_prefix(sd, key_prefix + ".")
            if not stripped:  # 裸 sd 回退(顺序保留)
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
            raise ValueError("reflow_vae ckpt missing unit_embed.weight")
        # use_attention:unit_embed 输出 = n_hidden(attn) 还是 out_dims(直连)
        n_hidden = int(shape("f0_embed.weight")[0])
        n_units = int(uw[1])
        use_attention = uw[0] == n_hidden
        out_dims = int(uw[0]) if not use_attention else int(
            shape("attention.1.weight")[0]
        )
        sw = shape("spk_embed.weight")
        n_spk = int(sw[0]) if sw is not None else 1
        aew = shape("aug_shift_embed.weight")
        use_pitch_aug = aew is not None
        # denoiser:reflow_model.velocity_fn 有 residual_layers.0.dilated_conv → wavenet
        wn0 = shape("reflow_model.velocity_fn.residual_layers.0.dilated_conv.weight")
        if wn0 is not None:
            back_bone = "wavenet"
            in_w = shape("reflow_model.velocity_fn.input_projection.weight")
            n_chans = int(in_w[0])
            n = 0
            while shape(
                f"reflow_model.velocity_fn.residual_layers.{n}.dilated_conv.weight"
            ) is not None:
                n += 1
            n_layers = n
            return dict(n_units=n_units, n_spk=n_spk, out_dims=out_dims,
                        n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden,
                        back_bone=back_bone, use_attention=use_attention,
                        use_pitch_aug=use_pitch_aug)
        ip = shape("reflow_model.velocity_fn.input_projection.weight")  # (dim,1,M)
        if ip is None:
            raise ValueError(
                "reflow_vae ckpt has neither NaiveV2Diff nor WaveNet denoiser keys"
            )
        back_bone = "lynxnet"
        out_dims2 = int(ip[2])
        n_chans = int(ip[0])
        n = 0
        while shape(f"reflow_model.velocity_fn.residual_layers.{n}.conformer.net.2.weight") is not None:
            n += 1
        n_layers = n
        return dict(n_units=n_units, n_spk=n_spk, out_dims=out_dims2,
                    n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden,
                    back_bone=back_bone, use_attention=use_attention,
                    use_pitch_aug=use_pitch_aug)
