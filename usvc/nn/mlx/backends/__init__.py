"""MLX backend 实现。

按实现进度逐步开放。当前:wav-domain 的 rvc(GeneratorNSF)+ mel-domain 的
diffusion / reflow_vae / reflow_shallow(结构测试通过,推理接入见
infer/mlx_engine.py)。
"""
from __future__ import annotations

from .rvc import GeneratorNSF, RVCSineGen, RVCSourceModuleHnNSF, f0_to_coarse
from .sovits import (
    Encoder,
    Generator,
    MultiHeadAttention,
    ResidualCouplingBlock,
    ResidualCouplingLayer,
    SoVITSGenerator,
    SoVITSModel,
    TextEncoder,
    WN,
    f0_to_coarse as sovits_f0_to_coarse,
)
from .diffusion import (
    DiffusionEmbedding,
    GaussianDiffusion,
    SinusoidalPosEmb,
    WaveNet,
)
from .reflow_shallow import (
    LYNXNet2,
    LYNXNet2Block,
    RectifiedFlow,
    ReflowShallowModel,
    SwiGLU,
)
from .reflow_vae import (
    Bi_RectifiedFlow,
    ConformerConvModule,
    NaiveV2Diff,
    NaiveV2DiffLayer,
    ReflowVAEModel,
    Unit2Wav_VAE,
)

__all__ = [
    # rvc
    "GeneratorNSF",
    "RVCSineGen",
    "RVCSourceModuleHnNSF",
    "f0_to_coarse",
    # sovits
    "Encoder",
    "Generator",
    "MultiHeadAttention",
    "ResidualCouplingBlock",
    "ResidualCouplingLayer",
    "SoVITSGenerator",
    "SoVITSModel",
    "TextEncoder",
    "WN",
    "sovits_f0_to_coarse",
    # diffusion(shared:reflow_vae 复用 WaveNet)
    "DiffusionEmbedding",
    "GaussianDiffusion",
    "SinusoidalPosEmb",
    "WaveNet",
    # reflow_shallow
    "LYNXNet2",
    "LYNXNet2Block",
    "RectifiedFlow",
    "ReflowShallowModel",
    "SwiGLU",
    # reflow_vae
    "Bi_RectifiedFlow",
    "ConformerConvModule",
    "NaiveV2Diff",
    "NaiveV2DiffLayer",
    "ReflowVAEModel",
    "Unit2Wav_VAE",
]
