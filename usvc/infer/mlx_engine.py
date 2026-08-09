"""MLX 推理引擎(torch InferenceEngine 的 mlx 委托目标)。

当 ``USVC_BACKEND=mlx`` 或 ``InferenceEngine(impl='mlx')`` 时,
``usvc.infer.engine.InferenceEngine`` 把 ``convert()`` 委托给本类。

设计:torch 路径**零改动**。本类独立组合已验证的 mlx 组件
(ContentVec + Mel + NSF-HiFiGAN + backend generator),复用 torch 版
InferenceEngine 的 numpy 侧后处理(LUFS / RMS 匹配 / 分块拼接,这些与
backend 无关)。

当前覆盖(按实现进度):
  - rvc / sovits(wav-domain):generator 直接解码 wav,无需 vocoder
  - mel-domain(ddsp6/diffusion/rift_dit/reflow_vae):需 vocoder,待 backend 落地

未实现的特征检索(index/cluster)、浅扩散增强器暂走 torch fallback
(若 mlx 模块缺失,convert 会提示用户用 torch impl)。
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

__all__ = ["MlxInferenceEngine", "is_mlx_backend_ready"]


def is_mlx_backend_ready(backend_name: str) -> bool:
    """该 backend 是否已有 mlx 实现。

    随 backends/* 落地逐步开放。wav-domain 优先(rvc/sovits 免 vocoder)。
    """
    # rvc generator 已就绪(含 SineGen + f0_to_coarse),但完整推理链路
    # (TextEncoder + flow + InferFeatures)待接。返回 False 直到 mlx_engine
    # 的 convert() 能端到端跑通。
    ready = {
        # "rvc": True,  # generator ready, pipeline pending
        # "sovits": True,  # TODO
    }
    return ready.get(backend_name, False)


class MlxInferenceEngine:
    """MLX 推理引擎。

    由 ``InferenceEngine`` 在 impl=mlx 时构造。接口对齐 torch 版的
    ``convert()``(子集),返回 float32 numpy wav。

    Args:
        config_path: backend config yaml。
        ckpt_path: 模型 checkpoint(.ckpt/.pth,torch pickle)。
        device: mlx 设备('gpu' | 'cpu')。
        impl_check: 若 False,跳过 backend 就绪检查(测试用)。
    """

    def __init__(
        self,
        config_path: str | Path,
        ckpt_path: str | Path,
        device: str = "gpu",
        impl_check: bool = True,
    ):
        if device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            mx.set_default_device(mx.cpu)

        # 解析 config(复用 torch 版 TrainConfig,纯 yaml 解析无 torch 依赖)
        from ..train.config import TrainConfig
        self.cfg = TrainConfig.from_yaml(config_path)
        self.backend_name = self.cfg.backend
        self.ckpt_path = str(ckpt_path)

        if impl_check and not is_mlx_backend_ready(self.backend_name):
            raise NotImplementedError(
                f"mlx impl for backend '{self.backend_name}' not yet available. "
                f"Use impl='torch' (default) or USVC_BACKEND=torch. "
                f"mlx backends ready: {[k for k,v in {('rvc',False),('sovits',False)} if v[1]]}"
            )

        opts = self.cfg.backend_opts
        from .engine import _DEFAULT_SR_HOP
        self.sr = int(opts.get("sampling_rate", opts.get("sr", _DEFAULT_SR_HOP[self.backend_name][0])))
        self.hop = int(opts.get("hop_length", opts.get("block_size", opts.get("hop", _DEFAULT_SR_HOP[self.backend_name][1]))))
        self.n_mels = int(opts.get("n_mel_channels", opts.get("out_dims", opts.get("n_mels", 128))))

        # 构建 mlx 推理管线(features + backend module + 可选 vocoder)
        self._build_pipeline()

    def _build_pipeline(self):
        """构建 mlx 推理管线(子类或 backend dispatcher 覆盖)。"""
        from .features import InferFeatures
        # InferFeatures 目前是 torch 实现;mlx 特征提取直接用 contentvec/mlx.mel
        # 待 backends/* mlx 落地后,这里构造对应 mlx module
        raise NotImplementedError(
            f"mlx pipeline for {self.backend_name} under construction. "
            "Feature extractors (contentvec/mel) and vocoder are ready in usvc.nn.mlx."
        )

    def convert(self, input_wav, **kwargs) -> np.ndarray:
        """转换单个 wav → float32 numpy wav(接口对齐 torch InferenceEngine)。"""
        raise NotImplementedError(
            f"mlx convert for {self.backend_name} pending backend module. "
            "torch impl is fully functional (USVC_BACKEND=torch)."
        )
