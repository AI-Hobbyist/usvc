"""MLX 推理引擎(torch InferenceEngine 的 mlx 委托目标)。

当 ``USVC_BACKEND=mlx`` 或 ``InferenceEngine(impl='mlx')`` 时,
``usvc.infer.engine.InferenceEngine`` 把 ``convert()`` 委托给本类。

设计:torch 路径**零改动**。本类独立组合已验证的 mlx 组件:
  - 特征:ContentVec(units)+ Mel + F0(RMVPE/FCPE)
  - backend:rvc/sovits(wav-domain)/ diffusion/ddsp6/reflow/rift_dit(mel-domain)
  - vocoder:NSF-HiFiGAN(mel-domain backends 用)

复用 torch 版 InferenceEngine 的 numpy 侧后处理(LUFS / RMS 匹配 / 分块拼接),
这些与 backend 无关。

当前覆盖(按实现进度):
  - rvc(wav-domain):RVCModel 完整链路(enc_p + flow + dec)
  - sovits(wav-domain):Generator + flow + TextEncoder
  - mel-domain:需 vocoder,待 backend model.sample 集成

特征检索(index/cluster)、浅扩散增强器暂走 torch fallback。
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

__all__ = ["MlxInferenceEngine", "is_mlx_backend_ready"]


def is_mlx_backend_ready(backend_name: str) -> bool:
    """该 backend 是否已有 mlx 实现(含完整推理链路)。

    wav-domain(rvc/sovits)优先就绪(免 vocoder)。mel-domain 需 vocoder +
    backend model.sample 集成,逐步开放。
    """
    # rvc/sovits 的 Generator + 特征提取已就绪,完整 RVCModel pipeline 待
    # mlx_engine.convert 端到端验证后开放。保守起见暂全 False,由 convert
    # 内部按 backend 分支处理(失败则提示用 torch)。
    return False


class MlxInferenceEngine:
    """MLX 推理引擎。

    由 ``InferenceEngine`` 在 impl=mlx 时构造。接口对齐 torch 版的
    ``convert()``(子集),返回 float32 numpy wav。

    Args:
        config_path: backend config yaml。
        ckpt_path: 模型 checkpoint(.ckpt/.pth,torch pickle)。
        vocoder_dir: mel-domain backends 的 NSF-HiFiGAN 目录。
        device: mlx 设备('gpu' | 'cpu')。
    """

    def __init__(
        self,
        config_path: str | Path,
        ckpt_path: str | Path,
        vocoder_dir: str | Path | None = None,
        device: str = "gpu",
    ):
        if device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            mx.set_default_device(mx.cpu)

        # 解析 config(纯 yaml,无 torch 依赖)
        from ..train.config import TrainConfig
        self.cfg = TrainConfig.from_yaml(config_path)
        self.backend_name = self.cfg.backend
        self.ckpt_path = str(ckpt_path)
        self.vocoder_dir = vocoder_dir

        opts = self.cfg.backend_opts
        from .engine import _DEFAULT_SR_HOP
        self.sr = int(opts.get("sampling_rate", opts.get("sr", _DEFAULT_SR_HOP[self.backend_name][0])))
        self.hop = int(opts.get("hop_length", opts.get("block_size", opts.get("hop", _DEFAULT_SR_HOP[self.backend_name][1]))))
        self.n_mels = int(opts.get("n_mel_channels", opts.get("out_dims", opts.get("n_mels", 128))))

        # 构建 mlx 推理管线
        self._build_pipeline()

    def _build_pipeline(self):
        """构建 mlx 特征提取器 + backend 模型 + 可选 vocoder。"""
        # 特征提取:ContentVec(units,16kHz)
        from ..hub import ensure_pretrained
        cvec_path = ensure_pretrained("contentvec/pytorch_model.bin")
        from ..nn.mlx.contentvec import contentvec_from_hf_ckpt
        self.cvec, _, _ = contentvec_from_hf_ckpt(cvec_path)
        self.cvec.eval()

        # backend 模型分发
        if self.backend_name == "rvc":
            self._build_rvc()
        elif self.backend_name == "sovits":
            self._build_sovits()
        elif self.backend_name in ("diffusion", "ddsp6", "reflow_vae", "rift_dit"):
            self._build_mel_domain()
        else:
            raise NotImplementedError(
                f"mlx pipeline for backend '{self.backend_name}' not yet wired. "
                f"Use impl='torch'."
            )

    def _build_rvc(self):
        """RVC wav-domain pipeline:enc_p + flow + dec(GeneratorNSF)。"""
        from ..nn.mlx.backends.rvc import RVCModel
        opts = self.cfg.backend_opts
        self.model = RVCModel(
            n_units=int(opts.get("units_dim", 768)),
            inter_channels=int(opts.get("inter_channels", 192)),
            hidden_channels=int(opts.get("hidden_channels", 192)),
            filter_channels=int(opts.get("filter_channels", 768)),
            n_heads=int(opts.get("n_heads", 2)),
            n_layers=int(opts.get("n_layers", 6)),
            kernel_size=int(opts.get("kernel_size", 3)),
            gin_channels=int(opts.get("gin_channels", 256)),
            sr=self.sr,
        )
        # TODO: from_ckpt 加载真实权重 + convtranspose_keys
        self.model.eval()

    def _build_sovits(self):
        """sovits wav-domain pipeline:Generator + flow + TextEncoder。"""
        from ..nn.mlx.backends.sovits import Generator
        # sovits 完整 SynthesizerTrn 集成待补;先持有 Generator
        # TODO: 完整 sovits model(enc_p + flow + dec)
        raise NotImplementedError(
            "sovits full mlx pipeline pending (Generator ready, flow/enc_p wiring TODO). "
            "Use impl='torch'."
        )

    def _build_mel_domain(self):
        """mel-domain backends:需 NSF-HiFiGAN vocoder + backend sample()。"""
        if self.vocoder_dir is None:
            from ..hub import ensure_dir
            self.vocoder_dir = ensure_dir("nsf_hifigan")
        from pathlib import Path
        vdir = Path(self.vocoder_dir)
        from ..nn.mlx.nsf_hifigan import NSFHiFiGAN
        self.vocoder = NSFHiFiGAN(
            vdir / "config.json", vdir / "model.ckpt", device="gpu"
        )
        # backend model.sample 集成待补
        raise NotImplementedError(
            f"{self.backend_name} mlx sample() pipeline pending (vocoder ready). "
            f"Use impl='torch'."
        )

    # ------------------------------------------------------------------
    # 特征提取(mlx)
    # ------------------------------------------------------------------

    def _extract_units(self, wav_16k: mx.array) -> mx.array:
        """wav 16kHz [T] -> units [T_frames, 768]。"""
        # ContentVec 输入 [B, T, 1]
        x = wav_16k[None, :, None]
        hidden = self.cvec.encode(x)  # [1, T', 768]
        return hidden[0]  # [T', 768]

    # ------------------------------------------------------------------
    # convert(接口对齐 torch InferenceEngine)
    # ------------------------------------------------------------------

    def convert(self, input_wav, spk: int = 0, trans: int = 0, **kwargs) -> np.ndarray:
        """转换单个 wav → float32 numpy wav。

        当前仅 rvc 链路实现完整(待 from_ckpt 接入后端到端可用)。
        """
        import librosa
        # 1. 读音频 + 重采样到 16kHz(units)
        wav_np, orig_sr = librosa.load(str(input_wav), sr=None)
        wav_16k = librosa.resample(wav_np, orig_sr=orig_sr, target_sr=16000,
                                   res_type="soxr_hq")
        # 2. 提取 units
        units = self._extract_units(mx.array(wav_16k.astype(np.float32)))
        # 3. backend-specific 推理
        if self.backend_name == "rvc":
            return self._convert_rvc(units, wav_np, orig_sr, spk, trans)
        raise NotImplementedError(
            f"{self.backend_name} mlx convert pending. Use impl='torch'."
        )

    def _convert_rvc(self, units, wav_np, orig_sr, spk: int, trans: int) -> np.ndarray:
        """RVC 端到端:units -> enc_p -> flow.reverse -> dec -> wav。"""
        # f0 提取(暂用占位;完整需 RMVPE/FCPE mlx,当前 RMVPE 数值待修)
        # TODO: 接入 mlx F0(RMVPE 修复后)或 torch fallback
        raise NotImplementedError(
            "RVC end-to-end convert pending: needs (a) mlx F0 extractor, "
            "(b) RVCModel.from_ckpt wired. Generator + flow + enc_p components "
            "are implemented; integration in progress. Use impl='torch'."
        )
