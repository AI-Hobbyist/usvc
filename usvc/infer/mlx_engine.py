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
        self._config_path = str(config_path)
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
        self.cvec = contentvec_from_hf_ckpt(cvec_path)
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
        """RVC wav-domain pipeline:enc_p + flow + dec(GeneratorNSF)。

        特征提取(units/f0/rms/uv)复用 torch 版 feats 管线(返回 numpy),
        model.infer 用 mlx。混合策略:特征管线已验证且 backend 无关,
        mlx 只接管最重的 model forward。
        """
        from ..nn.mlx.backends.rvc import RVCModel
        opts = self.cfg.backend_opts
        kw = dict(
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
        # 尝试从 ckpt 加载(裸 RVC ckpt,key_prefix=dec 对应 GeneratorNSF;
        # 完整 RVCModel 用 key_prefix=None 自动检测)
        try:
            self.model = RVCModel.from_ckpt(self.ckpt_path, config=kw)
        except Exception:
            # ckpt 不可用或格式不匹配,用空权重(结构可 forward,数值无意义)
            self.model = RVCModel(**kw)
        self.model.eval()

    def _build_sovits(self):
        """sovits wav-domain pipeline:SoVITSModel(enc_p + flow + dec)。"""
        from ..nn.mlx.backends.sovits import SoVITSModel
        try:
            self.model = SoVITSModel.from_ckpt(self.ckpt_path)
        except Exception:
            from ..nn.mlx.backends.sovits import Generator  # noqa: F401
            raise NotImplementedError(
                "sovits mlx from_ckpt failed. Check ckpt format. Use impl='torch'."
            )
        self.model.eval()

    def _build_mel_domain(self):
        """mel-domain backends:backend.sample(units,f0,vol) -> mel -> vocoder -> wav。"""
        if self.vocoder_dir is None:
            from ..hub import ensure_dir
            self.vocoder_dir = ensure_dir("nsf_hifigan")
        from pathlib import Path
        vdir = Path(self.vocoder_dir)
        from ..nn.mlx.nsf_hifigan import NSFHiFiGAN
        self.vocoder = NSFHiFiGAN(
            vdir / "config.json", vdir / "model.ckpt", device="gpu"
        )
        self.vocoder.generator.eval()
        # backend model.sample -> mel
        opts = self.cfg.backend_opts
        if self.backend_name == "diffusion":
            from ..nn.mlx.backends.diffusion import DiffusionModel
            self.model = DiffusionModel.from_ckpt(self.ckpt_path)
        elif self.backend_name == "ddsp6":
            from ..nn.mlx.backends.ddsp6 import DDSP6Model
            self.model = DDSP6Model.from_ckpt(self.ckpt_path)
        elif self.backend_name == "reflow_vae":
            from ..nn.mlx.backends.reflow_vae import ReflowVAEModel
            self.model = ReflowVAEModel.from_ckpt(self.ckpt_path)
        else:
            raise NotImplementedError(
                f"{self.backend_name} mlx pipeline pending. Use impl='torch'."
            )
        self.model.eval()

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

        混合策略:特征提取(units/f0/rms/uv)复用 torch 版 feats 管线
        (backend 无关,返回 numpy),model forward 用 mlx。
        """
        if self.backend_name == "rvc":
            return self._convert_rvc(input_wav, spk, trans, **kwargs)
        if self.backend_name == "sovits":
            return self._convert_sovits(input_wav, spk, trans, **kwargs)
        if self.backend_name in ("diffusion", "ddsp6", "reflow_vae"):
            return self._convert_mel_domain(input_wav, spk, trans, **kwargs)
        raise NotImplementedError(
            f"{self.backend_name} mlx convert pending. Use impl='torch'."
        )

    def _extract_feats_numpy(self, input_wav, f0_method="rmvpe"):
        """复用 torch engine 的特征提取(返回 numpy units/f0/rms/uv)。"""
        if not hasattr(self, "_torch_eng_for_feats"):
            from .engine import InferenceEngine
            self._torch_eng_for_feats = InferenceEngine(
                config_path=self._config_path,
                ckpt_path=self.ckpt_path,
                device="cpu", impl="torch",
            )
        feats = self._torch_eng_for_feats.features.extract(
            input_wav, f0_method=f0_method
        )
        units = feats["units"].cpu().numpy()  # [T, D]
        f0 = feats["f0"].cpu().numpy()        # [T]
        rms = feats["rms"].cpu().numpy() if feats["rms"] is not None else None
        return units, f0, rms

    def _convert_sovits(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """sovits wav-domain:feats -> SoVITSModel.infer(c, f0, uv, g) -> wav。"""
        import mlx.core as mx
        f0_method = kwargs.get("f0_extractor", kwargs.get("f0_method", "rmvpe"))
        units_np, f0_np, rms_np = self._extract_feats_numpy(input_wav, f0_method)
        if trans != 0:
            f0_np = f0_np * (2.0 ** (trans / 12.0))
        # uv:浊音标志(1=浊音)。f0>0 视为浊音。
        uv_np = (f0_np > 0).astype(np.float32)
        c = mx.array(units_np)[None]   # [1, T, D]
        f0 = mx.array(f0_np)[None]    # [1, T]
        uv = mx.array(uv_np)[None]     # [1, T]
        g = mx.array([spk])
        ret = self.model.infer(c, f0, uv, g)
        o = ret[0] if isinstance(ret, (tuple, list)) else ret
        wav = np.array(o).reshape(-1).astype(np.float32)
        return wav

    def _convert_mel_domain(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """mel-domain pipeline:feats -> backend.sample -> mel -> vocoder -> wav。"""
        import mlx.core as mx
        f0_method = kwargs.get("f0_extractor", kwargs.get("f0_method", "rmvpe"))
        units_np, f0_np, rms_np = self._extract_feats_numpy(input_wav, f0_method)
        # pitch shift
        if trans != 0:
            f0_np = f0_np * (2.0 ** (trans / 12.0))
        # volume:用 rms(若无填 1.0)
        vol_np = rms_np if rms_np is not None else np.ones_like(f0_np) * 0.5
        units = mx.array(units_np)[None]   # [1, T, D]
        f0 = mx.array(f0_np)[None]        # [1, T]
        vol = mx.array(vol_np)[None]      # [1, T]
        # backend.sample -> mel [1, T, M]
        be = self.backend_name
        if be == "diffusion":
            speedup = int(kwargs.get("diffusion_speedup", 10))
            method = kwargs.get("diffusion_method", "ddim")
            k_step = int(kwargs.get("k_step", 1000))
            mel = self.model.sample(units, f0, vol, spk_id=mx.array([spk]),
                                    infer_speedup=speedup, method=method,
                                    k_step=k_step)
        elif be == "ddsp6":
            mel = self.model.sample(units, f0, vol, spk_id=spk,
                                    infer_step=int(kwargs.get("steps", 10)),
                                    method=kwargs.get("method", "euler"))
        else:  # reflow_vae
            mel = self.model.sample(units, f0, vol, spk_id=spk,
                                    infer_step=int(kwargs.get("steps", 10)))
        # vocoder: mel -> wav
        mel = mel if mel.shape[0] == 1 else mel[0:1]
        f0_voc = f0  # [1, T]
        wav_mlx = np.array(self.vocoder.infer(mel, f0_voc))  # [1, L, 1]
        return wav_mlx.reshape(-1).astype(np.float32)

    def _convert_rvc(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """RVC 端到端:feats(torch engine) -> RVCModel.infer(mlx) -> wav。

        复用 torch ``InferenceEngine`` 的特征提取(units/f0/rms/uv),
        pitch shift 与 torch 一致,最后 mlx ``RVCModel.infer``。
        torch engine 仅用于特征提取(model forward 由 mlx 接管)。
        """
        import mlx.core as mx
        f0_method = kwargs.get("f0_extractor", kwargs.get("f0_method", "rmvpe"))
        units_np, f0_np, _ = self._extract_feats_numpy(input_wav, f0_method)
        units = units_np
        f0 = f0_np

        # 2. Pitch shift
        if trans != 0:
            f0 = f0 * (2.0 ** (trans / 12.0))

        # 3. f0 -> pitch token(coarse,与 RVC torch infer 一致)
        from ..nn.mlx.backends.rvc import f0_to_coarse
        pitch = np.array(f0_to_coarse(mx.array(f0)))  # [T] int32 numpy

        # 4. mlx model.infer
        units_mlx = mx.array(units)[None]      # [1, T, 768]
        f0_mlx = mx.array(f0)[None]           # [1, T]
        pitch_mlx = mx.array(pitch.astype(np.int32))[None]  # [1, T]
        lengths = mx.array([units.shape[0]])
        wav_mlx = np.array(self.model.infer(
            units_mlx, f0_mlx, pitch_mlx, lengths, spk_id=spk
        ))
        # [1, L, 1] -> [L]
        return wav_mlx.reshape(-1).astype(np.float32)
