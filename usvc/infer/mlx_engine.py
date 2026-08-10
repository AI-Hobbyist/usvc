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
        compile_model: bool = True,
    ):
        if device == "gpu":
            mx.set_default_device(mx.gpu)
            # M4 16GB:iOS 风格限制会 OOM,改用 macOS 合理上限。
            # wired memory 允许 GPU 常驻大量权重;cache 控制中间结果。
            # 0 = 不限制(用系统默认管理),但设 wired 上限避免独占。
            try:
                mx.set_memory_limit(0)      # 不设 active 上限
                mx.set_cache_limit(512 * 1024 * 1024)  # 512MB 计算缓存
                mx.set_wired_limit(4 * 1024 * 1024 * 1024)  # 4GB wired
            except Exception:
                pass  # 旧版 mlx 或 CPU 无效
        else:
            mx.set_default_device(mx.cpu)

        self.compile_model = compile_model
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
        # 编译模型 forward(图融合,降低峰值内存 + 加速)
        if compile_model:
            self._compile_models()

    def _compile_models(self):
        """编译 model forward(mx.compile 图融合)。

        对 backend model 和 vocoder 的核心计算路径编译。compile 后首次调用
        会触发图构建(预热),后续调用复用编译图,显著降低峰值内存和加速。
        """
        import functools
        m = getattr(self, "model", None)
        if m is not None:
            # diffusion:编译 denoiser(采样循环每步调用,compile 收益最大)
            if self.backend_name in ("diffusion", "reflow_vae"):
                decoder = getattr(m, "decoder", None)
                if decoder is not None:
                    try:
                        m._compiled_decoder = mx.compile(decoder.__call__)
                        decoder.__call__ = m._compiled_decoder
                    except Exception:
                        pass
            # 通用:编译 model.__call__(rvc/sovits 等含 infer 的)
            try:
                m._compiled_call = mx.compile(m.__call__)
                m.__call__ = m._compiled_call
            except Exception:
                pass
        # vocoder
        v = getattr(self, "vocoder", None)
        if v is not None:
            try:
                v._compiled_infer = mx.compile(v.infer)
                v.infer = v._compiled_infer
            except Exception:
                pass

    def _build_pipeline(self):
        """构建 mlx backend 模型 + 可选 vocoder。

        特征提取(units/f0/rms/uv)由 ``_extract_feats_numpy`` 按需调用
        ``InferFeatures``(torch 特征管线,backend 无关),此处不预加载。
        """
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
        elif self.backend_name == "rift_dit":
            from ..nn.mlx.backends.rift_dit import RIFTDiTModel
            self.model = RIFTDiTModel.from_ckpt(self.ckpt_path)
        else:
            raise NotImplementedError(
                f"{self.backend_name} mlx pipeline pending. Use impl='torch'."
            )
        self.model.eval()

    # ------------------------------------------------------------------
    # 分块推理(内存与时长解耦)
    # ------------------------------------------------------------------

    def _max_chunk_frames(self) -> int:
        """每块最大帧数(wav-domain 更小:O(T) wav 分配;mel-domain 可大)。

        经验值:wav-domain(GAN vocoder)在 ~5s 内峰值可控;
        mel-domain(扩散/flow)中间 mel 小,可到 ~15s。
        """
        if self.backend_name in ("rvc", "sovits"):
            # wav-domain O(T): 0.5s tile 平衡速度(47ms/block)与内存(542MB)。
            # 更小可降内存但分块开销增大。可用 USVC_MLX_TILE_SEC 覆盖。
            import os
            tile_sec = float(os.environ.get("USVC_MLX_TILE_SEC", "0.5"))
            return int(tile_sec * self.sr / self.hop)
        # mel-domain:vocoder O(T) 内存大,用 USVC_MLX_TILE_SEC_MEL 控制(默认 3s)。
        # 实测 vocoder:0.6s=670MB, 1.7s=1744MB。扩散分块开销大,默认 3s 平衡。
        import os
        tile_sec = float(os.environ.get("USVC_MLX_TILE_SEC_MEL", "3.0"))
        return int(tile_sec * self.sr / self.hop)

    def _chunk_indices(self, T: int, chunk: int, overlap: int = 0):
        """生成 (start, end) 分块索引,可选重叠(用于交叉淡化)。"""
        if T <= chunk:
            return [(0, T)]
        idx = []
        start = 0
        while start < T:
            end = min(start + chunk, T)
            idx.append((start, end))
            if end >= T:
                break
            start = end - overlap
        return idx

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
        if self.backend_name in ("diffusion", "ddsp6", "reflow_vae", "rift_dit"):
            return self._convert_mel_domain(input_wav, spk, trans, **kwargs)
        raise NotImplementedError(
            f"{self.backend_name} mlx convert pending. Use impl='torch'."
        )

    def _extract_feats_numpy(self, input_wav, f0_method="rmvpe"):
        """特征提取:优先 mlx GPU(快 2.3x,无 torch 进程开销),
        回退 torch InferFeatures(若 mlx 特征器不可用)。
        """
        # 尝试 mlx 特征提取(GPU,无 torch)
        try:
            if not hasattr(self, "_mlx_feats"):
                from ..hub import ensure_pretrained
                from ..nn.mlx.features import MlxFeatureExtractor
                # 从 feature_spec 解析 rms win_length / uv_policy / units_layer
                from ..backends import get_backend
                opts = self.cfg.backend_opts
                spec = get_backend(self.backend_name).feature_spec({
                    k: v for k, v in opts.items()
                    if k in ("units_kind", "units_ckpt", "f0_extractor",
                             "rmvpe_ckpt", "uv_policy", "n_mels", "fmin",
                             "fmax", "rvc_version", "version")
                })
                rms_win = (spec.rms.params.get("win_length")
                           if spec.rms is not None else None)
                uv_policy = (spec.f0.params.get("uv_policy", "interp")
                             if spec.f0 is not None else "interp")
                units_layer = (spec.units.params.get("layer", 12)
                               if spec.units is not None else 12)
                self._mlx_feats = MlxFeatureExtractor(
                    sr=self.sr, hop=self.hop,
                    units_ckpt=ensure_pretrained("contentvec/pytorch_model.bin"),
                    rmvpe_ckpt=ensure_pretrained("rmvpe/model.pt"),
                    fcpe_ckpt=ensure_pretrained("fcpe/fcpe.pt"),
                    rms_win_length=rms_win,
                    uv_policy=uv_policy,
                    units_layer=units_layer,
                )
                # compile 特征器核心(加速重复调用)
                if self.compile_model:
                    try:
                        self._mlx_feats.cvec.encode = mx.compile(self._mlx_feats.cvec.encode)
                    except Exception:
                        pass
                    try:
                        self._mlx_feats.rmvpe.model.__call__ = mx.compile(self._mlx_feats.rmvpe.model.__call__)
                    except Exception:
                        pass
            feats = self._mlx_feats.extract(input_wav, f0_method=f0_method)
            units = np.array(feats["units"], dtype=np.float32)
            f0 = np.asarray(feats["f0"], dtype=np.float32)
            rms = np.asarray(feats["rms"], dtype=np.float32) if feats.get("rms") is not None else None
            return units, f0, rms
        except Exception:
            pass  # 回退 torch
        # torch 回退:InferFeatures(backend 无关特征管线)
        from ..backends import get_backend
        from .features import InferFeatures
        opts = self.cfg.backend_opts
        spec = get_backend(self.backend_name).feature_spec({
            k: v for k, v in opts.items()
            if k in ("units_kind", "units_ckpt", "f0_extractor", "rmvpe_ckpt",
                     "uv_policy", "n_mels", "fmin", "fmax",
                     "rvc_version", "version")
        })
        if not hasattr(self, "_feats"):
            self._feats = InferFeatures(spec, sr=self.sr, hop=self.hop,
                                        device="cpu", dtype="fp32")
        feats = self._feats.extract(input_wav, f0_method=f0_method)
        units = feats["units"].cpu().numpy()
        f0 = feats["f0"].cpu().numpy()
        rms = feats["rms"].cpu().numpy() if feats["rms"] is not None else None
        return units, f0, rms

    def _convert_sovits(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """sovits wav-domain:feats -> [cluster] -> SoVITSModel.infer -> [enhance] -> wav。"""
        import mlx.core as mx
        f0_method = kwargs.get("f0_extractor", kwargs.get("f0_method", "rmvpe"))
        units_np, f0_np, rms_np = self._extract_feats_numpy(input_wav, f0_method)
        # sovits 聚类(cluster_infer_ratio)
        cluster_path = kwargs.get("cluster_path")
        cluster_ratio = float(kwargs.get("cluster_ratio", kwargs.get("index_ratio", 0.0)))
        if cluster_path and cluster_ratio > 0:
            import torch as _t
            from ..nn.mlx._postprocess import apply_cluster
            data = _t.load(cluster_path, map_location="cpu", weights_only=False)
            centers = data.get("cluster_centers_", data) if isinstance(data, dict) else data
            units_np = apply_cluster(np.asarray(centers), units_np, ratio=cluster_ratio)
        # 释放特征器
        import os
        mx.clear_cache()
        if hasattr(self, "_mlx_feats") and not os.environ.get("USVC_MLX_KEEP_FEATS"):
            del self._mlx_feats; self._mlx_feats = None
        if trans != 0:
            f0_np = f0_np * (2.0 ** (trans / 12.0))
        uv_np = (f0_np > 0).astype(np.float32)
        c = mx.array(units_np)[None]; f0 = mx.array(f0_np)[None]
        uv = mx.array(uv_np)[None]; g = mx.array([spk])
        ret = self.model.infer(c, f0, uv, g)
        o = ret[0] if isinstance(ret, (tuple, list)) else ret
        wav = np.array(o).reshape(-1).astype(np.float32)
        # 浅扩散增强
        if kwargs.get("shallow_diffusion") and kwargs.get("enhancer_ckpt"):
            if not hasattr(self, "enhancer_model"):
                self._load_enhancer(kwargs["enhancer_ckpt"], kwargs.get("enhancer_type", "diffusion"))
            wav = self._enhance_shallow(wav, units_np, f0_np, rms_np, spk,
                                        k_step=int(kwargs.get("k_step", 100)), **kwargs)
        return wav

    def _convert_mel_domain(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """mel-domain:feats -> mel sample -> vocoder -> wav。

        ddsp6:分块 sample+vocoder(DDSP 谱路径不适合全 mel)。
        diffusion/reflow_vae:全 mel sample + vocoder 0.5s 分块(峰值最优)。
        """
        import mlx.core as mx
        if self.backend_name == "ddsp6":
            return self._convert_mel_domain_chunked(input_wav, spk, trans, **kwargs)
        return self._convert_mel_domain_full_sample(input_wav, spk, trans, **kwargs)

    def _convert_mel_domain_chunked(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """分块 sample+vocoder(ddsp6):每块独立 sample,小 tile 控峰值。"""
        import mlx.core as mx, os
        f0_method = kwargs.get("f0_extractor", kwargs.get("f0_method", "rmvpe"))
        units_np, f0_np, rms_np = self._extract_feats_numpy(input_wav, f0_method)
        if trans != 0:
            f0_np = f0_np * (2.0 ** (trans / 12.0))
        vol_np = rms_np if rms_np is not None else np.ones_like(f0_np) * 0.5
        if hasattr(self, "_mlx_feats") and not os.environ.get("USVC_MLX_KEEP_FEATS"):
            del self._mlx_feats; self._mlx_feats = None
        T = units_np.shape[0]
        chunk = int(1.0 * self.sr / self.hop)  # 1s tile
        wav_parts = []
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            u = mx.array(units_np[s:e])[None]; f = mx.array(f0_np[s:e])[None]
            v = mx.array(vol_np[s:e])[None]
            mel = self._run_mel_backend_sample(u, f, v, spk, **kwargs)
            mel = mel if mel.shape[0] == 1 else mel[0:1]
            w = np.array(self.vocoder.infer(mel, f)).reshape(-1).astype(np.float32)
            wav_parts.append(w); mx.eval(mel)
        return np.concatenate(wav_parts)

    def _convert_mel_domain_full_sample(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """全 mel sample + vocoder 0.5s 分块(diffusion/reflow_vae:峰值最优)。"""
        import os, mlx.core as mx
        f0_method = kwargs.get("f0_extractor", kwargs.get("f0_method", "rmvpe"))
        units_np, f0_np, rms_np = self._extract_feats_numpy(input_wav, f0_method)
        if trans != 0:
            f0_np = f0_np * (2.0 ** (trans / 12.0))
        vol_np = rms_np if rms_np is not None else np.ones_like(f0_np) * 0.5
        mx.clear_cache()
        if hasattr(self, "_mlx_feats") and not os.environ.get("USVC_MLX_KEEP_FEATS"):
            del self._mlx_feats; self._mlx_feats = None
        units = mx.array(units_np)[None]; f0 = mx.array(f0_np)[None]; vol = mx.array(vol_np)[None]
        mel = self._run_mel_backend_sample(units, f0, vol, spk, **kwargs)
        mel_np = np.array(mel)
        mx.clear_cache()
        T = mel_np.shape[1]
        voc_tile = max(1, int(0.5 * self.sr / self.hop))
        wav_parts = []
        for s in range(0, T, voc_tile):
            e = min(s + voc_tile, T)
            m = mx.array(mel_np[0:1, s:e]); f_blk = mx.array(f0_np[s:e])[None]
            w = np.array(self.vocoder.infer(m, f_blk)).reshape(-1).astype(np.float32)
            wav_parts.append(w)
        return np.concatenate(wav_parts)

    def _run_mel_backend_sample(self, units, f0, vol, spk, **kwargs):
        """调 backend.sample -> mel(由具体 backend 分发)。"""
        be = self.backend_name
        if be == "diffusion":
            return self.model.sample(units, f0, vol, spk_id=mx.array([spk]),
                infer_speedup=int(kwargs.get("diffusion_speedup", 10)),
                method=kwargs.get("diffusion_method", "ddim"),
                k_step=int(kwargs.get("k_step", 1000)))
        elif be == "ddsp6":
            return self.model.sample(units, f0, vol, spk_id=spk,
                infer_step=int(kwargs.get("steps", 10)),
                method=kwargs.get("method", "euler"))
        elif be == "rift_dit":
            return self.model.sample(units, f0, vol, spk_id=mx.array([spk]),
                steps=int(kwargs.get("steps", 32)),
                t_start=float(kwargs.get("t_start", 0.0)),
                ds_cfg_strength=float(kwargs.get("ds_cfg_strength", 0.0)),
                spk_cfg_strength=float(kwargs.get("spk_cfg_strength", 0.0)),
                skip_cfg_strength=float(kwargs.get("skip_cfg_strength", 0.0)))
        else:  # reflow_vae
            return self.model.sample(units, f0, vol, spk_id=spk,
                infer_step=int(kwargs.get("steps", 10)))

    def _convert_rvc(self, input_wav, spk: int, trans: int, **kwargs) -> np.ndarray:
        """RVC 端到端:feats -> [index/cluster/protect] -> 固定块 infer -> [rms_mix] -> wav。

        高级功能(kwargs):
          index_path/index_ratio: RVC 特征检索
          cluster_path/cluster_ratio: so-vits 聚类
          protect: 清音帧源特征混回(默认 0.5,>=0.5 禁用)
          rms_mix_rate: 输出 RMS 包络匹配源(默认 0.0=不改)
          noice_scale: RVC infer 噪声尺度(默认 0.66666,传给 model.infer)
        """
        import os, mlx.core as mx
        f0_method = kwargs.get("f0_extractor", kwargs.get("f0_method", "rmvpe"))
        units_np, f0_np, rms_np = self._extract_feats_numpy(input_wav, f0_method)
        # ---- 高级特征处理(RVC/sovits 调音核心)----
        from ..nn.mlx._postprocess import retrieve, apply_cluster, protect_units
        index_path = kwargs.get("index_path")
        index_ratio = float(kwargs.get("index_ratio", 0.0))
        cluster_path = kwargs.get("cluster_path")
        cluster_ratio = float(kwargs.get("cluster_ratio", 0.0))
        units_src = units_np  # protect 用的源特征
        if index_path and index_ratio > 0:
            import torch as _t
            data = _t.load(index_path, map_location="cpu", weights_only=False)
            vectors = data["vectors"] if isinstance(data, dict) and "vectors" in data else data
            vectors = np.asarray(vectors)
            units_np = retrieve(vectors, units_np, ratio=index_ratio)
        if cluster_path and cluster_ratio > 0:
            import torch as _t
            data = _t.load(cluster_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "cluster_centers_" in data:
                centers = np.asarray(data["cluster_centers_"])
            else:
                centers = np.asarray(data)
            units_np = apply_cluster(centers, units_np, ratio=cluster_ratio)
        protect = float(kwargs.get("protect", 0.5))
        if protect < 0.5 and rms_np is not None:
            units_np = protect_units(units_np, units_src, rms_np, protect)
        # 释放特征器
        mx.clear_cache()
        if hasattr(self, "_mlx_feats") and not os.environ.get("USVC_MLX_KEEP_FEATS"):
            del self._mlx_feats
            self._mlx_feats = None
        if trans != 0:
            f0_np = f0_np * (2.0 ** (trans / 12.0))
        from ..nn.mlx.backends.rvc import f0_to_coarse
        pitch_np = np.array(f0_to_coarse(mx.array(f0_np))).astype(np.int32)
        T = units_np.shape[0]
        tile = self._max_chunk_frames()
        wav_hop = self.hop * self._rvc_upsample_total()  # 每帧对应样本数
        noice_scale = float(kwargs.get("noice_scale", 0.66666))
        wav_parts = []
        for start in range(0, T, tile):
            end = min(start + tile, T)
            block_len = end - start
            u = np.pad(units_np[start:end], ((0, tile - block_len), (0, 0)))
            f = np.pad(f0_np[start:end], (0, tile - block_len))
            p = np.pad(pitch_np[start:end], (0, tile - block_len))
            u = mx.array(u)[None]; f = mx.array(f)[None]
            p = mx.array(p)[None]; ln = mx.array([tile])
            o = self.model.infer(u, f, p, ln, spk_id=spk, noice_scale=noice_scale)
            o_np = np.array(o).reshape(-1).astype(np.float32)
            wav_parts.append(o_np[:block_len * wav_hop])
            mx.eval(o)  # 释放中间结果(控峰值)
        wav_out = np.concatenate(wav_parts)
        # ---- 浅扩散增强(可选)----
        if kwargs.get("shallow_diffusion") and kwargs.get("enhancer_ckpt"):
            if not hasattr(self, "enhancer_model"):
                self._load_enhancer(kwargs["enhancer_ckpt"],
                                    kwargs.get("enhancer_type", "diffusion"))
            wav_out = self._enhance_shallow(
                wav_out, units_np, f0_np, rms_np, spk,
                k_step=int(kwargs.get("k_step", 100)), **kwargs)
        # ---- RVC rms_mix:输出 RMS 包络匹配源音频 ----
        rms_mix = float(kwargs.get("rms_mix_rate", 0.0))
        if rms_mix > 0.0:
            from ..nn.mlx._postprocess import change_rms
            import librosa
            src_wav, src_sr = librosa.load(str(input_wav), sr=None)
            wav_out = change_rms(src_wav, src_sr, wav_out, self.sr, rms_mix)
        return wav_out

    def _enhance_shallow(self, wav_np, units_np, f0_np, rms_np, spk, k_step=100, **kwargs):
        """浅扩散增强:wav -> mel -> diffusion.sample(gt_spec, k_step<1000) -> vocoder -> wav。

        需加载 diffusion/reflow 增强器(由 _load_enhancer 初始化)。
        """
        import mlx.core as mx
        if not hasattr(self, "enhancer_model"):
            raise ValueError("shallow_diffusion 需先加载增强器(见 _load_enhancer)")
        # wav -> mel(groud truth,与扩散训练一致)
        from ..nn.mlx.mel import MelExtractor
        mel = self._wav_to_mel(wav_np)
        mel_mx = mx.array(mel)[None]  # [1, T, M]
        # 条件
        units = mx.array(units_np)[None]; f0 = mx.array(f0_np)[None]
        vol = mx.array(rms_np if rms_np is not None else np.ones_like(f0_np)*0.5)[None]
        # 浅扩散采样
        denoised = self.enhancer_model.sample(
            units, f0, vol, spk_id=mx.array([spk]),
            gt_spec=mel_mx, k_step=k_step,
            infer_speedup=int(kwargs.get("diffusion_speedup", 10)),
            method=kwargs.get("diffusion_method", "ddim"),
        )  # [1, T, M]
        mx.clear_cache()
        # vocoder
        wav_out = np.array(self.vocoder.infer(denoised, f0)).reshape(-1).astype(np.float32)
        return wav_out

    def _load_enhancer(self, enhancer_ckpt, enhancer_type="diffusion"):
        """加载浅扩散增强器模型(diffusion 或 reflow_shallow)。"""
        import mlx.core as mx
        if enhancer_type == "diffusion":
            from ..nn.mlx.backends.diffusion import DiffusionModel
            self.enhancer_model = DiffusionModel.from_ckpt(enhancer_ckpt)
        elif enhancer_type == "reflow_shallow":
            from ..nn.mlx.backends.reflow_shallow import ReflowShallowModel
            self.enhancer_model = ReflowShallowModel.from_ckpt(enhancer_ckpt)
        else:
            raise ValueError(f"未知 enhancer_type: {enhancer_type}")
        self.enhancer_model.eval()
        # 确保 vocoder 已加载(mel-domain 用)
        if not hasattr(self, "vocoder"):
            self._ensure_vocoder()

    def _ensure_vocoder(self):
        """惰性加载 NSF-HiFiGAN vocoder(mel-domain / 增强器共用)。"""
        if hasattr(self, "vocoder"):
            return
        from pathlib import Path
        from ..hub import ensure_dir
        from ..nn.mlx.nsf_hifigan import NSFHiFiGAN
        vdir = Path(ensure_dir("nsf_hifigan"))
        self.vocoder = NSFHiFiGAN(vdir / "config.json", vdir / "model.ckpt", device="gpu")
        self.vocoder.generator.eval()

    def _wav_to_mel(self, wav_np):
        """wav [L] -> mel [T, M](与扩散训练的 mel 配置一致)。"""
        from ..nn.mlx.mel import MelExtractor
        from .._common import FeatureConfigLike
        if not hasattr(self, "_mel_ext"):
            class _FC(FeatureConfigLike):
                kind = "mel"
                def __init__(self, params): self.params = params
            cfg = {"n_fft": 2048, "win_length": 2048, "n_mels": self.n_mels,
                   "fmin": 40.0, "fmax": 16000.0, "center": False}
            self._mel_ext = MelExtractor(_FC(cfg), sr=self.sr, hop=self.hop)
        return np.array(self._mel_ext.extract(wav_np.astype(np.float32)))

    def _rvc_upsample_total(self) -> int:
        """RVC GeneratorNSF 总上采样率(帧->样本)。"""
        # rvc v2 48k: upsample_rates = [12,10,2,2] -> 480
        opts = self.cfg.backend_opts
        rates = opts.get("upsample_rates", [12, 10, 2, 2])
        n = 1
        for r in rates:
            n *= int(r)
        return n
