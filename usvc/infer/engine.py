"""Offline inference: wav -> features -> backend -> wav (cli: usvc-infer).

mel-domain (rift_dit/ddsp6/diffusion/reflow_vae) 出 mel, 共享 NSF-HiFiGAN 声码器;
wav-domain (sovits/rvc) 生成器自带解码, 无外部 vocoder。
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext as _nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from ..backends import get_backend
from ..backends.sovits.dataset import interp_unvoiced_f0
from ..io.audio import save_audio
from ..io.ckpt import load_ckpt_state
from ..nn.vocoder import NSFHiFiGAN
from ..retrieval.predict import apply_cluster, retrieve
from ..train.config import TrainConfig
from ..train.dataset import SVCBatch
from .diffusion_enhancer import DiffusionEnhancer
from .features import InferFeatures


@contextmanager
def _nvtx_range(name: str):
    """nsys range (extract/run_model/stitch 分阶段); 无 profiler 时开销 ~0.6μs/push-pop."""
    if torch.cuda.is_available():
        from torch.cuda import nvtx
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()
    else:
        yield


def _resample_wav(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import numpy as np
    if orig_sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    import soxr
    return soxr.resample(np.asarray(wav, dtype=np.float32), orig_sr, target_sr)


def _resample_f0(f0: torch.Tensor, target_frames: int) -> torch.Tensor:
    # F.interpolate(align_corners=True) == np.interp(linspace(0,T-1,n), arange(T), f0), ~100x 快
    T = f0.shape[0]
    if T == target_frames:
        return f0
    out = F.interpolate(f0.detach().float().view(1, 1, T), size=target_frames,
                        mode="linear", align_corners=True)
    return out.view(target_frames).to(f0.device).to(f0.dtype)


def _resample_units(units: torch.Tensor, target_frames: int) -> torch.Tensor:
    T = units.shape[0]
    if T == target_frames:
        return units
    out = F.interpolate(units.detach().float().transpose(0, 1).unsqueeze(0),  # [T,D]->[1,D,T]
                        size=target_frames, mode="linear", align_corners=True)
    return out.squeeze(0).transpose(0, 1).to(units.device).to(units.dtype)


def _change_rms(data1, sr1, data2, sr2, rate):  # RVC change_rms: 输出 RMS 包络匹配输入
    import numpy as np
    import librosa
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)[0]
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)[0]
    n = data2.shape[0]
    env1 = np.interp(np.linspace(0, len(rms1) - 1, n), np.arange(len(rms1)), rms1)
    env2 = np.interp(np.linspace(0, len(rms2) - 1, n), np.arange(len(rms2)), rms2)
    return np.asarray(data2 * (env1 ** (1 - rate)) * (np.maximum(env2, 1e-6) ** (rate - 1)),
                      dtype=np.float32)


# -- LUFS loudness normalization (RIFT-SVC 风格) ------------------------------
_LUFS_METERS: dict[int, object] = {}
_MIN_LUFS = -70.0  # 以下视为静音/噪声底, 放大只会抬噪声


def _lufs_meter(sr: int):  # RIFT-SVC block_size=0.1
    from pyloudnorm import Meter
    m = _LUFS_METERS.get(sr)
    if m is None:
        m = Meter(sr, block_size=0.1)
        _LUFS_METERS[sr] = m
    return m


def _lufs_apply(wav, sr, target, want_loudness=False):
    """归一化到 target LUFS + 0.99 peak 保护; 静音/噪声底原样返回 (loudness=-inf)."""
    import numpy as np
    from pyloudnorm import normalize as pyln_norm
    loudness = _lufs_meter(sr).integrated_loudness(wav)
    if not np.isfinite(loudness) or loudness < _MIN_LUFS:
        return (wav, loudness) if want_loudness else wav
    out = pyln_norm.loudness(wav, loudness, target)
    max_amp = float(np.max(np.abs(out))) if out.size else 0.0
    if max_amp > 1.0:
        out = out * (0.99 / max_amp)
    out = np.asarray(out, dtype=np.float32)
    return (out, loudness) if want_loudness else out


def _normalize_loudness(wav, sr, target_lufs):  # 输入归一化 (RIFT-SVC extract_features)
    return _lufs_apply(wav, sr, target_lufs, want_loudness=True)


def _restore_loudness(wav, sr, original_lufs):  # 输出还原到输入响度 (RIFT-SVC generate_audio)
    return _lufs_apply(wav, sr, original_lufs)


MEL_DOMAIN = ("rift_dit", "ddsp6", "diffusion", "reflow_vae")  # 共享 NSF-HiFiGAN
WAV_DOMAIN = ("sovits", "rvc")  # 生成器自带解码

_DEFAULT_SR_HOP = {**{b: (44100, 512) for b in MEL_DOMAIN + ("sovits",)}, "rvc": (48000, 480)}


class InferenceEngine:
    def __init__(self, config_path, ckpt_path, vocoder_dir=None, device="cuda",
                 half=False, diffusion_ckpt=None, diffusion_config=None,
                 reflow_ckpt=None, reflow_config=None, t_start=0.6,
                 compile_graph=False, dec_cl=True, vocoder_cl=True, impl=None, config_override=None,
                 strict_shapes=False):
        from .._device import resolve_impl
        self.cfg = config_override if config_override is not None else TrainConfig.from_yaml(config_path)
        self.strict_shapes = strict_shapes
        self.backend_name = self.cfg.backend
        if self.backend_name in ("ddsp6", "sovits"):
            from ..speakers import load_model_speakers
            self.speaker_mapping, self.speaker_capacity = load_model_speakers(self.cfg, ckpt_path)
        if resolve_impl(impl) == "mlx":  # 委托 MlxInferenceEngine, torch 路径零改动
            from .mlx_engine import MlxInferenceEngine
            self._mlx_delegate = MlxInferenceEngine(
                config_path=config_path, ckpt_path=ckpt_path,
                vocoder_dir=vocoder_dir,
                device=device if device != "cuda" else "gpu")
            self.impl = "mlx"
            return
        self.impl = "torch"
        if self.backend_name not in MEL_DOMAIN + WAV_DOMAIN:
            raise ValueError(f"unknown backend: {self.backend_name}")
        self.device, self.half, self.ckpt_path = device, half, str(ckpt_path)
        # channels_last (NHWC) 需 sm80+; Turing/Volta 无硬件路径, 降级关闭 (TensorCore fp16 不受影响)
        if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] < 8:
            dec_cl = vocoder_cl = False
        opts = self.cfg.backend_opts
        sr, hop = _DEFAULT_SR_HOP[self.backend_name]
        self.sr = int(opts.get("sampling_rate", opts.get("sr", sr)))
        self.hop = int(opts.get("hop_length", opts.get("block_size", opts.get("hop", hop))))
        self.n_mels = int(opts.get("n_mel_channels", opts.get("out_dims", opts.get("n_mels", 128))))
        # rift_dit CFG 默认 OFF; spk CFG 需训练时 drop_spk_prob>0 才有效
        self.cfg_ds_strength = float(opts.get("ds_cfg_strength", 0.0))
        self.cfg_spk_strength = float(opts.get("spk_cfg_strength", 0.0))
        self.cfg_skip_strength = float(opts.get("skip_cfg_strength", 0.0))
        self.cfg_rescale = float(opts.get("cfg_rescale", 0.7))
        self.module = get_backend(self.backend_name)().build_module(self.cfg)
        self._load_weights(self.module)
        self.module.eval()
        # diffusion 采样保持 fp32 (DDPM/PLMS 数值敏感, half ~50% 跑出 NaN); vocoder 仍 autocast fp16
        self.half = half and self.backend_name != "diffusion"
        self.module = self.module.to(device)
        if dec_cl and self.backend_name in ("rvc", "sovits"):
            from ..backends.rvc.model import GeneratorNSF as _GNSF
            from ..backends.sovits.model import Generator as _SGen
            from ..nn.cl_helper import clear_wn_cache
            clear_wn_cache()
            for _mod in self.module.generator.modules():
                if isinstance(_mod, (_GNSF, _SGen)):
                    _mod._use_cl = True
        if compile_graph:
            self._compile()
        self._autocast_dtype = torch.float16 if self.half else None
        spec = get_backend(self.backend_name).feature_spec(
            {k: v for k, v in opts.items() if k in (
                "units_kind", "units_ckpt", "f0_extractor", "rmvpe_ckpt",
                "uv_policy", "n_mels", "fmin", "fmax", "rvc_version", "version")})
        self.features = InferFeatures(spec, sr=self.sr, hop=self.hop,
                                      device=device, dtype="fp32")
        self.vocoder = None
        if self.backend_name in MEL_DOMAIN:
            if vocoder_dir is None:
                from ..hub import ensure_dir
                vocoder_dir = ensure_dir("nsf_hifigan")
            vdir = Path(vocoder_dir)
            self.vocoder = NSFHiFiGAN(vdir / "config.json", vdir / "model.ckpt",
                                      device=device, is_half=False)
            if vocoder_cl:
                from ..nn.cl_helper import clear_wn_cache
                clear_wn_cache()
                self.vocoder.generator._use_cl = True
            if compile_graph:
                # forward_cl + reduce-overhead 实测 1.80x (30->17ms); 先 _use_cl 再 compile
                self._try_compile(self.vocoder, "generator", "reduce-overhead")
        self.t_start = float(t_start)
        self.diffusion_enhancer = None
        self.reflow_enhancer = None
        if (diffusion_ckpt is not None or diffusion_config is not None) or (
                reflow_ckpt is not None or reflow_config is not None):
            if self.backend_name not in ("sovits", "rvc"):
                raise ValueError("shallow diffusion is only supported for the "
                                 f"sovits/rvc backends, got backend={self.backend_name}")
            if vocoder_dir is None:
                vocoder_dir = Path(__file__).resolve().parents[2] / "pretrained" / "nsf_hifigan"
            if diffusion_ckpt is not None or diffusion_config is not None:
                if diffusion_ckpt is None or diffusion_config is None:
                    raise ValueError("shallow diffusion requires both diffusion_ckpt and "
                                     "diffusion_config")
                self.diffusion_enhancer = DiffusionEnhancer(
                    diffusion_ckpt, diffusion_config, vocoder_dir, device=device,
                    half=half, sr=self.sr, hop=self.hop, compile_sample=compile_graph)
            if reflow_ckpt is not None or reflow_config is not None:
                if reflow_ckpt is None or reflow_config is None:
                    raise ValueError("reflow shallow diffusion requires both reflow_ckpt "
                                     "and reflow_config")
                from .reflow_enhancer import ReflowEnhancer
                self.reflow_enhancer = ReflowEnhancer(
                    reflow_ckpt, reflow_config, vocoder_dir, device=device,
                    sr=self.sr, hop=self.hop)
                # reflow ODE 不编译, 但共享 vocoder 同样受益于 forward_cl + reduce-overhead
                if compile_graph and hasattr(self.reflow_enhancer.vocoder, "generator"):
                    self._try_compile(self.reflow_enhancer.vocoder, "generator",
                                      "reduce-overhead")

    def _try_compile(self, obj, attr, mode):
        import torch as _t
        try:
            setattr(obj, attr, _t.compile(getattr(obj, attr), mode=mode, dynamic=False))
        except Exception:
            pass

    def _compile(self):
        gen = getattr(self.module, "generator", None)
        if gen is not None and self.backend_name in ("rvc", "sovits"):
            # 编 enc_p+flow (default 模式): 消除 launch gap 实测 enc_p 1.49x; dec 不编
            # (channels_last+weight_norm compile 慢路径); 两者无 randn, 适合编译
            for sub in ("enc_p", "flow"):
                self._try_compile(gen, sub, "default")
        elif gen is not None:
            self._try_compile(self.module, "generator", "default")
        else:
            # mel-domain: 编 module.sample (DDPM/Flow 采样循环整体进图)
            self._try_compile(self.module, "sample", "reduce-overhead")

    @property
    def _shallow_enhancer(self):  # reflow 优先 (更快, ~10 步)
        return self.reflow_enhancer or self.diffusion_enhancer

    def _load_weights(self, module) -> None:
        sd = load_ckpt_state(self.ckpt_path)
        module_sd = module.state_dict()
        # torch.compile 包装加 _orig_mod. 前缀 -> 去前缀比对
        module_sd_flat = {k.replace("._orig_mod.", "."): v for k, v in module_sd.items()}
        # 原版 RVC/so-vits ckpt 无 generator./model. 包装 -> 逐键尝试前缀 (同 train/cli.py)
        module_keys = set(module_sd_flat.keys())
        remapped: dict = {}
        for k, v in sd.items():
            if k in module_keys:
                remapped[k] = v
                continue
            for prefix in ("generator.", "model."):
                cand = prefix + k
                if cand in module_keys:
                    remapped[cand] = v
                    break
            else:
                remapped[k] = v  # leave as-is (reported as unexpected below)
        # shape 不匹配的键跳过 (如 num_speaker 不同的 emb_g)
        skipped_shape = {k for k, v in remapped.items()
                         if k in module_sd_flat and v.shape != module_sd_flat[k].shape}
        if (self.backend_name in ("ddsp6", "sovits") or getattr(self, "strict_shapes", False)) and skipped_shape:
            raise RuntimeError(f"inference weight shape mismatch: {sorted(skipped_shape)[:8]}")
        filtered = {k: v for k, v in remapped.items() if k not in skipped_shape}
        flat_to_real = {k.replace("._orig_mod.", "."): k for k in module_sd}
        missing, unexpected = module.load_state_dict(
            {flat_to_real.get(k, k): v for k, v in filtered.items()}, strict=False)
        if self.backend_name in WAV_DOMAIN:
            gen_keys = [k for k in remapped if k.startswith("generator.")]
            if not gen_keys:
                raise RuntimeError(f"checkpoint has no 'generator.*' keys (sovits/rvc need the "
                                   f"generator state); got {list(sd)[:5]}")
        else:
            model_keys = [k for k in remapped if k.startswith("model.")]
            if not model_keys:
                raise RuntimeError(f"checkpoint has no 'model.*' keys; got {list(sd)[:5]}")
        # unexpected 只容忍 discriminator 残留
        if unexpected:
            bad = [k for k in unexpected if not k.startswith("discriminator.")]
            if bad:
                raise RuntimeError(f"checkpoint keys don't match module: {bad[:8]}")
        if missing:
            # 真缺 = 非可选键; 否则架构不匹配, 静默用未初始化权重会出垃圾输出
            OPT = ("discriminator.", "optimizer.", "criterion.", "loss.", "metrics.",
                   "generator.enc_q.")  # 发布版 RVC 剔除 training-only enc_q
            real_missing = [k for k in missing
                            if k not in skipped_shape and not k.startswith(OPT)]
            if real_missing:
                raise RuntimeError(f"checkpoint is missing {len(real_missing)} module "
                                   f"weight(s); the architecture likely mismatches "
                                   f"(n_units/n_mels/num_speaker). First few: "
                                   f"{real_missing[:8]}")

    def _enhance_units(self, units, index_path=None, index_ratio=0.0,
                       cluster_path=None, cluster_ratio=0.0, spk=None):
        mapping = getattr(self, "speaker_mapping", None)
        name = next((n for n, i in (mapping or {}).items() if i == spk), None)
        if index_path and index_ratio > 0 and Path(index_path).suffix.lower() == ".index":
            if self.backend_name != "rvc":
                raise ValueError("native .index retrieval supports the RVC backend")
            from ..retrieval.faiss_index import NativeIndexCache
            if not hasattr(self, "_native_index_cache"):
                self._native_index_cache = NativeIndexCache()
            units = self._native_index_cache.retrieve(index_path, units, ratio=index_ratio)
            index_path = None
        if index_path and index_ratio > 0:
            data = torch.load(index_path, map_location="cpu", weights_only=False)
            if self.backend_name == "sovits" and isinstance(data, dict) and data.get("spk_id") is not None:
                if name is None or data["spk_id"] != name:
                    raise ValueError("retrieval index belongs to a different speaker")
            vectors = data["vectors"] if isinstance(data, dict) and "vectors" in data else data
            if vectors.ndim != 2 or vectors.shape[1] != units.shape[-1]:
                raise ValueError("retrieval feature dimension mismatch")
            units = retrieve(vectors.to(self.device), units, ratio=index_ratio)
        if cluster_path and cluster_ratio > 0:
            data = torch.load(cluster_path, map_location="cpu", weights_only=False)
            if self.backend_name == "sovits" and isinstance(data, dict) and "cluster_centers_" not in data:
                if name is not None and name in data:
                    data = data[name]
                elif name is not None or getattr(self, "speaker_capacity", 1) > 1 or "kind" in data:
                    raise ValueError("cluster payload has no resource for the target speaker")
            if isinstance(data, dict) and "cluster_centers_" in data:
                centers = data["cluster_centers_"]
            elif isinstance(data, dict) and len(data) == 1 and isinstance(
                    next(iter(data.values())), dict):
                centers = next(iter(data.values()))["cluster_centers_"]  # {spk: {...}}
            else:
                centers = data
            if centers.ndim != 2 or centers.shape[1] != units.shape[-1]:
                raise ValueError("cluster feature dimension mismatch")
            units = apply_cluster(centers.to(self.device), units, ratio=cluster_ratio)
        return units

    @torch.no_grad()
    def convert(self, input_wav, spk=None, trans=0, index_path=None, index_ratio=0.0,
                cluster_path=None, cluster_ratio=0.0, steps=None,
                shallow_diffusion=False, k_step=100, diffusion_method="ddim",
                diffusion_speedup=10, t_start=None, reflow_steps=20,
                rms_mix_rate=1.0, protect=0.5, f0_method=None, noice_scale=0.4,
                predict_f0=False, formant_shift=0.0, ds_cfg_strength=None,
                spk_cfg_strength=None, skip_cfg_strength=None, target_loudness=None,
                restore_loudness=True, vocal_register_shift=0.0):
        """wav -> float32 wav @ backend sr. 长音频按静音分块 (RMS 阈值), 超块强制切
        最低能量帧 + 短交叉淡化; 各参数语义见 cli/infer.py --help."""
        import numpy as np
        if self.backend_name in ("ddsp6", "sovits"):
            from ..speakers import resolve_speaker
            spk = resolve_speaker(spk, self.speaker_mapping, self.speaker_capacity)
        elif spk is None:
            spk = 0
        if self.impl == "mlx":
            return self._mlx_delegate.convert(input_wav, spk=spk, trans=trans)
        if shallow_diffusion and self._shallow_enhancer is None:
            raise ValueError("shallow_diffusion=True requires diffusion_ckpt+diffusion_config "
                             "or reflow_ckpt+reflow_config at engine construction (sovits/rvc)")
        if shallow_diffusion and self.reflow_enhancer is not None:
            self._resolve_enhancer_speaker(spk)
        _t_start = self.t_start if t_start is None else float(t_start)
        original_loudness: float | None = None
        if target_loudness is not None:
            # 先归一化输入再提特征 (贴合 RIFT 训练分布)
            from ..io.audio import load_audio
            wav_in = load_audio(input_wav, sr=self.sr)
            wav_in, original_loudness = _normalize_loudness(wav_in, self.sr,
                                                            float(target_loudness))
            feats = self.features.extract(wav=wav_in, f0_method=f0_method)
        else:
            with _nvtx_range("extract"):
                feats = self.features.extract(input_wav, f0_method=f0_method)
        units = feats["units"].to(self.device)
        f0 = feats["f0"].to(self.device)
        rms = feats["rms"].to(self.device) if feats["rms"] is not None else None
        uv = feats["uv"].to(self.device)
        if trans != 0:
            f0 = f0 * (2.0 ** (trans / 12))
        units_src = units
        units = self._enhance_units(units, index_path, index_ratio,
                                    cluster_path, cluster_ratio, spk=spk)
        # protect < 0.5: 清音帧把增强前特征混回 (rms>1e-4 判 voiced; RVC f0 插值后恒>0)
        if protect < 0.5 and rms is not None and units is not units_src:
            units = self._protect_units(units, units_src, rms, protect)
        T = units.shape[0]
        # GAN 后端 O(T) wav 分配, 分块更小 (20s vs 30s)
        max_frames = (20 if self.backend_name in ("rvc", "sovits") else 30) * self.sr // self.hop
        boundaries = self._find_silence_boundaries(
            rms, T, max_frames, crossfade_frames=2 * self.sr // self.hop)
        chunks = []
        for start, end in boundaries:
            with _nvtx_range("run_model"):
                chunks.append(self._run_model(
                    units[start:end], f0[start:end],
                    rms[start:end] if rms is not None else None, uv[start:end], spk,
                    shallow_diffusion=shallow_diffusion, k_step=k_step,
                    diffusion_method=diffusion_method,
                    diffusion_speedup=diffusion_speedup, t_start=_t_start,
                    reflow_steps=reflow_steps, steps=steps, noice_scale=noice_scale,
                    predict_f0=predict_f0, formant_shift=formant_shift,
                    ds_cfg_strength=ds_cfg_strength, spk_cfg_strength=spk_cfg_strength,
                    skip_cfg_strength=skip_cfg_strength,
                    vocal_register_shift=vocal_register_shift))
        if len(chunks) > 1:
            with _nvtx_range("stitch"):
                out = self._stitch_chunks(boundaries, chunks)
        else:
            out = chunks[0].astype(np.float32)
        # 还原输入响度 (在 RMS 包络匹配之前, 保证 rms_mix 是最后一步包络)
        if original_loudness is not None and restore_loudness:
            out = _restore_loudness(out, self.sr, original_loudness)
        if rms_mix_rate != 1.0:
            out = _change_rms(feats["wav"], feats["sr"], out, self.sr, rms_mix_rate)
        return out

    def _protect_units(self, units, units_src, rms, protect):
        pitchff = torch.where(rms > 1e-4, torch.full_like(rms, protect),
                              torch.ones_like(rms)).unsqueeze(-1)  # voiced=protect, 清音=1
        return units * pitchff + units_src * (1 - pitchff)

    def _resolve_enhancer_speaker(self, spk):
        enhancer = self.reflow_enhancer
        main_capacity = getattr(self, "speaker_capacity", self.cfg.backend_opts.get("num_speaker", 1))
        capacity = getattr(enhancer, "speaker_capacity", 1)
        if main_capacity == capacity == 1:
            return 0
        mapping = getattr(self, "speaker_mapping", None)
        name = next((n for n, i in (mapping or {}).items() if i == spk), None)
        from ..speakers import resolve_speaker
        if name is None or getattr(enhancer, "speaker_mapping", None) is None:
            raise ValueError("multi-speaker enhancer requires verified matching speaker names")
        return resolve_speaker(name, enhancer.speaker_mapping, capacity)

    def _find_silence_boundaries(self, rms, T, max_chunk_frames, crossfade_frames):
        """静音切分: 贪心累积到 max_frames, 超过则切最近静音中心; 无静音则切
        近末尾最低能量帧 (overlap 供交叉淡化, 静音切零 overlap)."""
        import numpy as np
        min_sil = int(0.15 * self.sr / self.hop)  # 150ms = 静音
        if rms is not None and T > 0:
            rms_np = rms.float().cpu().numpy()
            # 自适应阈值: 30 分位 RMS 的 10% (对少数尖峰鲁棒); 全响则落到 -40dB
            silence_thresh = max(float(np.percentile(rms_np, 30)) * 0.10, 1e-4)
            is_silence = rms_np < silence_thresh
        else:
            is_silence = np.zeros(T, dtype=bool)
        centers = np.zeros(T, dtype=bool)
        if is_silence.any():
            d = np.diff(np.concatenate(([0], is_silence.astype(np.int8), [0])))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            for s, e in zip(starts, ends):
                if e - s >= min_sil:
                    centers[s + (e - s) // 2] = True  # 静音段中心帧为切点
        boundaries = []
        start = 0
        while start < T:
            ideal_end = start + max_chunk_frames
            if ideal_end >= T:
                boundaries.append((start, T))
                break
            sil = np.where(centers[start + max_chunk_frames // 2:ideal_end])[0]
            if len(sil) > 0:
                cut = start + max_chunk_frames // 2 + int(sil[-1]) + 1  # 最靠近上限的静音
                boundaries.append((start, cut))
                start = cut
            else:
                win = max(self.sr // self.hop, crossfade_frames)  # 1s 窗口
                lo = max(start, ideal_end - win)
                if rms is not None:
                    cut = lo + int(np.argmin(rms[lo:ideal_end].float().cpu().numpy()))
                else:
                    cut = ideal_end - crossfade_frames
                boundaries.append((start, cut))
                start = cut - crossfade_frames  # overlap 供交叉淡化
        return boundaries

    def _run_model(self, units, f0, rms, uv, spk, shallow_diffusion=False,
                   k_step=100, diffusion_method="ddim", diffusion_speedup=10,
                   t_start=0.6, reflow_steps=20, steps=None, noice_scale=0.4,
                   predict_f0=False, formant_shift=0.0, ds_cfg_strength=None,
                   spk_cfg_strength=None, skip_cfg_strength=None,
                   vocal_register_shift=0.0):
        T = units.shape[0]
        spk_id = torch.zeros(1, dtype=torch.long, device=self.device) + spk
        frame_len = torch.tensor([T], device=self.device, dtype=torch.long)
        f0_for_model = f0
        # pc 音区移位: 模型见移位 f0 (换音区), vocoder 见原 f0 (音高不变);
        # 仅 mel-domain + pc-aug vocoder 有效, rvc/sovits 是 wav-domain 忽略
        if vocal_register_shift != 0.0 and self.backend_name in (
                "ddsp6", "diffusion", "reflow_vae", "rift_dit"):
            f0_for_model = f0 / (2.0 ** (vocal_register_shift / 12.0))
        batch = SVCBatch(
            units=units.unsqueeze(0), f0=f0_for_model.unsqueeze(0),
            mel=torch.zeros(1, T, self.n_mels, device=self.device, dtype=units.dtype),
            wav=torch.zeros(1, 1, T * self.hop, device=self.device, dtype=units.dtype),
            rms=rms.unsqueeze(0) if rms is not None else None,
            spk_id=spk_id, uv=uv.unsqueeze(0),
            aug_shift=(torch.tensor([[formant_shift]], device=self.device,
                                    dtype=units.dtype) if formant_shift != 0.0 else None),
            pitch_token=None, frame_len=frame_len, key=None)
        ac = (torch.autocast("cuda", dtype=self._autocast_dtype)
              if self._autocast_dtype is not None else _nullcontext())
        with ac:
            if self.backend_name == "rift_dit":
                _ds = self.cfg_ds_strength if ds_cfg_strength is None else ds_cfg_strength
                _spk = self.cfg_spk_strength if spk_cfg_strength is None else spk_cfg_strength
                _skip = self.cfg_skip_strength if skip_cfg_strength is None else skip_cfg_strength
                batch.mel = torch.zeros(1, T, self.n_mels, device=self.device, dtype=units.dtype)
                bad_cvec = (_resample_units(batch.units[0, ::2, :], T).unsqueeze(0)
                            if _ds > 1e-5 else None)
                mel = self.module.sample(batch, ds_cfg_strength=_ds, spk_cfg_strength=_spk,
                                         skip_cfg_strength=_skip, cfg_rescale=self.cfg_rescale,
                                         bad_cvec=bad_cvec, steps=steps)
                if isinstance(mel, tuple):
                    mel = mel[0]
                wav = self.vocoder.infer(mel.float(), f0.float().unsqueeze(0))
            elif self.backend_name in ("ddsp6", "diffusion", "reflow_vae"):
                mel = self.module.sample(batch, steps=steps)
                wav = self.vocoder.infer(mel.float(), f0.float().unsqueeze(0))
            elif self.backend_name == "sovits":
                wav, _ = self.module.generator.infer(
                    batch.units.transpose(1, 2),
                    interp_unvoiced_f0(batch.f0[0]).unsqueeze(0),
                    batch.uv, batch.spk_id.unsqueeze(-1), vol=batch.rms,
                    noice_scale=noice_scale, predict_f0=predict_f0)
            elif self.backend_name == "rvc":
                wav, *_ = self.module.generator.infer(
                    batch.units, batch.frame_len,
                    self.module.generator.f0_to_coarse(batch.f0.float()),
                    batch.f0, batch.spk_id)
            else:  # pragma: no cover
                raise ValueError(self.backend_name)
        if shallow_diffusion and self.backend_name in ("sovits", "rvc"):
            return self._enhance_wav(units, f0, rms, spk, wav[0, 0].float(),
                                     k_step=k_step, diffusion_method=diffusion_method,
                                     diffusion_speedup=diffusion_speedup,
                                     t_start=t_start, reflow_steps=reflow_steps)
        return wav[0, 0].float().cpu().numpy()

    def _enhance_wav(self, units, f0, rms, spk, gen_wav, k_step=100,
                     diffusion_method="ddim", diffusion_speedup=10, t_start=0.6,
                     reflow_steps=20):
        import numpy as np
        enh = self._shallow_enhancer
        enh_sr, backend_sr = enh.sr, self.sr
        wav_in, f0_in, units_in, rms_in = gen_wav, f0, units, rms
        if backend_sr != enh_sr:
            # 48k rvc -> 44.1k 增强器域: wav 与帧域张量全部重对齐到 enh 帧网格
            # (否则 T 错位: 增强器尾部补零而 units/f0 留在后端网格, 静默损坏条件)
            wav_in = torch.from_numpy(_resample_wav(gen_wav.cpu().numpy(),
                                                    backend_sr, enh_sr)).to(gen_wav.dtype).to(self.device)
            enh_frames = wav_in.shape[-1] // enh.hop
            f0_in = _resample_f0(f0, enh_frames)
            units_in = _resample_units(units, enh_frames)
            rms_in = _resample_f0(rms, enh_frames) if rms_in is not None else None
        if self.reflow_enhancer is not None:
            enhancer_spk = self._resolve_enhancer_speaker(spk)
            enh_out = self.reflow_enhancer.enhance(units_in, f0_in, rms_in, enhancer_spk, wav_in,
                                                   t_start=t_start, infer_step=reflow_steps)
        else:
            enh_out = self.diffusion_enhancer.enhance(units_in, f0_in, rms_in, spk, wav_in,
                                                      k_step=k_step,
                                                      infer_speedup=diffusion_speedup,
                                                      method=diffusion_method)
        if backend_sr != enh_sr:
            enh_out = _resample_wav(enh_out, enh_sr, backend_sr)
        return enh_out.astype(np.float32)

    def _stitch_chunks(self, boundaries, chunks):
        import numpy as np
        out = chunks[0].copy()
        for i in range(1, len(chunks)):
            cur = chunks[i]
            prev_end, cur_start = boundaries[i - 1][1], boundaries[i][0]
            if cur_start >= prev_end:  # 静音切: 直接拼接
                out = np.concatenate([out, cur])
            else:  # 强制切 (overlap): 线性交叉淡化
                overlap = min((prev_end - cur_start) * self.hop, len(out), len(cur))
                if overlap <= 0:
                    out = np.concatenate([out, cur])
                else:
                    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
                    out[-overlap:] = out[-overlap:] * (1 - ramp) + cur[:overlap] * ramp
                    out = np.concatenate([out, cur[overlap:]])
        return out.astype(np.float32)

    def convert_file(self, input_wav, output_wav, resample_sr=None, **kwargs):
        wav = self.convert(input_wav, **kwargs)
        sr = self.sr
        if resample_sr is not None and resample_sr != self.sr:
            wav = _resample_wav(wav, self.sr, resample_sr)
            sr = resample_sr
        save_audio(str(output_wav), wav, sr=sr)
