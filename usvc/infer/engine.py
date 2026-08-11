"""Offline inference engine: wav -> features -> backend model -> wav.

Supports all 6 backends:
  - mel-domain (rift_dit / ddsp6 / diffusion / reflow_vae): model samples mel,
    shared NSF-HiFiGAN vocoder converts mel+f0 -> wav.
  - wav-domain (sovits / rvc): the generator itself decodes waveform
    (VITS + built-in NSF), no external vocoder needed.

Usage (CLI):  usvc-infer input.wav --config configs/sovits.yaml --ckpt ck.ckpt
"""
from __future__ import annotations

from contextlib import nullcontext as _nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..backends import get_backend
from ..backends.sovits.dataset import interp_unvoiced_f0
from ..io.audio import save_audio
from ..nn.vocoder import NSFHiFiGAN
from ..retrieval.predict import apply_cluster, retrieve
from ..train.config import TrainConfig
from ..train.dataset import SVCBatch
from .diffusion_enhancer import DiffusionEnhancer
from .features import InferFeatures


def _resample_wav(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Soxr-based wav resampling (high quality, linear phase)."""
    if orig_sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    import soxr
    return soxr.resample(np.asarray(wav, dtype=np.float32), orig_sr, target_sr)


def _resample_f0(f0: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Align a [T] f0 trajectory to ``target_frames`` via linear interpolation.

    Uses GPU ``F.interpolate`` (align_corners=True) which is numerically
    equivalent to numpy ``np.interp(linspace(0,T-1,n), arange(T), f0)`` but
    ~100x faster (avoids CPU round-trip).
    """
    T = f0.shape[0]
    if T == target_frames:
        return f0
    x = f0.detach().float().view(1, 1, T)
    out = F.interpolate(x, size=target_frames, mode="linear", align_corners=True)
    return out.view(target_frames).to(f0.device).to(f0.dtype)


def _resample_units(units: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Align [T, D] content features to ``target_frames`` via linear interp.

    GPU ``F.interpolate`` (align_corners=True) replaces the old per-dimension
    numpy loop (~14ms -> ~0.5ms for 768-d contentvec).
    """
    T = units.shape[0]
    if T == target_frames:
        return units
    # [T, D] -> [1, D, T] for interpolate along time axis
    x = units.detach().float().transpose(0, 1).unsqueeze(0)
    out = F.interpolate(x, size=target_frames, mode="linear", align_corners=True)
    return out.squeeze(0).transpose(0, 1).to(units.device).to(units.dtype)


def _change_rms(data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float) -> np.ndarray:
    """RVC pipeline.change_rms (numpy port): match output RMS envelope to input.

    data1 = source (input) wav at sr1, data2 = converted wav at sr2.
    rate=1.0 -> untouched; rate=0.0 -> fully input envelope.
    """
    import librosa

    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)[0]
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)[0]
    n = data2.shape[0]
    # Interpolate both envelopes to the output sample grid (np.interp clamps
    # at the endpoints, matching F.interpolate's linear behavior).
    env1 = np.interp(np.linspace(0, len(rms1) - 1, n), np.arange(len(rms1)), rms1)
    env2 = np.interp(np.linspace(0, len(rms2) - 1, n), np.arange(len(rms2)), rms2)
    env2 = np.maximum(env2, 1e-6)
    out = data2 * (env1 ** (1 - rate)) * (env2 ** (rate - 1))
    return np.asarray(out, dtype=np.float32)


# -- LUFS loudness normalization (RIFT-SVC style) -----------------------------

_LUFS_METERS: dict[int, object] = {}
# Integrated loudness below this (basically silence / pure noise floor) is not
# normalized — amplifying it by tens of dB just boosts the noise floor.
_MIN_LUFS = -70.0


def _lufs_meter(sr: int):
    """Cached pyloudnorm Meter (RIFT-SVC uses block_size=0.1)."""
    from pyloudnorm import Meter

    m = _LUFS_METERS.get(sr)
    if m is None:
        m = Meter(sr, block_size=0.1)
        _LUFS_METERS[sr] = m
    return m


def _normalize_loudness(
    wav: np.ndarray, sr: int, target_lufs: float
) -> tuple[np.ndarray, float]:
    """Normalize wav to target integrated loudness (LUFS).

    Mirrors RIFT-SVC extract_features: measure original loudness, normalize to
    target, then scale back to 0.99 peak if the gain would clip. Returns
    (normalized_wav, original_lufs); silence / noise-floor audio is returned
    untouched with original_lufs = -inf so callers skip the restore step.
    """
    from pyloudnorm import normalize as pyln_normalize

    loudness = _lufs_meter(sr).integrated_loudness(wav)
    if not np.isfinite(loudness) or loudness < _MIN_LUFS:
        return wav, loudness
    out = pyln_normalize.loudness(wav, loudness, target_lufs)
    max_amp = float(np.max(np.abs(out))) if out.size else 0.0
    if max_amp > 1.0:
        out = out * (0.99 / max_amp)
    return np.asarray(out, dtype=np.float32), loudness


def _restore_loudness(wav: np.ndarray, sr: int, original_lufs: float) -> np.ndarray:
    """Restore output loudness to the input's original integrated loudness.

    Mirrors RIFT-SVC generate_audio: measure the output loudness and normalize
    it back to the original input loudness, with 0.99 peak protection.
    """
    from pyloudnorm import normalize as pyln_normalize

    loudness = _lufs_meter(sr).integrated_loudness(wav)
    if not np.isfinite(loudness) or loudness < _MIN_LUFS:
        return wav
    out = pyln_normalize.loudness(wav, loudness, original_lufs)
    max_amp = float(np.max(np.abs(out))) if out.size else 0.0
    if max_amp > 1.0:
        out = out * (0.99 / max_amp)
    return np.asarray(out, dtype=np.float32)

# mel-domain backends use the shared NSF-HiFiGAN vocoder
MEL_DOMAIN = ("rift_dit", "ddsp6", "diffusion", "reflow_vae")
# wav-domain backends decode waveform directly from the generator
WAV_DOMAIN = ("sovits", "rvc")

# Default sample-rate / hop per backend family (override via backend_opts)
_DEFAULT_SR_HOP = {
    "rift_dit": (44100, 512),
    "ddsp6": (44100, 512),
    "diffusion": (44100, 512),
    "reflow_vae": (44100, 512),
    "sovits": (44100, 512),
    "rvc": (48000, 480),
}


def _load_ckpt_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        # Accept common top-level wrappers: lightning (state_dict), original
        # RVC/so-vits (model), and bare weight dicts (weight / no wrapper).
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt.get("weight", ckpt)))
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f"cannot interpret checkpoint: {ckpt_path}")
    return sd


class InferenceEngine:
    """Offline SVC inference for one model checkpoint."""

    def __init__(
        self,
        config_path: str | Path,
        ckpt_path: str | Path,
        vocoder_dir: str | Path | None = None,
        device: str = "cuda",
        half: bool = False,
        diffusion_ckpt: str | Path | None = None,
        diffusion_config: str | Path | None = None,
        reflow_ckpt: str | Path | None = None,
        reflow_config: str | Path | None = None,
        t_start: float = 0.5,
        compile_graph: bool = False,
        dec_cl: bool = True,
        vocoder_cl: bool = True,
        impl: str | None = None,
    ) -> None:
        # ---- MLX 后端分发(与 torch 并列,torch 路径零改动)----
        # impl='mlx'(或 USVC_BACKEND=mlx 且 impl='auto')时,委托给
        # MlxInferenceEngine,跳过下面的 torch 初始化。
        from .._device import resolve_impl
        resolved_impl = resolve_impl(impl)
        if resolved_impl == "mlx":
            from .mlx_engine import MlxInferenceEngine
            self._mlx_delegate = MlxInferenceEngine(
                config_path=config_path,
                ckpt_path=ckpt_path,
                vocoder_dir=vocoder_dir,
                device=device if device != "cuda" else "gpu",
            )
            self.impl = "mlx"
            return
        self.impl = "torch"

        self.cfg = TrainConfig.from_yaml(config_path)
        self.backend_name = self.cfg.backend
        if self.backend_name not in MEL_DOMAIN + WAV_DOMAIN:
            raise ValueError(f"unknown backend: {self.backend_name}")
        self.device = device
        self.half = half
        self.ckpt_path = str(ckpt_path)

        # channels_last (NHWC) 原生支持需 Ampere+ (sm80+)。更早架构（Turing sm75 /
        # Volta sm70）无硬件 NHWC 路径，dec_cl/vocoder_cl 反而引入逐层布局转换开销，
        # 故在 sm<80 上自动降级关闭（TensorCore fp16 加速不受影响）。
        if device.startswith("cuda"):
            _cap = torch.cuda.get_device_capability(device)
            if _cap[0] < 8:
                dec_cl = False
                vocoder_cl = False

        # Sample rate / hop for inference features (from config or defaults).
        opts = self.cfg.backend_opts
        self.sr = int(opts.get("sampling_rate", opts.get("sr", _DEFAULT_SR_HOP[self.backend_name][0])))
        self.hop = int(opts.get("hop_length", opts.get("block_size", opts.get("hop", _DEFAULT_SR_HOP[self.backend_name][1]))))
        self.n_mels = int(opts.get("n_mel_channels", opts.get("out_dims", opts.get("n_mels", 128))))

        # rift_dit CFG (classifier-free guidance) inference params.
        # Default OFF (0.0); enable via config backend_opts to match RIFT-SVC
        # defaults (ds=0.2, spk=0.8, skip=0.0, rescale=0.7).
        # Note: spk CFG requires drop_spk_prob>0 at training time to be effective.
        self.cfg_ds_strength = float(opts.get("ds_cfg_strength", 0.0))
        self.cfg_spk_strength = float(opts.get("spk_cfg_strength", 0.0))
        self.cfg_skip_strength = float(opts.get("skip_cfg_strength", 0.0))
        self.cfg_rescale = float(opts.get("cfg_rescale", 0.7))

        # Backend module + weights.
        backend_cls = get_backend(self.backend_name)
        self.module = backend_cls().build_module(self.cfg)
        self._load_weights(self.module)
        self.module.eval()
        self.module.to(device)
        # Diffusion keeps fp32 sampling: the DDPM/PLMS iterative loop is
        # numerically sensitive and random-weight models diverge to NaN in
        # half precision (observed ~50% of runs). The vocoder below still
        # runs in half for the mel->wav stage.
        self.half = half and self.backend_name != "diffusion"
        self.module = self.module.to(device)
        # rvc/sovits dec channels_last 推理（conv1d→conv2d(1,k) 等价变换，消除逐层
        # nchw<->nhwc 转换 ~12ms）。仅 GAN 后端生效。
        if dec_cl and self.backend_name in ("rvc", "sovits"):
            from ..backends.rvc.model import GeneratorNSF as _GNSF
            from ..backends.sovits.model import Generator as _SGen
            from ..nn.cl_helper import clear_wn_cache

            clear_wn_cache()  # 新引擎从干净缓存开始（权重已冻结）
            for _mod in self.module.generator.modules():
                if isinstance(_mod, (_GNSF, _SGen)):
                    _mod._use_cl = True
        # torch.compile 可编译的推理路径（reduce-overhead 走 CUDA graph，减发射开销）。
        # 实测 rvc 162→140ms (1.16×)，SM 61% launch-bound 场景收益明显。
        # GAN 后端编 generator；mel-domain 后端编 module.sample（不是 model——
        # 推理走 sample 方法，DDPM/Flow 采样循环整体进 CUDA graph，否则
        # 每步的 Python 发射开销是 launch-bound 主瓶颈）。默认关。
        if compile_graph:
            if getattr(self.module, "generator", None) is not None:
                target = self.module.generator
                try:
                    import torch as _t
                    self.module.generator = _t.compile(
                        target, mode="reduce-overhead", dynamic=False)
                except Exception:
                    pass
            else:
                try:
                    import torch as _t
                    self.module.sample = _t.compile(
                        self.module.sample, mode="reduce-overhead", dynamic=False)
                except Exception:
                    pass
        # autocast: conv-heavy nets (NSF-HiFiGAN, GAN generators) get fp16
        # tensor-core acceleration while accumulation/reductions stay fp32.
        # Unlike pure .half(), autocast keeps weight_norm norms in fp32,
        # avoiding the ~3x slowdown seen with pure half precision.
        self._autocast_dtype = torch.float16 if self.half else None

        # Feature spec drives inference extraction (same encoder as training).
        # Keep this filter list in sync with TrainConfig._FEATURE_OVERRIDE_KEYS
        # so inference uses the same units/f0/mel encoder as the trained ckpt.
        spec = backend_cls.feature_spec({k: v for k, v in opts.items() if k in (
            "units_kind", "units_ckpt", "f0_extractor", "rmvpe_ckpt", "uv_policy",
            "n_mels", "fmin", "fmax", "rvc_version", "version",
        )})
        self.features = InferFeatures(
            spec, sr=self.sr, hop=self.hop, device=device,
            dtype="fp32",
        )

        # Vocoder for mel-domain backends (uses shared openvpi NSF-HiFiGAN).
        self.vocoder = None
        if self.backend_name in MEL_DOMAIN:
            if vocoder_dir is None:
                from ..hub import ensure_dir
                vocoder_dir = ensure_dir("nsf_hifigan")
            vocoder_dir = Path(vocoder_dir)
            self.vocoder = NSFHiFiGAN(
                vocoder_dir / "config.json",
                vocoder_dir / "model.ckpt",
                device=device,
                # Keep vocoder in fp32; autocast (via self._autocast_dtype)
                # handles fp16 tensor-core acceleration at forward time.
                is_half=False,
            )
            if vocoder_cl:
                from ..nn.cl_helper import clear_wn_cache

                clear_wn_cache()  # 新引擎从干净缓存开始（权重已冻结）
                self.vocoder.generator._use_cl = True
            # compile 共享 vocoder（实测 forward_cl + reduce-overhead 1.80×：
            # 30→17ms/10s。纯 forward + compile 反而变慢（nchw↔nhwc 胶水叠加），
            # 必须先 _use_cl=True 再 compile。仅固定形状稳态 chunk 生效，
            # 变长分块由上层 convert 按固定 chunk 调用保证。）
            if compile_graph:
                try:
                    import torch as _t
                    self.vocoder.generator = _t.compile(
                        self.vocoder.generator,
                        mode="reduce-overhead", dynamic=False)
                except Exception:
                    pass

        self.t_start = float(t_start)

        # Shallow-diffusion enhancers (optional; sovits and rvc).  Two enhancer
        # backends are supported and may coexist:
        #   - DiffusionEnhancer (Diffusion-SVC, k-step DDPM denoising)
        #   - ReflowEnhancer   (DDSP6 RectifiedFlow, t_start shallow flow)
        # ``convert(shallow_diffusion=True)`` picks whichever is loaded; if
        # both are loaded, reflow is preferred (faster, ~10 steps).
        self.diffusion_enhancer = None
        self.reflow_enhancer = None
        if (diffusion_ckpt is not None or diffusion_config is not None) or (
            reflow_ckpt is not None or reflow_config is not None
        ):
            if self.backend_name not in ("sovits", "rvc"):
                raise ValueError(
                    "shallow diffusion is only supported for the "
                    f"sovits/rvc backends, got backend={self.backend_name}"
                )
            if vocoder_dir is None:
                vocoder_dir = Path(__file__).resolve().parents[2] / "pretrained" / "nsf_hifigan"
            if diffusion_ckpt is not None or diffusion_config is not None:
                if diffusion_ckpt is None or diffusion_config is None:
                    raise ValueError(
                        "shallow diffusion requires both diffusion_ckpt and "
                        "diffusion_config"
                    )
                self.diffusion_enhancer = DiffusionEnhancer(
                    diffusion_ckpt, diffusion_config, vocoder_dir,
                    device=device, half=half, sr=self.sr, hop=self.hop,
                    compile_sample=compile_graph,
                )
            if reflow_ckpt is not None or reflow_config is not None:
                if reflow_ckpt is None or reflow_config is None:
                    raise ValueError(
                        "reflow shallow diffusion requires both reflow_ckpt "
                        "and reflow_config"
                    )
                from .reflow_enhancer import ReflowEnhancer
                self.reflow_enhancer = ReflowEnhancer(
                    reflow_ckpt, reflow_config, vocoder_dir,
                    device=device, sr=self.sr, hop=self.hop,
                )
                # reflow_enhancer 无 compile_sample 概念（ODE 不编译），
                # 但其共享 vocoder 同样受益于 forward_cl + reduce-overhead。
                if compile_graph and hasattr(self.reflow_enhancer.vocoder, "generator"):
                    try:
                        import torch as _t
                        self.reflow_enhancer.vocoder.generator = _t.compile(
                            self.reflow_enhancer.vocoder.generator,
                            mode="reduce-overhead", dynamic=False)
                    except Exception:
                        pass

    @property
    def _shallow_enhancer(self):
        """Return the active shallow-diffusion enhancer (reflow preferred)."""
        if self.reflow_enhancer is not None:
            return self.reflow_enhancer
        return self.diffusion_enhancer

    def _load_weights(self, module) -> None:
        sd = _load_ckpt_state(self.ckpt_path)
        module_sd = module.state_dict()

        # torch.compile 包装（OptimizedModule）会让子模块 state_dict 键带
        # ``_orig_mod.`` 前缀（如 denoise_fn._orig_mod.input_projection.weight）；
        # 把 module 侧键去前缀后对比，保持与未编译 ckpt 兼容。
        module_sd_flat = {k.replace("._orig_mod.", "."): v for k, v in module_sd.items()}

        # Remap bare keys -> prefixed keys: original RVC/so-vits checkpoints
        # store weights without the ``generator.``/``model.`` wrapper. Match
        # each key individually against the module state_dict, trying (no
        # prefix, generator., model.) in order (same strategy as
        # train/cli.py pretrained loading).
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

        # Shape-filter: keys whose shape mismatches the module (e.g. emb_g
        # when num_speaker differs from the checkpoint) are skipped instead
        # of crashing load_state_dict.
        skipped_shape = {
            k for k, v in remapped.items()
            if k in module_sd_flat and v.shape != module_sd_flat[k].shape
        }
        filtered = {k: v for k, v in remapped.items() if k not in skipped_shape}
        # flat 键（无 _orig_mod.）映射回 module 真实键后再加载。
        flat_to_real = {k.replace("._orig_mod.", "."): k for k in module_sd}
        filtered_real = {flat_to_real.get(k, k): v for k, v in filtered.items()}
        missing, unexpected = module.load_state_dict(filtered_real, strict=False)
        # strict=False tolerates optimizer/discriminator-only keys in lightning
        # ckpts (sovits/rvc have discriminator + generator keys; generator is
        # required). Filter expected suffixes: bare model ckpts may drop
        # discriminator, so only *required* model keys must be present.
        if self.backend_name in WAV_DOMAIN:
            gen_keys = [k for k in remapped if k.startswith("generator.")]
            if not gen_keys:
                raise RuntimeError(
                    f"checkpoint has no 'generator.*' keys (sovits/rvc need the "
                    f"generator state); got {list(sd)[:5]}"
                )
        else:
            model_keys = [k for k in remapped if k.startswith("model.")]
            if not model_keys:
                raise RuntimeError(
                    f"checkpoint has no 'model.*' keys; got {list(sd)[:5]}"
                )
        # Any unexpected keys not matching the module are tolerated only if
        # they are discriminator/optimizer leftovers.
        if unexpected:
            tolerable = [k for k in unexpected if k.startswith(("discriminator.",))]
            bad = [k for k in unexpected if k not in tolerable]
            if bad:
                raise RuntimeError(f"checkpoint keys don't match module: {bad[:8]}")
        # Guard against silently partial loads: a checkpoint from a different
        # architecture (e.g. different n_units / n_mels / num_speaker) that
        # lacks module keys without a shape clash would otherwise load with
        # UNINITIALIZED weights and produce garbage. Only truly optional keys
        # (discriminator / optimizer / criterion buffers / shape-skipped /
        # training-only enc_q) may be missing.
        if missing:
            _OPTIONAL_PREFIXES = (
                "discriminator.", "optimizer.", "criterion.", "loss.", "metrics.",
                "generator.enc_q.",  # released RVC weights omit training-only enc_q
            )
            real_missing = [
                k for k in missing
                if k not in skipped_shape and not k.startswith(_OPTIONAL_PREFIXES)
            ]
            if real_missing:
                raise RuntimeError(
                    f"checkpoint is missing {len(real_missing)} module weight(s); "
                    f"the architecture likely mismatches (n_units/n_mels/"
                    f"num_speaker). First few: {real_missing[:8]}"
                )

    # -- feature enhancement (retrieval / cluster) ---------------------------

    def _enhance_units(self, units: torch.Tensor, index_path=None, index_ratio=0.0,
                       cluster_path=None, cluster_ratio=0.0) -> torch.Tensor:
        """Apply RVC-style feature retrieval and/or so-vits cluster mapping."""
        if index_path and index_ratio > 0:
            data = torch.load(index_path, map_location="cpu", weights_only=False)
            vectors = data["vectors"] if isinstance(data, dict) and "vectors" in data else data
            units = retrieve(vectors.to(self.device), units, ratio=index_ratio)
        if cluster_path and cluster_ratio > 0:
            data = torch.load(cluster_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "cluster_centers_" in data:
                centers = data["cluster_centers_"]
            elif isinstance(data, dict) and len(data) == 1 and isinstance(
                next(iter(data.values())), dict
            ):
                # per-speaker cluster dict {spk: {"cluster_centers_": ...}}
                centers = next(iter(data.values()))["cluster_centers_"]
            else:
                centers = data
            units = apply_cluster(centers.to(self.device), units, ratio=cluster_ratio)
        return units

    # -- conversion -----------------------------------------------------------

    @torch.no_grad()
    def convert(
        self,
        input_wav: str | Path,
        spk: int = 0,
        trans: int = 0,
        index_path: str | Path | None = None,
        index_ratio: float = 0.0,
        cluster_path: str | Path | None = None,
        cluster_ratio: float = 0.0,
        steps: int | None = None,
        shallow_diffusion: bool = False,
        k_step: int = 100,
        diffusion_method: str = "ddim",
        diffusion_speedup: int = 10,
        t_start: float | None = None,
        rms_mix_rate: float = 1.0,
        protect: float = 0.5,
        f0_method: str | None = None,
        noice_scale: float = 0.4,
        predict_f0: bool = False,
        formant_shift: float = 0.0,
        ds_cfg_strength: float | None = None,
        spk_cfg_strength: float | None = None,
        skip_cfg_strength: float | None = None,
        target_loudness: float | None = None,
        restore_loudness: bool = True,
        vocal_register_shift: float = 0.0,
    ) -> np.ndarray:
        """Convert a wav file; returns float32 wav at the backend sample rate.

        Long audio is split at silence boundaries to bound GPU memory and
        avoid artifacts at chunk edges. Silence is detected via RMS energy;
        chunks are inferred independently and concatenated. If a speech
        segment exceeds the max chunk size, it is force-split at the lowest-
        energy frame with a short crossfade.

        shallow_diffusion: enhance each chunk's wav via the loaded enhancer
            (DiffusionEnhancer k-step denoising, or ReflowEnhancer t_start
            shallow flow — whichever is loaded, reflow preferred).
        t_start: reflow ODE start time in [0,1); overrides the engine default.
            Ignored when only the diffusion enhancer is loaded.
        rms_mix_rate: RVC-style output RMS envelope match (1.0 = untouched;
            0.0 = fully input envelope).
        protect: RVC-style source-feature blend on speech frames (>= 0.5 =
            disabled; 0.33 typical). Uses rms > threshold as the speech-frame
            criterion (rvc f0 is interpolated so it is never 0 except digital
            silence). Only meaningful with index/cluster enhancement.
        f0_method: optional runtime f0 extractor override ("rmvpe" | "fcpe");
            None uses the config default.
        target_loudness: LUFS value to normalize the input to before feature
            extraction (RIFT-SVC style; None = disabled). Features are
            extracted from the normalized audio, so this also matches the
            RIFT training distribution for the rift_dit backend.
        restore_loudness: after conversion, restore the output loudness to the
            input's original integrated loudness (only when target_loudness
            is set and the input loudness was measurable).
        vocal_register_shift: pc vocal register shift in semitones (mel-domain
            backends only: ddsp6/diffusion/reflow_vae/rift_dit with a pc-aug
            vocoder). The model sees f0/(2^(shift/12)) while the vocoder sees
            the original f0: pitch stays, register shifts. 0.0 = off.
        """
        if self.impl == "mlx":
            return self._mlx_delegate.convert(input_wav, spk=spk, trans=trans)
        if shallow_diffusion and self._shallow_enhancer is None:
            raise ValueError(
                "shallow_diffusion=True requires diffusion_ckpt+diffusion_config "
                "or reflow_ckpt+reflow_config at engine construction (sovits/rvc)"
            )
        _t_start = self.t_start if t_start is None else float(t_start)
        original_loudness: float | None = None
        if target_loudness is not None:
            # Normalize the input first so features are extracted from the
            # loudness-matched audio (matches RIFT training distribution).
            from ..io.audio import load_audio

            wav_in = load_audio(input_wav, sr=self.sr)
            wav_in, original_loudness = _normalize_loudness(
                wav_in, self.sr, float(target_loudness)
            )
            feats = self.features.extract(wav=wav_in, f0_method=f0_method)
        else:
            feats = self.features.extract(input_wav, f0_method=f0_method)
        units = feats["units"].to(self.device)  # [T, dim]
        f0 = feats["f0"].to(self.device)  # [T]
        rms = feats["rms"].to(self.device) if feats["rms"] is not None else None
        uv = feats["uv"].to(self.device)

        # autocast handles dtype internally; keep fp32 inputs.

        # Pitch shift (semitones).
        if trans != 0:
            f0 = f0 * (2.0 ** (trans / 12))

        # Feature retrieval / cluster enhancement (RVC / so-vits style).
        # protect: keep the pre-enhancement units as the source features and
        # blend them back on speech frames afterwards.
        units_src = units
        units = self._enhance_units(
            units, index_path, index_ratio, cluster_path, cluster_ratio
        )
        if protect < 0.5 and rms is not None and units is not units_src:
            units = self._protect_units(units, units_src, rms, protect)

        T = units.shape[0]
        # Max chunk size by backend: GAN backends (rvc/sovits) allocate O(T) wav
        # via NSF vocoder, so cap smaller. Flow backends (mel-domain) can go larger.
        max_chunk_sec = 30
        if self.backend_name in ("rvc", "sovits"):
            max_chunk_sec = 20
        max_chunk_frames = max_chunk_sec * self.sr // self.hop

        if T <= max_chunk_frames:
            out = self._run_model(
                units, f0, rms, uv, spk,
                shallow_diffusion=shallow_diffusion, k_step=k_step,
                diffusion_method=diffusion_method,
                diffusion_speedup=diffusion_speedup,
                t_start=_t_start,
                steps=steps,
                noice_scale=noice_scale,
                predict_f0=predict_f0,
                formant_shift=formant_shift,
                ds_cfg_strength=ds_cfg_strength,
                spk_cfg_strength=spk_cfg_strength,
                skip_cfg_strength=skip_cfg_strength,
                vocal_register_shift=vocal_register_shift,
            ).astype(np.float32)
        else:
            # Silence-aware segmentation: split at silence boundaries.
            boundaries = self._find_silence_boundaries(
                rms, T, max_chunk_frames, crossfade_frames=2 * self.sr // self.hop
            )

            chunk_outputs = []
            for start, end in boundaries:
                chunk_units = units[start:end]
                chunk_f0 = f0[start:end]
                chunk_rms = rms[start:end] if rms is not None else None
                chunk_uv = uv[start:end]
                wav_chunk = self._run_model(
                    chunk_units, chunk_f0, chunk_rms, chunk_uv, spk,
                    shallow_diffusion=shallow_diffusion, k_step=k_step,
                    diffusion_method=diffusion_method,
                    diffusion_speedup=diffusion_speedup,
                    t_start=_t_start,
                    steps=steps,
                    noice_scale=noice_scale,
                    predict_f0=predict_f0,
                    formant_shift=formant_shift,
                    ds_cfg_strength=ds_cfg_strength,
                    spk_cfg_strength=spk_cfg_strength,
                    skip_cfg_strength=skip_cfg_strength,
                    vocal_register_shift=vocal_register_shift,
                )
                chunk_outputs.append(wav_chunk)

            out = self._stitch_chunks(boundaries, chunk_outputs)

        # Restore the input's original loudness (before RMS envelope match so
        # rms_mix_rate stays the final envelope step).
        if original_loudness is not None and restore_loudness:
            out = _restore_loudness(out, self.sr, original_loudness)
        # RVC pipeline: optional output RMS envelope match to the input wav.
        if rms_mix_rate != 1.0:
            out = _change_rms(feats["wav"], feats["sr"], out, self.sr, rms_mix_rate)
        return out

    def _protect_units(
        self,
        units: torch.Tensor,
        units_src: torch.Tensor,
        rms: torch.Tensor,
        protect: float,
    ) -> torch.Tensor:
        """Blend pre-enhancement (source) units back on speech frames.

        RVC pipeline formula: pitchff=1 on voiced frames (keep enhanced units),
        pitchff=protect on unvoiced frames (mix source back). RVC's f0 is
        interpolated so pitchf==0 only in digital silence; we use rms > 1e-4
        as the speech-frame criterion instead (unvoiced consonants have
        energy, so they are covered). Silence frames keep enhanced units.
        """
        pitchff = torch.where(
            rms > 1e-4, torch.full_like(rms, protect), torch.ones_like(rms)
        )
        pitchff = pitchff.unsqueeze(-1)  # [T, 1]
        return units * pitchff + units_src * (1 - pitchff)

    def _find_silence_boundaries(
        self,
        rms: torch.Tensor | None,
        T: int,
        max_chunk_frames: int,
        crossfade_frames: int,
    ) -> list[tuple[int, int]]:
        """Find [start, end) frame boundaries that split audio at silence.

        Strategy:
        1. If RMS available, find silence frames (energy below adaptive threshold).
        2. Greedily accumulate segments up to max_chunk_frames.
        3. When a segment would exceed max, prefer cutting at the nearest
           silence frame before the limit; if none, force-cut at the lowest-
           energy frame in the last 1s (these force-cuts use crossfade).

        Returns list of (start, end) covering [0, T) with small overlaps at
        force-cut boundaries (silence cuts have zero overlap).
        """
        min_silence_frames = int(0.15 * self.sr / self.hop)  # 150ms = silence
        # Adaptive silence threshold: 10% of the 30th percentile RMS (robust
        # to a few loud peaks). Falls back to absolute -40dB if all loud.
        if rms is not None and T > 0:
            rms_np = rms.float().cpu().numpy()
            p30 = float(np.percentile(rms_np, 30))
            silence_thresh = max(p30 * 0.10, 1e-4)
            is_silence = rms_np < silence_thresh
        else:
            is_silence = np.zeros(T, dtype=bool)

        # Mark valid silence-cut centers: frames in runs of silence >= min length.
        silence_centers = np.zeros(T, dtype=bool)
        if is_silence.any():
            # Find runs of consecutive silence.
            idx = 0
            while idx < T:
                if is_silence[idx]:
                    run_start = idx
                    while idx < T and is_silence[idx]:
                        idx += 1
                    run_len = idx - run_start
                    if run_len >= min_silence_frames:
                        # Mark the center frame of this silence run as a cut point.
                        center = run_start + run_len // 2
                        silence_centers[center] = True
                else:
                    idx += 1

        boundaries: list[tuple[int, int]] = []
        start = 0
        while start < T:
            ideal_end = start + max_chunk_frames
            if ideal_end >= T:
                boundaries.append((start, T))
                break
            # Look for a silence center in [start + 0.5*max, ideal_end].
            search_lo = start + max_chunk_frames // 2
            search_hi = ideal_end
            silence_in_window = np.where(silence_centers[search_lo:search_hi])[0]
            if len(silence_in_window) > 0:
                # Pick the silence center closest to ideal_end (largest segment
                # without exceeding max).
                cut = search_lo + int(silence_in_window[-1]) + 1
                boundaries.append((start, cut))
                start = cut  # no overlap at silence cuts
            else:
                # No silence: force-cut at lowest-energy frame near the end.
                # Use a 1s window before ideal_end to find the quietest point.
                win = max(1 * self.sr // self.hop, crossfade_frames)
                lo = max(start, ideal_end - win)
                if rms is not None:
                    seg_rms = rms[lo:ideal_end].float().cpu().numpy()
                    rel = int(np.argmin(seg_rms))
                    cut = lo + rel
                else:
                    cut = ideal_end - crossfade_frames
                boundaries.append((start, cut))
                start = cut - crossfade_frames  # overlap for crossfade

        return boundaries

    def _run_model(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        rms: torch.Tensor | None,
        uv: torch.Tensor,
        spk: int,
        shallow_diffusion: bool = False,
        k_step: int = 100,
        diffusion_method: str = "ddim",
        diffusion_speedup: int = 10,
        t_start: float = 0.5,
        steps: int | None = None,
        noice_scale: float = 0.4,
        predict_f0: bool = False,
        formant_shift: float = 0.0,
        ds_cfg_strength: float | None = None,
        spk_cfg_strength: float | None = None,
        skip_cfg_strength: float | None = None,
        vocal_register_shift: float = 0.0,
    ) -> np.ndarray:
        """Run the backend model on one contiguous feature segment."""
        T = units.shape[0]
        spk_id = torch.zeros(1, dtype=torch.long, device=self.device) + spk
        frame_len = torch.tensor([T], device=self.device, dtype=torch.long)

        # pc vocal register shift: the model sees a shifted f0 (so it renders
        # a different register) while the vocoder sees the original f0 (so the
        # pitch stays the same). Only meaningful for mel-domain backends whose
        # mel is decoded by the (pc-aug) NSF-HiFiGAN vocoder; rvc/sovits are
        # wav-domain (no separate vocoder step) and ignore this.
        f0_for_model = f0
        if vocal_register_shift != 0.0 and self.backend_name in (
            "ddsp6", "diffusion", "reflow_vae", "rift_dit"
        ):
            f0_for_model = f0 / (2.0 ** (vocal_register_shift / 12.0))

        batch = SVCBatch(
            units=units.unsqueeze(0),          # [1, T, D]
            f0=f0_for_model.unsqueeze(0),      # [1, T] (shifted for register shift)
            mel=torch.zeros(1, T, self.n_mels, device=self.device, dtype=units.dtype),
            wav=torch.zeros(1, 1, T * self.hop, device=self.device, dtype=units.dtype),
            rms=rms.unsqueeze(0) if rms is not None else None,
            spk_id=spk_id,
            uv=uv.unsqueeze(0),
            aug_shift=(
                torch.tensor([[formant_shift]], device=self.device, dtype=units.dtype)
                if formant_shift != 0.0 else None
            ),
            pitch_token=None,
            frame_len=frame_len,
            key=None,
        )

        ac = (torch.autocast("cuda", dtype=self._autocast_dtype)
              if self._autocast_dtype is not None else _nullcontext())
        with ac:
            if self.backend_name == "rift_dit":
                src_mel = torch.zeros(1, T, self.n_mels, device=self.device, dtype=units.dtype)
                batch.mel = src_mel
                _ds = self.cfg_ds_strength if ds_cfg_strength is None else ds_cfg_strength
                _spk = self.cfg_spk_strength if spk_cfg_strength is None else spk_cfg_strength
                _skip = self.cfg_skip_strength if skip_cfg_strength is None else skip_cfg_strength
                bad_cvec = None
                if _ds > 1e-5:
                    bad_cvec = _resample_units(batch.units[0, ::2, :], T).unsqueeze(0)
                mel = self.module.sample(
                    batch,
                    ds_cfg_strength=_ds,
                    spk_cfg_strength=_spk,
                    skip_cfg_strength=_skip,
                    cfg_rescale=self.cfg_rescale,
                    bad_cvec=bad_cvec,
                    steps=steps,
                )
                if isinstance(mel, tuple):
                    mel = mel[0]
                wav = self.vocoder.infer(mel.float(), f0.float().unsqueeze(0))

            elif self.backend_name in ("ddsp6", "diffusion", "reflow_vae"):
                mel = self.module.sample(batch, steps=steps)
                wav = self.vocoder.infer(mel.float(), f0.float().unsqueeze(0))

            elif self.backend_name == "sovits":
                f0_interp = interp_unvoiced_f0(batch.f0[0]).unsqueeze(0)
                wav, _ = self.module.generator.infer(
                    batch.units.transpose(1, 2),
                    f0_interp,
                    batch.uv,
                    batch.spk_id.unsqueeze(-1),
                    vol=batch.rms,
                    noice_scale=noice_scale,
                    predict_f0=predict_f0,
                )

            elif self.backend_name == "rvc":
                pitch_token = self.module.generator.f0_to_coarse(batch.f0.float())
                wav, *_ = self.module.generator.infer(
                    batch.units,
                    batch.frame_len,
                    pitch_token,
                    batch.f0,
                    batch.spk_id,
                )

            else:  # pragma: no cover
                raise ValueError(self.backend_name)

        if shallow_diffusion and self.backend_name in ("sovits", "rvc"):
            return self._enhance_wav(
                units, f0, rms, spk, wav[0, 0].float(),
                k_step=k_step, diffusion_method=diffusion_method,
                diffusion_speedup=diffusion_speedup, t_start=t_start,
            )

        return wav[0, 0].float().cpu().numpy()

    def _enhance_wav(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        rms: torch.Tensor | None,
        spk: int,
        gen_wav: torch.Tensor,
        k_step: int = 100,
        diffusion_method: str = "ddim",
        diffusion_speedup: int = 10,
        t_start: float = 0.5,
    ) -> np.ndarray:
        """Dispatch to the loaded shallow enhancer (diffusion or reflow).

        Handles the RVC sample-rate bridge: RVC runs at 48k/32k/40k while
        the enhancers operate at 44.1k/hop512.  The input wav is resampled
        to the enhancer domain and the output back to the backend rate.
        """
        enh = self._shallow_enhancer
        enh_sr = enh.sr  # enhancer domain sample rate (44100)
        backend_sr = self.sr

        if self.reflow_enhancer is not None:
            # ---- ReflowEnhancer (DDSP6 RectifiedFlow, t_start shallow flow)
            wav_in = gen_wav
            f0_in = f0
            units_in = units
            rms_in = rms
            if backend_sr != enh_sr:
                wav_in = _resample_wav(gen_wav.cpu().numpy(), backend_sr, enh_sr)
                wav_in = torch.from_numpy(wav_in).to(gen_wav.dtype).to(self.device)
                # align frame-domain tensors (units/f0/rms) to enhancer frames
                enh_frames = wav_in.shape[-1] // enh.hop
                f0_in = _resample_f0(f0, enh_frames)
                units_in = _resample_units(units, enh_frames)
                if rms_in is not None:
                    rms_in = _resample_f0(rms_in, enh_frames)
            enh_out = self.reflow_enhancer.enhance(
                units_in, f0_in, rms_in, spk, wav_in,
                t_start=t_start,
            )
            if backend_sr != enh_sr:
                enh_out = _resample_wav(enh_out, enh_sr, backend_sr)
            return enh_out.astype(np.float32)

        # ---- DiffusionEnhancer (Diffusion-SVC, k-step DDPM denoising)
        # operates in the enhancer domain (44.1k). For rvc at 48k/32k/40k the
        # synthesized wav AND the frame-domain tensors (units/f0/rms) must all
        # be realigned to the enhancer frame grid (enh_sr/enh.hop), otherwise T
        # (backend frames) != mel frames extracted from the 44.1k wav and the
        # enhancer zero-pads the tail while units/f0 stay on the backend grid —
        # silently corrupting the shallow-diffusion conditioning.
        wav_in = gen_wav
        f0_in = f0
        units_in = units
        rms_in = rms
        if backend_sr != enh_sr:
            wav_in = _resample_wav(gen_wav.cpu().numpy(), backend_sr, enh_sr)
            wav_in = torch.from_numpy(wav_in).to(gen_wav.dtype).to(self.device)
            enh_frames = wav_in.shape[-1] // enh.hop
            f0_in = _resample_f0(f0, enh_frames)
            units_in = _resample_units(units, enh_frames)
            if rms_in is not None:
                rms_in = _resample_f0(rms_in, enh_frames)
        enh_out = self.diffusion_enhancer.enhance(
            units_in, f0_in, rms_in, spk, wav_in,
            k_step=k_step,
            infer_speedup=diffusion_speedup,
            method=diffusion_method,
        )
        if backend_sr != enh_sr:
            enh_out = _resample_wav(enh_out, enh_sr, backend_sr)
        return enh_out.astype(np.float32)

    def _stitch_chunks(
        self,
        boundaries: list[tuple[int, int]],
        chunks: list[np.ndarray],
    ) -> np.ndarray:
        """Stitch wav chunks according to their frame boundaries.

        Adjacent boundaries either abut (silence cut: prev_end == next_start,
        concatenate directly) or overlap (force cut: prev_end > next_start by
        overlap_frames, apply linear crossfade on the wav-domain overlap).
        """
        out = chunks[0].copy()
        for i in range(1, len(chunks)):
            cur = chunks[i]
            prev_end = boundaries[i - 1][1]
            cur_start = boundaries[i][0]
            if cur_start >= prev_end:
                # Abutting (silence cut): direct concatenation.
                out = np.concatenate([out, cur])
            else:
                # Overlapping (force cut): linear crossfade.
                overlap_frames = prev_end - cur_start
                overlap = min(overlap_frames * self.hop, len(out), len(cur))
                if overlap <= 0:
                    out = np.concatenate([out, cur])
                else:
                    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
                    tail = out[-overlap:] * (1 - ramp)
                    head = cur[:overlap] * ramp
                    out[-overlap:] = tail + head
                    out = np.concatenate([out, cur[overlap:]])
        return out.astype(np.float32)

    def convert_file(
        self,
        input_wav: str | Path,
        output_wav: str | Path,
        resample_sr: int | None = None,
        **kwargs,
    ) -> None:
        """Convert and write the result to a wav file.

        resample_sr: optional output sample rate; None (default) keeps the
            backend sample rate. The wav is resampled with soxr before saving.
        """
        wav = self.convert(input_wav, **kwargs)
        sr = self.sr
        if resample_sr is not None and resample_sr != self.sr:
            wav = _resample_wav(wav, self.sr, resample_sr)
            sr = resample_sr
        save_audio(str(output_wav), wav, sr=sr)
