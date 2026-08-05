"""Offline inference engine: wav -> features -> backend model -> wav.

Supports all 6 backends:
  - mel-domain (rift_dit / ddsp6 / diffusion / reflow_vae): model samples mel,
    shared NSF-HiFiGAN vocoder converts mel+f0 -> wav.
  - wav-domain (sovits / rvc): the generator itself decodes waveform
    (VITS + built-in NSF), no external vocoder needed.

Usage (CLI):  usvc-infer input.wav --config configs/sovits.yaml --ckpt ck.ckpt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..backends import get_backend
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
    """Align a [T] f0 trajectory to ``target_frames`` via linear interpolation."""
    T = f0.shape[0]
    if T == target_frames:
        return f0
    f0_np = f0.detach().float().cpu().numpy()
    src = np.arange(T, dtype=np.float32)
    dst = np.linspace(0, T - 1, target_frames, dtype=np.float32)
    out = np.interp(dst, src, f0_np).astype(np.float32)
    return torch.from_numpy(out).to(f0.device).to(f0.dtype)


def _resample_units(units: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Align [T, D] content features to ``target_frames`` via linear interp."""
    T = units.shape[0]
    if T == target_frames:
        return units
    u = units.detach().float().cpu().numpy()  # [T, D]
    src = np.arange(T, dtype=np.float32)
    dst = np.linspace(0, T - 1, target_frames, dtype=np.float32)
    out = np.stack([np.interp(dst, src, u[:, d]) for d in range(u.shape[1])], axis=1)
    return torch.from_numpy(out.astype(np.float32)).to(units.device).to(units.dtype)

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
        sd = ckpt.get("state_dict", ckpt)
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
    ) -> None:
        self.cfg = TrainConfig.from_yaml(config_path)
        self.backend_name = self.cfg.backend
        if self.backend_name not in MEL_DOMAIN + WAV_DOMAIN:
            raise ValueError(f"unknown backend: {self.backend_name}")
        self.device = device
        self.half = half
        self.ckpt_path = str(ckpt_path)

        # Sample rate / hop for inference features (from config or defaults).
        opts = self.cfg.backend_opts
        self.sr = int(opts.get("sampling_rate", opts.get("sr", _DEFAULT_SR_HOP[self.backend_name][0])))
        self.hop = int(opts.get("hop_length", opts.get("block_size", opts.get("hop", _DEFAULT_SR_HOP[self.backend_name][1]))))
        self.n_mels = int(opts.get("n_mel_channels", opts.get("out_dims", opts.get("n_mels", 128))))

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
        self.module_half = half and self.backend_name != "diffusion"
        if self.module_half:
            self.module.half()
        self.module = self.module.to(device)

        # Feature spec drives inference extraction (same encoder as training).
        spec = backend_cls.feature_spec({k: v for k, v in opts.items() if k in (
            "units_kind", "units_ckpt", "f0_extractor", "n_mels", "fmin", "fmax",
        )})
        self.features = InferFeatures(
            spec, sr=self.sr, hop=self.hop, device=device,
            dtype="fp16" if self.module_half else "fp32",
        )

        # Vocoder for mel-domain backends (uses shared openvpi NSF-HiFiGAN).
        self.vocoder = None
        if self.backend_name in MEL_DOMAIN:
            if vocoder_dir is None:
                vocoder_dir = Path(__file__).resolve().parents[2] / "pretrained" / "nsf_hifigan"
            vocoder_dir = Path(vocoder_dir)
            self.vocoder = NSFHiFiGAN(
                vocoder_dir / "config.json",
                vocoder_dir / "model.ckpt",
                device=device,
                is_half=self.module_half,
            )

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
                    device=device, half=half, sr=self.sr, hop=self.hop,
                )

    @property
    def _shallow_enhancer(self):
        """Return the active shallow-diffusion enhancer (reflow preferred)."""
        if self.reflow_enhancer is not None:
            return self.reflow_enhancer
        return self.diffusion_enhancer

    def _load_weights(self, module) -> None:
        sd = _load_ckpt_state(self.ckpt_path)
        # Accept both bare-model ckpts and lightning ckpts (module-prefixed).
        missing, unexpected = module.load_state_dict(sd, strict=False)
        # strict=False tolerates optimizer/discriminator-only keys in lightning
        # ckpts (sovits/rvc have discriminator + generator keys; generator is
        # required). Filter expected suffixes: bare model ckpts may drop
        # discriminator, so only *required* model keys must be present.
        if self.backend_name in WAV_DOMAIN:
            gen_keys = [k for k in sd if k.startswith("generator.")]
            if not gen_keys:
                raise RuntimeError(
                    f"checkpoint has no 'generator.*' keys (sovits/rvc need the "
                    f"generator state); got {list(sd)[:5]}"
                )
        else:
            model_keys = [k for k in sd if k.startswith("model.")]
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
        """
        if shallow_diffusion and self._shallow_enhancer is None:
            raise ValueError(
                "shallow_diffusion=True requires diffusion_ckpt+diffusion_config "
                "or reflow_ckpt+reflow_config at engine construction (sovits/rvc)"
            )
        _t_start = self.t_start if t_start is None else float(t_start)
        feats = self.features.extract(input_wav)
        units = feats["units"].to(self.device)  # [T, dim]
        f0 = feats["f0"].to(self.device)  # [T]
        rms = feats["rms"].to(self.device) if feats["rms"] is not None else None
        uv = feats["uv"].to(self.device)

        if self.module_half:
            units = units.half()
            f0 = f0.half()
            if rms is not None:
                rms = rms.half()

        # Pitch shift (semitones).
        if trans != 0:
            f0 = f0 * (2.0 ** (trans / 12))

        # Feature retrieval / cluster enhancement (RVC / so-vits style).
        units = self._enhance_units(
            units, index_path, index_ratio, cluster_path, cluster_ratio
        )

        T = units.shape[0]
        # Max chunk size by backend: GAN backends (rvc/sovits) allocate O(T) wav
        # via NSF vocoder, so cap smaller. Flow backends (mel-domain) can go larger.
        max_chunk_sec = 30
        if self.backend_name in ("rvc", "sovits"):
            max_chunk_sec = 20
        max_chunk_frames = max_chunk_sec * self.sr // self.hop

        if T <= max_chunk_frames:
            return self._run_model(
                units, f0, rms, uv, spk,
                shallow_diffusion=shallow_diffusion, k_step=k_step,
                diffusion_method=diffusion_method,
                diffusion_speedup=diffusion_speedup,
                t_start=_t_start,
            ).astype(np.float32)

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
            )
            chunk_outputs.append(wav_chunk)

        return self._stitch_chunks(boundaries, chunk_outputs)

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
    ) -> np.ndarray:
        """Run the backend model on one contiguous feature segment."""
        T = units.shape[0]
        spk_id = torch.zeros(1, dtype=torch.long, device=self.device) + spk
        frame_len = torch.tensor([T], device=self.device, dtype=torch.long)
        batch = SVCBatch(
            units=units.unsqueeze(0),          # [1, T, D]
            f0=f0.unsqueeze(0),                # [1, T]
            mel=torch.zeros(1, T, self.n_mels, device=self.device, dtype=units.dtype),
            wav=torch.zeros(1, 1, T * self.hop, device=self.device, dtype=units.dtype),
            rms=rms.unsqueeze(0) if rms is not None else None,
            spk_id=spk_id,
            uv=uv.unsqueeze(0),
            aug_shift=None,
            pitch_token=None,
            frame_len=frame_len,
            key=None,
        )

        if self.backend_name == "rift_dit":
            src_mel = torch.zeros(1, T, self.n_mels, device=self.device, dtype=units.dtype)
            batch.mel = src_mel
            mel = self.module.sample(batch)
            if isinstance(mel, tuple):
                mel = mel[0]
            wav = self.vocoder.infer(mel.float(), f0.float().unsqueeze(0))

        elif self.backend_name in ("ddsp6", "diffusion", "reflow_vae"):
            mel = self.module.sample(batch)
            wav = self.vocoder.infer(mel.float(), f0.float().unsqueeze(0))

        elif self.backend_name == "sovits":
            wav, _ = self.module.generator.infer(
                batch.units.transpose(1, 2),  # [B, D, T]
                batch.f0,
                batch.uv,
                batch.spk_id.unsqueeze(-1),
                vol=batch.rms,  # [1, T] volume (only used if vol_embedding=True)
            )
            if shallow_diffusion:
                return self._enhance_wav(
                    units, f0, rms, spk, wav[0, 0].float(),
                    k_step=k_step, diffusion_method=diffusion_method,
                    diffusion_speedup=diffusion_speedup, t_start=t_start,
                )

        elif self.backend_name == "rvc":
            pitch_token = self.module.generator.f0_to_coarse(batch.f0.float())
            wav, *_ = self.module.generator.infer(
                batch.units,
                batch.frame_len,
                pitch_token,
                batch.f0,  # continuous f0 for the NSF source
                batch.spk_id,
            )
            if shallow_diffusion:
                return self._enhance_wav(
                    units, f0, rms, spk, wav[0, 0].float(),
                    k_step=k_step, diffusion_method=diffusion_method,
                    diffusion_speedup=diffusion_speedup, t_start=t_start,
                )

        else:  # pragma: no cover
            raise ValueError(self.backend_name)

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
        # operates in the backend domain (sovits is already 44.1k); for rvc
        # at 48k we still feed 44.1k since sovits/diffusion enhancers expect
        # 44.1k — the DiffusionEnhancer extracts its own mel at enh_sr.
        wav_in = gen_wav
        if backend_sr != enh_sr:
            wav_in = _resample_wav(gen_wav.cpu().numpy(), backend_sr, enh_sr)
            wav_in = torch.from_numpy(wav_in).to(gen_wav.dtype).to(self.device)
        enh_out = self.diffusion_enhancer.enhance(
            units, f0, rms, spk, wav_in,
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
        crossfade_frames, apply linear crossfade on the wav-domain overlap).
        """
        crossfade_samples = (2 * self.sr // self.hop) * self.hop
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

    def convert_file(self, input_wav: str | Path, output_wav: str | Path, **kwargs) -> None:
        """Convert and write the result to a wav file."""
        wav = self.convert(input_wav, **kwargs)
        save_audio(str(output_wav), wav, sr=self.sr)
