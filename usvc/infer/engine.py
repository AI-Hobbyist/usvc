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
from .features import InferFeatures

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
    ) -> np.ndarray:
        """Convert a wav file; returns float32 wav at the backend sample rate.

        Long audio is split at silence boundaries to bound GPU memory and
        avoid artifacts at chunk edges. Silence is detected via RMS energy;
        chunks are inferred independently and concatenated. If a speech
        segment exceeds the max chunk size, it is force-split at the lowest-
        energy frame with a short crossfade.
        """
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
            return self._run_model(units, f0, rms, uv, spk).astype(np.float32)

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
            wav_chunk = self._run_model(chunk_units, chunk_f0, chunk_rms, chunk_uv, spk)
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

        else:  # pragma: no cover
            raise ValueError(self.backend_name)

        return wav[0, 0].float().cpu().numpy()

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
