"""Preprocess engine: dedupe scheduling + shared 16kHz cache (DSP-002 §4).

Given one or more FeatureSpec, computes the union of required features, skipping
any whose config hash is already complete in meta.json. Identical configs across
backends collapse to a single computation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from ..data.feature_config import FeatureConfig, FeatureSpec
from ..features.f0 import F0Extractor
from ..features.mel import MelExtractor
from ..features.rms import RmsExtractor
from ..features.units import UnitsExtractor
from ..io.audio import SharedAudio16kCache, load_audio
from .meta import DatasetMeta, load_meta, save_meta

__all__ = ["PreprocessEngine", "prepare_audio_dir"]


def prepare_audio_dir(
    src_dir: str | Path,
    dataset_dir: str | Path,
    *,
    sr: int,
    extensions: tuple[str, ...] = ("wav", "flac", "mp3", "ogg"),
    also_16k: bool = True,
) -> list[tuple[str, str]]:
    """Stage 0: resample/normalize source audio.

    Writes dataset_sr version to audio/{spk}/{file}.wav (for mel/f0/rms/spec)
    and, if ``also_16k``, a 16kHz version to audio_16k/{spk}/{file}.wav (for
    units content encoders). Pre-staging the 16k version eliminates the
    per-clip soxr_hq resample at feature-extraction time (the measured 70%
    bottleneck), at the cost of ~37% extra disk (16k audio is small).

    Resampling is parallelized across processes (librosa holds the GIL, so
    threads give no speedup; processes do).
    """
    from concurrent.futures import ProcessPoolExecutor

    src = Path(src_dir)
    out = Path(dataset_dir) / "audio"
    out16 = Path(dataset_dir) / "audio_16k"
    discovered: list[tuple[str, str]] = []
    jobs = []
    for ext in extensions:
        for p in sorted(src.rglob(f"*.{ext}")):
            speaker = p.relative_to(src).parts[0] if len(p.relative_to(src).parts) > 1 else "_root"
            file_name = p.stem + ".wav"
            discovered.append((speaker, file_name))
            jobs.append((str(p), str(out / speaker / file_name), str(out16 / speaker / file_name) if also_16k else None, sr))

    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_stage_one, jobs, chunksize=4))
    return discovered


def _stage_one(job):
    """Worker: resample source -> dataset_sr + 16k versions. Module-level for picklability."""
    src_path, out_path, out16_path, sr = job
    import soundfile as sf
    import librosa
    from pathlib import Path

    wav, orig_sr = sf.read(src_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # dataset_sr version (mel/f0/rms/spec — quality-insensitive, use fast resample)
    if orig_sr != sr:
        wav_sr = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr, res_type="soxr_lq")
    else:
        wav_sr = wav
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wav_sr.astype("float32"), sr, subtype="FLOAT")
    # 16k version (content encoders — sensitive, use soxr_hq)
    if out16_path:
        if orig_sr != 16000:
            wav16 = librosa.resample(wav, orig_sr=orig_sr, target_sr=16000, res_type="soxr_hq")
        else:
            wav16 = wav
        Path(out16_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out16_path, wav16.astype("float32"), 16000, subtype="FLOAT")


class PreprocessEngine:
    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.audio_dir = self.dataset_dir / "audio"
        mpath = self.dataset_dir / "meta.json"
        if mpath.is_file():
            self.meta: DatasetMeta = load_meta(self.dataset_dir)
        else:
            # caller must init meta with sr/hop/f0_min/f0_max before features
            raise FileNotFoundError(
                f"meta.json not found in {self.dataset_dir}. "
                "Call svc.data.meta.init_dataset_meta first."
            )

    # ------------------------------------------------------------------
    def run(
        self,
        specs: list[FeatureSpec],
        *,
        device: str = "cpu",
        force: bool = False,
        batch_size: int = 8,
        dtype: str = "fp32",
    ) -> None:
        # 1. Union + dedupe: same FeatureConfig.hash() collapses to one task.
        todo: dict[str, FeatureConfig] = {}
        for spec in specs:
            for fc in spec.all_required():
                h = fc.hash()
                if h in todo:
                    continue
                if not force and self.meta.is_hash_registered(fc):
                    continue
                todo[h] = fc

        if not todo:
            return

        cache16k = SharedAudio16kCache(target_sr=16000)

        # 2. Group by kind so per-kind extractors are built once.
        by_kind: dict[str, list[FeatureConfig]] = {}
        for fc in todo.values():
            by_kind.setdefault(fc.kind, []).append(fc)

        for kind, fcs in by_kind.items():
            for fc in tqdm(fcs, desc=f"feature:{kind}"):
                self._run_one_feature(fc, cache16k, device=device, batch_size=batch_size, dtype=dtype)

        save_meta(self.meta)
        cache16k.clear()

    # ------------------------------------------------------------------
    def _run_one_feature(
        self, fc: FeatureConfig, cache16k: SharedAudio16kCache, *, device: str, batch_size: int = 8, dtype: str = "fp32"
    ) -> None:
        sr, hop = self.meta.sr, self.meta.hop
        out_root = self.dataset_dir / fc.path_tag
        out_root.mkdir(parents=True, exist_ok=True)

        extractor = self._build_extractor(fc, sr, hop, device=device, dtype=dtype)

        # Gather pending clips (skip already-computed).
        pending: list[tuple[str, str, int]] = []  # (speaker, file_name, audio_len)
        count = 0
        for a in self.meta.audios:
            out_path = out_root / a.speaker / (Path(a.file_name).stem + ".pt")
            if out_path.is_file():
                count += 1
                continue
            pending.append((a.speaker, a.file_name, a.audio_samples))

        if not pending:
            self.meta.register(fc, count)
            return

        # Dispatch by compute placement:
        #  - rms: pure windowed-mean, CPU-vectorized is fastest (no GPU benefit,
        #    and ProcessPool is worse due to torch thread contention).
        #  - mel: batched GPU stft (~130x faster than per-clip via fused kernel).
        #  - f0(fcpe): batched GPU — fcpe tolerates padding well (<1Hz at 10%),
        #    so bucket-batch gives ~2.7x over per-clip. RMVPE falls back to
        #    single-clip pipelined (its conv is padding-sensitive).
        #  - units: neural model, GPU-only via the pipelined path (conv is too
        #    padding-sensitive for batch — needs <0.5% which limits batch to ~3).
        #  - aug_shift: cheap per-file scalar (random keyshift, reproducible).
        #  - aug_mel: per-file keyshift (window size differs per shift) —
        #    cannot share one fused stft, so sequential per-clip extraction.
        if fc.kind == "aug_shift":
            count += self._run_aug_shift(fc, pending, out_root)
        elif fc.kind == "aug_mel":
            count += self._run_aug_mel(fc, extractor, pending, out_root, sr)
        elif fc.kind == "rms":
            count += self._run_cpu_parallel(fc, extractor, pending, out_root, sr)
        elif fc.kind == "mel":
            count += self._run_mel_batch(fc, extractor, pending, out_root, sr)
        elif fc.kind == "f0" and extractor.extractor == "fcpe":
            count += self._run_f0_batch(fc, extractor, pending, out_root, sr)
        elif device.startswith("cuda"):
            count += self._run_pipelined(fc, extractor, pending, out_root, sr)
        else:
            count += self._run_sequential(fc, extractor, pending, cache16k, out_root, sr)

        self.meta.register(fc, count)

    def _run_aug_shift(self, fc, pending, out_root) -> int:
        """Generate one random keyshift (semitones) per clip, stored as a
        scalar [1] tensor. Deterministic: seeded by sha256(spk:stem) so
        re-preprocessing reproduces identical shifts.
        """
        import hashlib
        import random
        from pathlib import Path

        lo, hi = fc.params.get("range", [-5.0, 5.0])
        done = 0
        with tqdm(total=len(pending), desc=f"aug_shift@{fc.hash()}", leave=False) as pbar:
            for spk, fn, _ in pending:
                stem = Path(fn).stem
                seed = int(hashlib.sha256(f"aug_shift:{spk}:{stem}".encode()).hexdigest()[:8], 16)
                shift = random.Random(seed).uniform(lo, hi)
                out_path = out_root / spk / (stem + ".pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.tensor([shift], dtype=torch.float32), out_path)
                done += 1
                pbar.update(1)
        return done

    def _run_aug_mel(self, fc, extractor, pending, out_root, sr) -> int:
        """Extract keyshifted mel per clip using its recorded aug_shift scalar.

        The per-file keyshift lives under aug_shift/{hash}/{spk}/{stem}.pt.
        Sequential path: each clip may have a different keyshift (scaled
        window), so no fused batch stft is possible.
        """
        from pathlib import Path

        shift_entries = self.meta.features.get("aug_shift", [])
        if shift_entries:
            shift_hash = shift_entries[0]["hash"]
        else:
            # Fallback (e.g. augmented-only preprocess): hash is deterministic.
            shift_hash = FeatureConfig("aug_shift", {"range": [-5.0, 5.0]}).hash()
        shift_root = self.dataset_dir / "aug_shift" / shift_hash

        done = 0
        with tqdm(total=len(pending), desc=f"aug_mel@{fc.hash()}", leave=False) as pbar:
            for spk, fn, _ in pending:
                stem = Path(fn).stem
                shift_path = shift_root / spk / (stem + ".pt")
                if not shift_path.is_file():
                    pbar.update(1)
                    continue  # no aug_shift for this clip -> skip
                shift = torch.load(shift_path, map_location="cpu", weights_only=True).item()
                wav = torch.from_numpy(
                    load_audio(self.audio_dir / spk / fn, sr=sr, quality="fast")
                ).float()
                feat = extractor.extract(wav, keyshift=shift)
                out_path = out_root / spk / (stem + ".pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat.cpu().contiguous(), out_path)
                done += 1
                pbar.update(1)
        return done

    def _run_sequential(self, fc, extractor, pending, cache16k, out_root, sr) -> int:
        from concurrent.futures import ThreadPoolExecutor

        # Even on CPU, prefetching audio in parallel threads helps (librosa
        # resample releases the GIL during its C inner loop).
        def _load(item):
            spk, fn, _ = item
            if fc.kind == "units":
                return cache16k.get(spk, fn, self.audio_dir, sr)
            return torch.from_numpy(load_audio(self.audio_dir / spk / fn, sr=sr, quality="fast")).float()

        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            pbar = tqdm(total=len(pending), desc=f"{fc.kind}@{fc.hash()}", leave=False)
            for item, wav in zip(pending, pool.map(_load, pending)):
                spk, fn, _ = item
                feat = extractor.extract(wav)
                out_path = out_root / spk / (Path(fn).stem + ".pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat.cpu().contiguous(), out_path)
                done += 1
                pbar.update(1)
            pbar.close()
        return done

    def _run_cpu_parallel(self, fc, extractor, pending, out_root, sr) -> int:
        """CPU features (mel/rms/spec).

        Runs single-threaded in the main process. Profiling showed ProcessPool
        here is a net loss: torch's internal threading (MKL/OpenMP) already
        uses all cores per call, so forking extra processes causes thread
        contention (measured 96ms/clip in pool vs 19ms/clip single-thread for
        mel). The extractor is built once and reused.
        """
        from pathlib import Path
        import torch
        from ..io.audio import load_audio

        done = 0
        with tqdm(total=len(pending), desc=f"{fc.kind}@{fc.hash()}", leave=False) as pbar:
            for spk, fn, _ in pending:
                wav = load_audio(self.audio_dir / spk / fn, sr=sr, quality="fast")
                feat = extractor.extract(torch.from_numpy(wav).float())
                out_path = out_root / spk / (Path(fn).stem + ".pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat.cpu().contiguous(), out_path)
                done += 1
                pbar.update(1)
        return done

    def _run_mel_batch(self, fc, extractor, pending, out_root, sr) -> int:
        """Batched GPU mel: fused stft over N clips per call.

        Clips are loaded on CPU threads (soundfile releases the GIL), then
        grouped into fixed-size batches and sent to the GPU as a single [B, T]
        tensor for one torch.stft call. The per-clip launch overhead (18.5ms
        × 1540 = 28s) collapses to ~0.15ms/clip (~0.3s total).
        """
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        import torch
        from ..io.audio import load_audio

        BATCH = 64

        def _load(item):
            spk, fn, _ = item
            return torch.from_numpy(load_audio(self.audio_dir / spk / fn, sr=sr, quality="fast")).float()

        done = 0
        with tqdm(total=len(pending), desc=f"mel@{fc.hash()} (batch)", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=4) as pool:
                wavs_iter = pool.map(_load, pending)
                buf_wavs = []
                buf_items = []
                for item, wav in zip(pending, wavs_iter):
                    buf_items.append(item)
                    buf_wavs.append(wav)
                    if len(buf_wavs) >= BATCH:
                        feats = extractor.extract_batch(buf_wavs)
                        for (spk, fn, _), feat in zip(buf_items, feats):
                            out_path = out_root / spk / (Path(fn).stem + ".pt")
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(feat.cpu().contiguous(), out_path)
                            done += 1
                            pbar.update(1)
                        buf_wavs.clear(); buf_items.clear()
                if buf_wavs:
                    feats = extractor.extract_batch(buf_wavs)
                    for (spk, fn, _), feat in zip(buf_items, feats):
                        out_path = out_root / spk / (Path(fn).stem + ".pt")
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(feat.cpu().contiguous(), out_path)
                        done += 1
                        pbar.update(1)
        return done

    def _run_f0_batch(self, fc, extractor, pending, out_root, sr) -> int:
        """Bucket-batched FCPE: group near-equal-length clips, forward together.

        FCPE's mel-extraction uses center-padding, so trailing zero-pad has
        minimal boundary bleed (<1Hz error at 10% padding). This lets us use a
        loose 10% padding tolerance and still build large batches, giving ~2.7x
        over per-clip (measured 4.2ms vs 11.5ms per clip).
        """
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        import torch
        from ..features.batching import bucket_batch
        from ..io.audio import load_audio

        BATCH = 64
        PAD_TOL = 0.10  # FCPE tolerates 10% padding with <1Hz error

        # Bucket by audio length (proxy for f0 native frame count).
        lengths = [p[2] for p in pending]
        batches = bucket_batch(lengths, batch_size=BATCH, max_pad_ratio=PAD_TOL)

        def _load(item):
            spk, fn, _ = item
            return torch.from_numpy(load_audio(self.audio_dir / spk / fn, sr=sr, quality="fast")).float()

        done = 0
        with tqdm(total=len(pending), desc=f"f0@{fc.hash()} (batch)", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=4) as pool:
                # Prefetch all clips (soundfile releases GIL).
                wavs = list(pool.map(_load, pending))
            for bi in batches:
                bwavs = [wavs[i].to(extractor.device) for i in bi]
                feats = extractor.extract_batch(bwavs)
                for idx, feat in zip(bi, feats):
                    spk, fn, _ = pending[idx]
                    out_path = out_root / spk / (Path(fn).stem + ".pt")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(feat.cpu().contiguous(), out_path)
                    done += 1
                    pbar.update(1)
        return done

    def _run_pipelined(self, fc, extractor, pending, out_root, sr) -> int:
        """GPU pipeline: prefetch audio on CPU *processes* while GPU runs forward.

        Measured bottleneck on A10G + 48k corpus:
          CPU resample (soxr_hq, units): 50ms/clip single-core, ~4ms with 4 procs
          GPU forward:                   8ms/clip  -> the new ceiling
        ProcessPool is mandatory here: librosa/numba holds the GIL, so
        ThreadPool gives zero parallelism for resampling (measured 12x slower).
        """
        import queue
        import threading
        from concurrent.futures import ProcessPoolExecutor

        PREFETCH = 4  # audio clips buffered ahead of the GPU
        kind = fc.kind
        audio_dir = self.audio_dir

        # module-level worker fn (must be picklable for ProcessPool)
        import usvc.data.engine as _eng

        def produce(q):
            with ProcessPoolExecutor(max_workers=PREFETCH) as pool:
                futs = {
                    pool.submit(_eng._proc_load, item, kind, str(audio_dir), sr): item
                    for item in pending
                }
                for fut in futs:
                    item = futs[fut]
                    wav = fut.result()
                    q.put((item, wav))
            q.put(None)

        q: queue.Queue = queue.Queue(maxsize=PREFETCH)
        producer = threading.Thread(target=produce, args=(q,))
        producer.start()

        done = 0
        pbar = tqdm(total=len(pending), desc=f"{fc.kind}@{fc.hash()} (pipeline)", leave=False)
        try:
            while True:
                blob = q.get()
                if blob is None:
                    break
                (spk, fn, _), wav = blob
                feat = extractor.extract(wav)
                out_path = out_root / spk / (Path(fn).stem + ".pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat.cpu().contiguous(), out_path)
                done += 1
                pbar.update(1)
        finally:
            pbar.close()
            producer.join(timeout=1)
        return done

    def _build_extractor(self, fc: FeatureConfig, sr: int, hop: int, *, device: str, dtype: str = "fp32"):
        if fc.kind in ("mel", "aug_mel"):
            return MelExtractor(fc, sr, hop).to(device)
        if fc.kind == "aug_shift":
            return None  # pure per-file scalar generation; no model needed
        if fc.kind == "f0":
            return F0Extractor(fc, sr, hop, device=device, dtype=dtype)
        if fc.kind == "rms":
            return RmsExtractor(fc, sr, hop)  # CPU — vectorized unfold
        if fc.kind == "units":
            return UnitsExtractor(fc, sr, hop, device=device, dtype=dtype)
        if fc.kind == "spec":
            from ..features.spec import SpecExtractor
            return SpecExtractor(fc, sr, hop).to(device)
        raise ValueError(f"unknown feature kind: {fc.kind}")


# Module-level worker for ProcessPool (must be picklable — no closures).
def _proc_load(item, kind: str, audio_dir: str, sr: int):
    """Load one clip in a worker process. Reads pre-staged files (no resample).

    units: read audio_16k/{spk}/{fn} (pre-staged 16k, soxr_hq at stage time).
    others: read audio/{spk}/{fn} at dataset sr.
    Worker holds no model state; returns a torch tensor (CPU).
    """
    import torch
    from pathlib import Path
    import soundfile as sf

    spk, fn, _ = item
    if kind == "units":
        # Prefer pre-staged 16k; fall back to runtime resample if absent (legacy).
        p16 = Path(audio_dir).parent / "audio_16k" / spk / fn
        if p16.is_file():
            wav, _ = sf.read(str(p16), dtype="float32")
            return torch.from_numpy(wav).float()
        # fallback: runtime resample from staged dataset-sr file
        wav, file_sr = sf.read(str(Path(audio_dir) / spk / fn))
        if file_sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=16000, res_type="soxr_hq")
        return torch.from_numpy(wav).float()
    return torch.from_numpy(_read_at_sr(Path(audio_dir) / spk / fn, sr)).float()


def _read_at_sr(path, sr):
    import soundfile as sf
    wav, file_sr = sf.read(str(path), dtype="float32")
    if file_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr, res_type="soxr_lq")
    return wav



