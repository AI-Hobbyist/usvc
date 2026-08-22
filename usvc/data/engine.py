"""Preprocess engine (DSP-002 §4): computes the union of required features across specs,
skipping any hash already complete in meta.json; identical configs collapse to one computation.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from ..data.feature_config import FeatureConfig
from ..features.f0 import F0Extractor
from ..features.mel import MelExtractor
from ..features.rms import RmsExtractor
from ..features.units import UnitsExtractor
from ..io.audio import SharedAudio16kCache, load_audio
from .meta import DatasetMeta, load_meta, save_meta


def prepare_audio_dir(src_dir, dataset_dir, *, sr, extensions=("wav", "flac", "mp3", "ogg"), also_16k=True):
    """Stage 0: stage dataset-sr copies to audio/{spk}/{f}.wav (mel/f0/rms/spec, soxr_lq) plus,
    if also_16k, audio_16k/ (units encoders, soxr_hq) — pre-staging removes the per-clip resample
    hot spot (~37% extra disk). Processes, not threads: librosa holds the GIL."""
    src, out, out16 = Path(src_dir), Path(dataset_dir) / "audio", Path(dataset_dir) / "audio_16k"
    discovered, jobs = [], []
    for ext in extensions:
        for p in sorted(src.rglob(f"*.{ext}")):
            rel = p.relative_to(src)
            spk = rel.parts[0] if len(rel.parts) > 1 else "_root"
            fn = p.stem + ".wav"
            discovered.append((spk, fn))
            jobs.append((str(p), str(out / spk / fn),
                         str(out16 / spk / fn) if also_16k else None, sr))
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_stage_one, jobs, chunksize=4))
    return discovered


def _stage_one(job):  # module-level: must stay picklable for ProcessPool
    import librosa
    import soundfile as sf

    src_path, out_path, out16_path, sr = job
    wav, orig_sr = sf.read(src_path)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    for path, target, res_type in ((out_path, sr, "soxr_lq"), (out16_path, 16000, "soxr_hq")):
        if path is None: continue
        w = wav if orig_sr == target else \
            librosa.resample(wav, orig_sr=orig_sr, target_sr=target, res_type=res_type)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, w.astype("float32"), target, subtype="FLOAT")


def _dump(out_root, spk, fn, feat):
    p = out_root / spk / (Path(fn).stem + ".pt")
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat.cpu().contiguous(), p)


def _load_wav(item, audio_dir, sr):  # dataset-sr clip for mel/f0/rms/spec/aug_mel
    spk, fn, _ = item
    return torch.from_numpy(load_audio(Path(audio_dir) / spk / fn, sr=sr, quality="fast")).float()


class PreprocessEngine:
    def __init__(self, dataset_dir):
        self.dataset_dir = d = Path(dataset_dir)
        self.audio_dir = d / "audio"
        if not (d / "meta.json").is_file():  # caller must init meta (sr/hop/f0 bounds) first
            raise FileNotFoundError(
                f"meta.json not found in {d}. Call svc.data.meta.init_dataset_meta first.")
        self.meta: DatasetMeta = load_meta(d)

    def run(self, specs, *, device="cpu", force=False, batch_size=8, dtype="fp32") -> None:
        todo: dict[str, FeatureConfig] = {}  # hash -> fc; identical configs dedupe here
        for spec in specs:
            for fc in spec.all_required():
                if fc.hash() not in todo and (force or not self.meta.is_hash_registered(fc)):
                    todo[fc.hash()] = fc
        if not todo: return
        cache16k = SharedAudio16kCache(target_sr=16000)
        by_kind: dict[str, list[FeatureConfig]] = {}
        for fc in todo.values(): by_kind.setdefault(fc.kind, []).append(fc)
        for kind, fcs in by_kind.items():
            for fc in tqdm(fcs, desc=f"feature:{kind}"):
                self._run_one(fc, cache16k, device=device, batch_size=batch_size, dtype=dtype)
        save_meta(self.meta)
        cache16k.clear()

    def _run_one(self, fc, cache16k, *, device, batch_size=8, dtype="fp32") -> None:
        sr, hop = self.meta.sr, self.meta.hop
        out_root = self.dataset_dir / fc.path_tag
        out_root.mkdir(parents=True, exist_ok=True)
        extractor = self._build_extractor(fc, sr, hop, device=device, dtype=dtype)
        pending, count = [], 0  # pending: (speaker, file_name, audio_samples)
        for a in self.meta.audios:
            if (out_root / a.speaker / (Path(a.file_name).stem + ".pt")).is_file(): count += 1
            else: pending.append((a.speaker, a.file_name, a.audio_samples))
        if not pending:
            self.meta.register(fc, count); return
        # Placement rationale (measured): rms = CPU windowed mean, ProcessPool loses to MKL/OpenMP
        # threads (96 vs 19ms/clip); mel = fused GPU stft over N clips (~130x launch overhead win);
        # f0-fcpe tolerates 10% pad (<1Hz bleed) -> bucket-batch ~2.7x, rmvpe conv is padding-
        # sensitive -> single-clip pipeline; units needs <0.5% pad (batch<=3) -> GPU-only pipeline;
        # aug_shift is a cheap per-file scalar; aug_mel windows vary per keyshift -> strictly serial.
        if fc.kind == "aug_shift": count += self._aug_shift(fc, pending, out_root)
        elif fc.kind == "aug_mel": count += self._aug_mel(fc, extractor, pending, out_root, sr)
        elif fc.kind == "mel": count += self._mel_batch(fc, extractor, pending, out_root, sr)
        elif fc.kind == "f0" and extractor.extractor == "fcpe":
            count += self._f0_batch(fc, extractor, pending, out_root, sr)
        elif fc.kind == "rms": count += self._sequential(fc, extractor, pending, cache16k, out_root, sr)
        elif device.startswith("cuda"): count += self._pipelined(fc, extractor, pending, out_root, sr)
        else: count += self._sequential(fc, extractor, pending, cache16k, out_root, sr)
        self.meta.register(fc, count)

    def _aug_shift(self, fc, pending, out_root):
        import hashlib
        import random

        lo, hi = fc.params.get("range", [-5.0, 5.0])
        done = 0
        with tqdm(total=len(pending), desc=f"aug_shift@{fc.hash()}", leave=False) as pbar:
            for spk, fn, _ in pending:
                # deterministic: sha256-seeded per (speaker, stem) so re-preprocess reproduces shifts
                seed = int(hashlib.sha256(f"aug_shift:{spk}:{Path(fn).stem}".encode()).hexdigest()[:8], 16)
                _dump(out_root, spk, fn, torch.tensor([random.Random(seed).uniform(lo, hi)], dtype=torch.float32))
                done += 1; pbar.update(1)
        return done

    def _aug_mel(self, fc, extractor, pending, out_root, sr):
        entries = self.meta.features.get("aug_shift", [])
        # fallback (augmented-only preprocess): the default-range hash is deterministic
        shift_hash = entries[0]["hash"] if entries else \
            FeatureConfig("aug_shift", {"range": [-5.0, 5.0]}).hash()
        shift_root = self.dataset_dir / "aug_shift" / shift_hash
        done = 0
        with tqdm(total=len(pending), desc=f"aug_mel@{fc.hash()}", leave=False) as pbar:
            for spk, fn, _ in pending:
                p = shift_root / spk / (Path(fn).stem + ".pt")
                if not p.is_file(): pbar.update(1); continue  # no aug_shift recorded -> skip
                shift = torch.load(p, map_location="cpu", weights_only=True).item()
                _dump(out_root, spk, fn,
                      extractor.extract(_load_wav((spk, fn, 0), self.audio_dir, sr), keyshift=shift))
                done += 1; pbar.update(1)
        return done

    def _sequential(self, fc, extractor, pending, cache16k, out_root, sr):
        # threads pay off even on CPU: soundfile/librosa release the GIL inside their C loops
        def load(item):
            if fc.kind == "units": return cache16k.get(item[0], item[1], self.audio_dir, sr)
            return _load_wav(item, self.audio_dir, sr)

        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool, \
                tqdm(total=len(pending), desc=f"{fc.kind}@{fc.hash()}", leave=False) as pbar:
            for (spk, fn, _), wav in zip(pending, pool.map(load, pending)):
                _dump(out_root, spk, fn, extractor.extract(wav)); done += 1; pbar.update(1)
        return done

    def _mel_batch(self, fc, extractor, pending, out_root, sr):
        BATCH = 64  # 18.5ms/clip launch overhead x1540=28s collapses to ~0.3s total
        done = 0

        def flush(buf, items):
            nonlocal done
            for (spk, fn, _), feat in zip(items, extractor.extract_batch(buf)):
                _dump(out_root, spk, fn, feat); done += 1; pbar.update(1)

        with ThreadPoolExecutor(max_workers=4) as pool, \
                tqdm(total=len(pending), desc=f"mel@{fc.hash()} (batch)", leave=False) as pbar:
            buf, items = [], []  # clips load on CPU threads (soundfile releases the GIL)
            for item, wav in zip(pending, pool.map(lambda i: _load_wav(i, self.audio_dir, sr), pending)):
                items.append(item); buf.append(wav)
                if len(buf) >= BATCH: flush(buf, items); buf, items = [], []
            if buf: flush(buf, items)
        return done

    def _f0_batch(self, fc, extractor, pending, out_root, sr):
        from ..features.batching import bucket_batch

        # bucket near-equal lengths together; trailing pad stays under the <1Hz-error budget
        batches = bucket_batch([p[2] for p in pending], batch_size=64, max_pad_ratio=0.10)
        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool, \
                tqdm(total=len(pending), desc=f"f0@{fc.hash()} (batch)", leave=False) as pbar:
            wavs = list(pool.map(lambda i: _load_wav(i, self.audio_dir, sr), pending))  # prefetch all
            for bi in batches:
                feats = extractor.extract_batch([wavs[i].to(extractor.device) for i in bi])
                for idx, feat in zip(bi, feats):
                    _dump(out_root, pending[idx][0], pending[idx][1], feat)
                    done += 1; pbar.update(1)
        return done

    def _pipelined(self, fc, extractor, pending, out_root, sr):
        # GPU pipeline: loads run on CPU *processes* — librosa/numba hold the GIL so threads give
        # zero parallelism (12x slower measured); GPU forward (~8ms/clip) becomes the ceiling.
        import queue
        import threading

        import usvc.data.engine as _eng

        q: queue.Queue = queue.Queue(maxsize=4)  # 4 clips buffered ahead of the GPU

        def produce(q):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futs = {pool.submit(_eng._proc_load, item, fc.kind, str(self.audio_dir), sr): item
                        for item in pending}
                for fut in futs: q.put((futs[fut], fut.result()))
            q.put(None)  # terminal sentinel

        producer = threading.Thread(target=produce, args=(q,))
        producer.start()
        done = 0
        try:
            with tqdm(total=len(pending), desc=f"{fc.kind}@{fc.hash()} (pipeline)", leave=False) as pbar:
                while True:
                    blob = q.get()
                    if blob is None: break
                    (spk, fn, _), wav = blob
                    _dump(out_root, spk, fn, extractor.extract(wav)); done += 1; pbar.update(1)
        finally:
            producer.join(timeout=1)  # never hang on error paths; producer self-terminates when drained
        return done

    def _build_extractor(self, fc, sr, hop, *, device, dtype="fp32"):
        if fc.kind in ("mel", "aug_mel"): return MelExtractor(fc, sr, hop).to(device)
        if fc.kind == "f0": return F0Extractor(fc, sr, hop, device=device, dtype=dtype)
        if fc.kind == "rms": return RmsExtractor(fc, sr, hop)  # CPU vectorized unfold
        if fc.kind == "units": return UnitsExtractor(fc, sr, hop, device=device, dtype=dtype)
        if fc.kind == "spec":
            from ..features.spec import SpecExtractor
            return SpecExtractor(fc, sr, hop).to(device)
        if fc.kind == "aug_shift": return None  # pure scalar generation, no model
        raise ValueError(f"unknown feature kind: {fc.kind}")


def _proc_load(item, kind, audio_dir, sr):
    """ProcessPool worker: read pre-staged files (no resample). units prefer audio_16k/
    (soxr_hq at stage time), legacy fallback = runtime resample; others read audio/ at dataset sr."""
    import soundfile as sf

    spk, fn, _ = item
    if kind == "units":
        p16 = Path(audio_dir).parent / "audio_16k" / spk / fn
        if p16.is_file():
            return torch.from_numpy(sf.read(str(p16), dtype="float32")[0]).float()
        wav, file_sr = sf.read(str(Path(audio_dir) / spk / fn))
        if file_sr != 16000: wav = _resample_librosa(wav, file_sr, 16000, "soxr_hq")
        return torch.from_numpy(wav).float()
    return torch.from_numpy(_read_at_sr(Path(audio_dir) / spk / fn, sr)).float()


def _read_at_sr(path, sr):
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32")
    return wav if file_sr == sr else _resample_librosa(wav, file_sr, sr, "soxr_lq")


def _resample_librosa(wav, orig_sr, target_sr, res_type):
    import librosa
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)
