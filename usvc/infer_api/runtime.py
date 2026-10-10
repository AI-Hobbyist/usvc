"""Serial service-owned execution, local-only resources, CPU results, GPU teardown."""
import copy
import gc
import io
import logging
from pathlib import Path
import tempfile
import threading
import time
import subprocess
import sys
import json

import numpy as np
import soundfile as sf
import torch

from ..hub import local_pretrained_only, pretrained_dir
from ..infer.engine import InferenceEngine
from ..retrieval.faiss_index import NativeIndexCache
from .errors import APIError

LOG = logging.getLogger(__name__)


def memory(device):
    if not device.startswith("cuda"):
        return dict(allocated=0, reserved=0, peak=0)
    return dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                peak=torch.cuda.max_memory_allocated(device))


def caches():
    from ..features.units import _HUBERT_CACHE
    from ..features.f0 import _FCPE_CACHE, _RMVPE_CACHE
    from ..nn.cl_helper import WN_CACHE
    return [_HUBERT_CACHE, _FCPE_CACHE, _RMVPE_CACHE, WN_CACHE]


def release_owned(snapshots, device, compiled=False):
    for cache, previous in snapshots:
        for key in list(cache):
            if key not in previous:
                del cache[key]
    if compiled:
        torch._dynamo.reset()
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        # PyTorch's cuBLAS handle pool retains a 32 MiB workspace on this GPU
        # independently of model references. Release it after work has stopped.
        clear_workspace = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_workspace is not None:
            clear_workspace()
        # FFT plans retain idle allocator segments even after every tensor is gone.
        # This runtime is the dedicated service process's serial GPU owner.
        torch.backends.cuda.cufft_plan_cache[torch.device(device).index or 0].clear()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        # The first allocator pass retires segments used by released FFT plans;
        # drain those deferred frees before the final cache release.
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def encode(wav, sr):
    wav = np.asarray(wav, dtype=np.float32)
    if not wav.size or not np.isfinite(wav).all():
        raise ValueError("Invalid output audio")
    output = io.BytesIO()
    sf.write(output, wav, sr, format="WAV", subtype="PCM_16")
    return output.getvalue()


def sine_audio(model, path, values, device):
    from ..data.feature_config import FeatureConfig
    from ..features.f0 import F0Extractor
    from ..io.audio import resample_np
    wav, source_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    opts = model.cfg.backend_opts
    sr = int(opts.get("sampling_rate", opts.get("sr", 44100)))
    wav = resample_np(wav, source_sr, sr)
    count = len(wav)
    if not np.any(wav):
        return np.zeros(count, dtype=np.float32), sr
    method = values.get("f0_method") or opts.get("f0_extractor", "fcpe")
    ckpt = opts.get("rmvpe_ckpt", "pretrained/rmvpe/model.pt") if method == "rmvpe" else opts.get("fcpe_ckpt", "pretrained/fcpe/fcpe.pt")
    hop = int(opts.get("hop_length", opts.get("block_size", 512)))
    extractor = F0Extractor(FeatureConfig("f0", dict(extractor=method, model_ckpt=ckpt, uv_policy="raw")), sr, hop, device)
    f0 = extractor.extract(wav).numpy()
    sample_frame = np.arange(count) / hop
    pitch = np.interp(sample_frame, np.arange(len(f0)), f0) * 2. ** (values["trans"] / 12.)
    # Nearest frame UV mask prevents interpolation from voicing unvoiced frames.
    voiced = f0[np.minimum(np.rint(sample_frame).astype(int), len(f0) - 1)] > 0
    rms = np.sqrt(np.array([np.mean(wav[i:i + hop] ** 2) for i in range(0, count, hop)], dtype=np.float64))
    envelope = np.interp(sample_frame, np.arange(len(rms)), rms) * np.sqrt(2.)
    amplitude = np.minimum(envelope, .95) * voiced
    # Fade every voiced region, not only the beginning/end of the whole file.
    edges = np.diff(np.r_[False, voiced, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        fade = min(round(sr * .005), (end - start) // 2)
        if fade:
            amplitude[start:start + fade] *= np.linspace(0., 1., fade)
            amplitude[end - fade:end] *= np.linspace(1., 0., fade)
    phase = np.cumsum(pitch / sr, dtype=np.float64)
    return (np.sin(2. * np.pi * phase) * amplitude).astype(np.float32), sr


class Runtime:
    def __init__(self, device, probe_timeout=90.):
        self.device, self.probe_timeout = device, probe_timeout
        self.lock = threading.RLock()
        self.index_cache = NativeIndexCache()
        self.history = []
        self.loads = 0

    def execute(self, model, path, values, network_check=False):
        # Final cleanup runs after the complete inference frame has unwound, so
        # engine construction locals and native-operation temporaries are gone.
        with self.lock:
            snapshots = [(cache, set(cache)) for cache in caches()]
            try:
                return self._execute(model, path, values, network_check)
            finally:
                release_owned(snapshots, self.device, values.get("compile_graph", False))
                if self.history:
                    self.history[-1]["after"] = memory(self.device)
                    LOG.info("USVC resource lifecycle %s", self.history[-1])

    def _execute(self, model, path, values, network_check=False):
        with self.lock, local_pretrained_only():
            before = memory(self.device)
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            snapshots = [(cache, set(cache)) for cache in caches()]
            engine = None
            failure = None
            start = time.monotonic()
            try:
                if network_check:
                    from .fixture_engine import validate_network
                    validate_network(model, self.device)
                if model.fixture:
                    wav, sr = sine_audio(model, path, values, self.device)
                else:
                    kwargs = {key: values[key] for key in ("half", "compile_graph", "dec_cl", "vocoder_cl") if key in values}
                    for key, prefix in (("diffusion_model", "diffusion"), ("reflow_model", "reflow")):
                        if values.get(key):
                            enhancer = model.enhancers[values[key]]
                            kwargs[prefix + "_ckpt"], kwargs[prefix + "_config"] = enhancer.checkpoint, enhancer.config
                    vocoder_dir = model.metadata.get("vocoder_dir")
                    if vocoder_dir:
                        vocoder_dir = (model.directory / vocoder_dir).resolve()
                        if not vocoder_dir.is_relative_to(model.directory.parent.parent) and not vocoder_dir.is_relative_to(pretrained_dir()):
                            raise ValueError("Vocoder path escapes allowed roots")
                    else:
                        vocoder_dir = pretrained_dir() / "nsf_hifigan"
                    # Retain a reference even if __init__ fails part-way through CUDA loading.
                    engine = InferenceEngine.__new__(InferenceEngine)
                    cfg = copy.deepcopy(model.cfg)
                    if values.get("f0_method"):
                        cfg.backend_opts["f0_extractor"] = values["f0_method"]
                    InferenceEngine.__init__(engine, model.config, model.checkpoint, device=self.device,
                                             impl="torch", vocoder_dir=vocoder_dir, config_override=cfg, strict_shapes=True, **kwargs)
                    if values.get("compile_graph"):
                        compiled = hasattr(engine.module.sample, "_torchdynamo_orig_callable") or any(hasattr(child, "_orig_mod") for child in engine.module.modules())
                        if not compiled:
                            raise ValueError("Graph compilation was requested but no graph was created")
                    engine._native_index_cache = self.index_cache
                    self.loads += 1
                    params = {key: value for key, value in values.items() if key not in kwargs and key not in ("speaker", "index", "cluster", "resample_sr", "diffusion_model", "reflow_model", "vocoder")}
                    params["spk"] = values["speaker"]
                    params["index_path"] = model.indexes.get(values.get("index")) if values.get("index") is not False else None
                    params["cluster_path"] = model.clusters.get(values.get("cluster"))
                    wav, sr = engine.convert(path, **params), engine.sr
                if values.get("resample_sr") and values["resample_sr"] != sr:
                    from ..io.audio import resample_np
                    wav = resample_np(wav, sr, values["resample_sr"])
                    sr = values["resample_sr"]
                raw = encode(wav, sr)
                del wav
            except Exception as exc:
                # Copy error information; retaining a traceback would retain CUDA tensors.
                failure = (type(exc).__name__, str(exc))
            finally:
                engine = None
                release_owned(snapshots, self.device, values.get("compile_graph", False))
                record = dict(model=model.id, before=before, after=memory(self.device), elapsed=time.monotonic() - start,
                              success=failure is None, device=self.device)
                self.history.append(record)
                self.history = self.history[-100:]
            if failure:
                kind, message = failure
                status = 503 if kind in ("FileNotFoundError", "ImportError", "OutOfMemoryError") else 422 if kind in ("ValueError", "LibsndfileError") else 500
                raise APIError(status, "inference_failed", message)
            return raw, sr

    def probe(self, model, enhancer=False):
        # A thread cannot safely interrupt a stuck CUDA operation. Isolate probes
        # in a bounded child process while retaining the common GPU scheduling lock.
        with self.lock, tempfile.TemporaryDirectory(prefix="usvc-probe-run-") as temporary:
            folder = Path(temporary)
            report = folder / "result.json"
            log = folder / "build.log"
            def quote(value):
                return "'" + str(value).replace("'", "''") + "'"
            argv = [sys.executable, "-u", "-m", "usvc.infer_api.probe", "--model-root", model.directory.parent.parent,
                    "--model", model.id, "--device", self.device, "--report", report]
            if enhancer:
                argv.append("--enhancer")
            command = "& " + " ".join(map(quote, argv)) + " 2>&1 | Tee-Object -FilePath " + quote(log)
            command += "\n$buildExitCode = $LASTEXITCODE\nif ($buildExitCode -ne 0) { Get-Content " + quote(log) + "; exit $buildExitCode }"
            process = subprocess.Popen(["powershell", "-NoProfile", "-Command", command])
            try:
                code = process.wait(timeout=self.probe_timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
                process.wait(timeout=10)
                raise ValueError("Availability probe exceeded timeout") from None
            if report.exists():
                details = json.loads(report.read_text(encoding="utf-8"))
                self.history.extend(details.get("history", []))
                self.history = self.history[-100:]
            else:
                details = {}
            if code:
                raise ValueError(details.get("error", "Availability probe process failed"))

    def probe_local(self, model):
        from .settings import validate
        values = validate(model, {"speaker": model.default_speaker or 0, **({"steps": 2} if not model.fixture and model.backend in ("ddsp6", "diffusion", "reflow_vae", "rift_dit") else {})}, self.device)
        descriptor, filename = tempfile.mkstemp(suffix=".wav", prefix="usvc-probe-")
        import os
        os.close(descriptor)
        try:
            sr = 44100
            t = np.arange(round(sr * .3)) / sr
            sf.write(filename, .1 * np.sin(2. * np.pi * 180. * t), sr)
            raw, _ = self.execute(model, filename, values, network_check=model.fixture)
            if self.history[-1]["elapsed"] > self.probe_timeout:
                raise ValueError("Availability probe exceeded timeout")
            if not raw:
                raise ValueError("Empty availability probe")
        finally:
            Path(filename).unlink(missing_ok=True)

    def close(self):
        with self.lock:
            self.index_cache.clear()

    def probe_enhancer(self, model):
        return self.probe(model, enhancer=True)

    def probe_enhancer_local(self, model):
        with self.lock, local_pretrained_only():
            snapshots = [(cache, set(cache)) for cache in caches()]
            enhancer = None
            error = None
            try:
                from ..infer.reflow_enhancer import ReflowEnhancer
                enhancer = ReflowEnhancer(model.checkpoint, model.config, pretrained_dir() / "nsf_hifigan", device=self.device)
                units = torch.zeros(16, model.cfg.backend_opts.get("n_units", 768), device=self.device)
                f0 = torch.full((16,), 180., device=self.device)
                rms = torch.full((16,), .1, device=self.device)
                wav = torch.sin(torch.arange(16 * 512, device=self.device) * (2 * np.pi * 180 / 44100)) * .1
                output = enhancer.enhance(units, f0, rms, 0, wav, infer_step=2)
                if not np.isfinite(output).all():
                    raise ValueError("Nonfinite enhancer output")
                del units, f0, rms, wav, output
            except Exception as exc:
                error = str(exc)
            finally:
                enhancer = None
                release_owned(snapshots, self.device)
            if error:
                raise ValueError(error)
