"""M2 binary API, real F0 and GPU teardown; no model teardown deletes fixtures."""
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from usvc.infer_api.app import create_app
from usvc.infer_api.catalog import Catalog
from usvc.infer_api.runtime import Runtime, encode
from usvc.infer_api.settings import validate

ROOT = Path(__file__).resolve().parents[1]
MODEL = "ddsp6/__test_ddsp65_multi"


def archive(raw, details):
    digest = hashlib.sha256(raw).hexdigest()
    path = ROOT / "tests/output" / f"{digest}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.stem
    details.update(output_sha256=digest, path=str(path))
    path.with_suffix(".json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AUDIO: {path}", flush=True)
    return path


def test_real_f0_source_transpose_and_gpu_release():
    runtime = Runtime("cuda")
    app = create_app(ROOT / "models", device="cuda", runtime=runtime, validator=lambda model: None)
    source = ROOT / "tests/input/1.wav"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    results = []
    with TestClient(app) as client:
        for trans in (0, 12):
            params = {"speaker": "测试乙", "trans": trans, "f0_method": "fcpe"}
            response = client.post("/infer", params={"model": MODEL, "params": json.dumps(params)}, content=source.read_bytes(), headers={"Content-Type": "audio/wav"})
            assert response.status_code == 200, response.text
            assert response.headers["X-USVC-Test-Fixture"] == "true"
            assert response.headers["X-USVC-Sample-Rate"] == "44100"
            info = sf.info(io.BytesIO(response.content))
            assert abs(info.duration - sf.info(source).duration) < 1 / 44100
            path = archive(response.content, dict(model=MODEL, params=params, device="cuda", source_sha256=source_hash, test_fixture=True))
            results.append(sf.read(path)[0])
            assert not app.state.uploads
            record = runtime.history[-1]
            assert record["after"]["allocated"] <= record["before"]["allocated"] + 1024 ** 2
            assert record["after"]["reserved"] <= record["before"]["reserved"] + 1024 ** 2
        # Frequency from zero crossings at a stable voiced region doubles.
        x, y = results
        source_voiced = np.flatnonzero(np.abs(x) > .03)
        assert len(source_voiced) > 1000
        starts = source_voiced[::max(1, len(source_voiced) // 30)]
        ratios = []
        for start in starts:
            a, b = x[start:start + 2205], y[start:start + 2205]
            ca, cb = np.count_nonzero(np.diff(np.signbit(a))), np.count_nonzero(np.diff(np.signbit(b)))
            if ca > 8:
                ratios.append(cb / ca)
        assert np.median(ratios) == pytest.approx(2., abs=.15)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_silence_resampling_and_bad_audio():
    app = create_app(ROOT / "models", device="cuda", validator=lambda model: None)
    with TestClient(app) as client:
        source = encode(np.zeros(4410, dtype=np.float32), 44100)
        response = client.post("/infer", params={"model": MODEL, "params": json.dumps({"speaker": 0, "resample_sr": 24000})}, content=source, headers={"Content-Type": "audio/wav"})
        assert response.status_code == 200
        wav, sr = sf.read(io.BytesIO(response.content))
        assert sr == 24000 and len(wav) == 2400 and not wav.any()
        archive(response.content, dict(model=MODEL, params={"speaker": 0, "resample_sr": 24000}, device="cuda", input="deterministic silence"))
        for content in (b"", b"not an audio file"):
            assert client.post("/infer", params={"model": MODEL, "params": '{"speaker":0}'}, content=content).status_code == 422
        assert not app.state.uploads


@pytest.mark.parametrize("params", ["[]", "{", '{"speaker":0,"unknown":1}', '{"speaker":true}', '{"speaker":2}', '{"trans":NaN}', '{"speaker":0,"index_ratio":0.5}'])
def test_invalid_parameters_before_execution(params):
    runtime = Runtime("cuda")
    app = create_app(ROOT / "models", runtime=runtime, device="cuda", validator=lambda model: None)
    with TestClient(app) as client:
        response = client.post("/infer", params={"model": MODEL, "params": params}, content=b"bad")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_parameter"
        assert "\n  \"error\"" in response.text
        assert not runtime.history and not app.state.uploads


def test_authentication_does_not_consume_body_or_load():
    runtime = Runtime("cuda")
    app = create_app(ROOT / "models", runtime=runtime, token="secret", device="cuda")
    with TestClient(app) as client:
        assert client.post("/infer", params={"model": MODEL}, content=b"bad").status_code == 401
        assert not runtime.history and not app.state.uploads


def test_upload_and_media_limits():
    runtime = Runtime("cuda")
    app = create_app(ROOT / "models", runtime=runtime, device="cuda", max_upload_bytes=30, validator=lambda model: None)
    with TestClient(app) as client:
        params = {"model": MODEL, "params": '{"speaker":0}'}
        assert client.post("/infer", params=params, content=b"a" * 31).status_code == 413
        assert client.post("/infer", params=params, content=b"a", headers={"Content-Type": "application/json"}).status_code == 415
        assert client.post("/infer", params=params, content=iter([b"a" * 20, b"b" * 20])).status_code == 413
        assert not runtime.history and not app.state.uploads


def test_real_engine_rvc_forward_and_unload(monkeypatch):
    import weakref
    from usvc.infer.engine import InferenceEngine
    original = InferenceEngine.__init__
    references, devices, selected = [], [], []
    def observed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        references.extend([weakref.ref(self.module), weakref.ref(self.features)])
        devices.append(str(next(self.module.parameters()).device))
        self.module.generator.emb_g.register_forward_pre_hook(lambda module, args: selected.extend(args[0].detach().cpu().flatten().tolist()))
    monkeypatch.setattr(InferenceEngine, "__init__", observed)
    runtime = Runtime("cuda")
    catalog = Catalog(ROOT / "models", lambda model: None)
    model = catalog.candidate("rvc/guanguanV1")
    values = validate(model, {"rms_mix_rate": 1.}, "cuda")
    raw, sr = runtime.execute(model, ROOT / "tests/input/1.wav", values)
    archive(raw, dict(model=model.id, params=values, device="cuda", source_sha256=hashlib.sha256((ROOT / "tests/input/1.wav").read_bytes()).hexdigest(), test_fixture=False))
    assert sr == 40000 and runtime.loads == 1
    record = runtime.history[-1]
    assert record["after"]["allocated"] <= record["before"]["allocated"], record
    assert record["after"]["reserved"] <= record["before"]["reserved"], record
    from usvc.infer_api.runtime import caches
    assert not any(caches())
    raw, _ = runtime.execute(model, ROOT / "tests/input/1.wav", validate(model, {"trans": 12}, "cuda"))
    archive(raw, dict(model=model.id, params={"trans": 12}, device="cuda", source_sha256=hashlib.sha256((ROOT / "tests/input/1.wav").read_bytes()).hexdigest(), test_fixture=False))
    assert runtime.loads == 2 and all(ref() is None for ref in references)
    assert devices == ["cuda:0", "cuda:0"] and selected and set(selected) == {0}
    assert runtime.history[-1]["after"]["allocated"] == 0
    assert runtime.history[-1]["after"]["reserved"] == 0


def test_serial_admission_and_distinct_speakers():
    class SlowRuntime:
        def __init__(self):
            self.entered = threading.Event()
            self.resume = threading.Event()
            self.speakers = []
        def execute(self, model, path, values):
            self.speakers.append(values["speaker"])
            self.entered.set()
            assert self.resume.wait(10)
            return encode(np.zeros(100), 44100), 44100
        def close(self):
            pass
    runtime = SlowRuntime()
    app = create_app(ROOT / "models", runtime=runtime, validator=lambda model: None, max_pending=1)
    with TestClient(app) as client:
        result = []
        def first():
            result.append(client.post("/infer", params={"model": MODEL, "params": '{"speaker":1}'}, content=encode(np.zeros(100), 44100)))
        thread = threading.Thread(target=first)
        thread.start()
        assert runtime.entered.wait(10)
        try:
            second = client.post("/infer", params={"model": MODEL, "params": '{"speaker":0}'}, content=encode(np.zeros(100), 44100))
            assert second.status_code == 503
        finally:
            runtime.resume.set()
            thread.join(10)
        assert result[0].status_code == 200 and runtime.speakers == [1]
        assert not app.state.uploads


def test_probe_local_only_and_cache_invalidation(monkeypatch):
    from usvc import hub
    from usvc.hub import local_pretrained_only
    def forbidden(*args, **kwargs):
        pytest.fail("availability triggered a download")
    monkeypatch.setattr(hub, "_download_hf", forbidden)
    monkeypatch.setattr(hub, "_download_ms", forbidden)
    with local_pretrained_only(), pytest.raises(FileNotFoundError):
        hub.ensure_pretrained("missing-infer-api-fixture.pt")
    runtime = Runtime("cuda")
    catalog = Catalog(ROOT / "models", runtime.probe)
    model = catalog.candidate(MODEL)
    runtime.probe(model)
    assert runtime.history[-1]["success"]
    assert runtime.history[-1]["after"]["allocated"] <= 1024 ** 2


@pytest.mark.parametrize("where", ["adapter", "constructor", "convert"])
def test_failure_releases_partial_cuda_objects(monkeypatch, where):
    from usvc.infer_api import runtime as runtime_module
    runtime = Runtime("cuda")
    catalog = Catalog(ROOT / "models", lambda model: None)
    model = catalog.candidate(MODEL if where == "adapter" else "rvc/guanguanV1")
    before = torch.cuda.memory_allocated()
    def failed_adapter(*args):
        temporary = torch.ones(1024 * 1024, device="cuda")
        raise RuntimeError("injected adapter failure")
    def failed_constructor(self, *args, **kwargs):
        self.module = torch.ones(1024 * 1024, device="cuda")
        raise RuntimeError("injected load failure")
    def constructor(self, *args, **kwargs):
        self.module = torch.ones(1024 * 1024, device="cuda")
        self.sr = 40000
    def failed_convert(self, *args, **kwargs):
        temporary = torch.ones(1024 * 1024, device="cuda")
        raise RuntimeError("injected conversion failure")
    if where == "adapter":
        monkeypatch.setattr(runtime_module, "sine_audio", failed_adapter)
    elif where == "constructor":
        monkeypatch.setattr(runtime_module.InferenceEngine, "__init__", failed_constructor)
    else:
        monkeypatch.setattr(runtime_module.InferenceEngine, "__init__", constructor)
        monkeypatch.setattr(runtime_module.InferenceEngine, "convert", failed_convert)
    from usvc.infer_api.errors import APIError
    with pytest.raises(APIError):
        runtime.execute(model, ROOT / "tests/input/1.wav", validate(model, {"speaker": 0}, "cuda"))
    assert runtime.history[-1]["after"]["allocated"] == before
    assert not runtime.history[-1]["success"]


def test_cancel_waits_for_real_worker_and_cleans_upload():
    import anyio
    import httpx
    from usvc.infer_api.runtime import release_owned, caches
    class BlockingRuntime:
        def __init__(self):
            self.entered, self.finished = threading.Event(), threading.Event()
        def execute(self, model, path, values):
            snapshots = [(cache, set(cache)) for cache in caches()]
            tensor = torch.ones(1024 * 1024, device="cuda")
            self.entered.set()
            time.sleep(.3)
            assert path.exists(), "cancel deleted input while worker was still running"
            del tensor
            release_owned(snapshots, "cuda")
            self.finished.set()
            return encode(np.zeros(100), 44100), 44100
        def close(self):
            pass
    runtime = BlockingRuntime()
    app = create_app(ROOT / "models", runtime=runtime, validator=lambda model: None)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            async with anyio.create_task_group() as group:
                async def request():
                    await client.post("/infer", params={"model": MODEL, "params": '{"speaker":1}'}, content=encode(np.zeros(100), 44100))
                group.start_soon(request)
                while not runtime.entered.is_set():
                    await anyio.sleep(.01)
                group.cancel_scope.cancel()
    anyio.run(exercise)
    assert runtime.finished.is_set() and not app.state.uploads
    assert torch.cuda.memory_allocated() == 0


def test_real_index_selection_defaults_disable_and_missing():
    catalog = Catalog(ROOT / "models", lambda model: None)
    model = catalog.candidate("rvc/guanguanV1")
    default = validate(model, {"index_ratio": .5}, "cuda")
    assert default["index"] == "guanguanV1.index"
    assert validate(model, {"index_ratio": .5, "index": False}, "cuda")["index"] is False
    from usvc.infer_api.errors import APIError
    for values in ({"index": "C:/arbitrary.index"}, {"cluster_ratio": .5}, {"shallow_diffusion": True}):
        with pytest.raises(APIError):
            validate(model, values, "cuda")
    missing = catalog.candidate("rvc/youzhanv2-xi")
    with pytest.raises(APIError):
        validate(missing, {"index_ratio": .5}, "cuda")


def test_catalog_revokes_and_revalidates_without_mutating_files():
    calls = []
    failure = [False]
    def validator(model):
        calls.append(model.id)
        if failure[0] and model.id == MODEL:
            raise ValueError("controlled unavailable resource")
    catalog = Catalog(ROOT / "models", validator)
    assert MODEL in {m["id"] for m in catalog.refresh()}
    before = len(calls)
    catalog.refresh()
    assert len(calls) == before
    catalog.invalidate(MODEL)
    failure[0] = True
    assert MODEL not in {m["id"] for m in catalog.refresh()}
    failure[0] = False
    assert MODEL in {m["id"] for m in catalog.refresh()}


def test_source_pitch_tracks_two_tones_and_unvoiced_silence():
    sr = 44100
    t = np.arange(sr // 2) / sr
    source = np.r_[.15 * np.sin(2 * np.pi * 180 * t), np.zeros(sr // 2), .15 * np.sin(2 * np.pi * 300 * t)]
    app = create_app(ROOT / "models", device="cuda", validator=lambda model: None)
    with TestClient(app) as client:
        response = client.post("/infer", params={"model": MODEL, "params": '{"speaker":0}'}, content=encode(source, sr))
        assert response.status_code == 200
        archive(response.content, dict(model=MODEL, device="cuda", input="deterministic 180Hz/silence/300Hz", params={"speaker": 0}))
        output, _ = sf.read(io.BytesIO(response.content))
        for start, expected in ((.1, 180.), (1.1, 300.)):
            segment = output[round(start * sr):round((start + .2) * sr)]
            frequency = np.count_nonzero(np.diff(np.signbit(segment))) / (.4)
            assert frequency == pytest.approx(expected, abs=10)
        assert np.max(np.abs(output[round(.7 * sr):round(.9 * sr)])) == 0


def test_probe_timeout_kills_its_process_tree():
    runtime = Runtime("cuda", probe_timeout=.1)
    model = Catalog(ROOT / "models", lambda model: None).candidate(MODEL)
    started = time.monotonic()
    with pytest.raises(ValueError, match="timeout"):
        runtime.probe(model)
    assert time.monotonic() - started < 10
    assert torch.cuda.memory_allocated() == 0


def test_unavailable_fixture_rejected_on_first_infer():
    runtime = Runtime("cuda")
    def rejected(model):
        raise ValueError("controlled incompatible checkpoint")
    app = create_app(ROOT / "models", runtime=runtime, device="cuda", validator=rejected)
    with TestClient(app) as client:
        response = client.post("/infer", params={"model": MODEL, "params": '{"speaker":0}'}, content=encode(np.zeros(100), 44100))
        assert response.status_code == 404
        assert not runtime.history and not app.state.uploads


def test_service_strict_weight_shapes_preserve_legacy_default():
    import uuid
    from usvc.infer.engine import InferenceEngine
    directory = ROOT / "tests/output" / ("strict-shapes-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    path = directory / "model.pth"
    torch.save({"state_dict": {"generator.weight": torch.ones(2, 4), "generator.bias": torch.zeros(3)}}, path)
    module = torch.nn.Module()
    module.generator = torch.nn.Linear(4, 3)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.backend_name, engine.ckpt_path = "rvc", str(path)
    engine.strict_shapes = True
    with pytest.raises(RuntimeError, match="shape mismatch"):
        engine._load_weights(module)
    engine.strict_shapes = False
    engine._load_weights(module)
