"""Real Windows launcher / HTTP / CUDA acceptance; no replacement engines."""
import ast
from contextlib import contextmanager
import hashlib
import http.client
import io
import json
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(sys.executable).parent / "Scripts/usvc-infer-api.exe"
SOURCE = ROOT / "tests/input/1.wav"
SOURCE_HASH = "620d7fc3bdd866e4eaf4eba498b2a3b7ecfc924fe557d3b47d8bd7dc39081039"


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def pipeline(args, logfile):
    return "$PSDefaultParameterValues['Out-File:Width']=4096; & " + " ".join(map(quote, args)) + " 2>&1 | Tee-Object -FilePath " + quote(logfile) + "; $buildExitCode=$LASTEXITCODE; if ($buildExitCode -ne 0) { Get-Content " + quote(logfile) + "; exit $buildExitCode }"


def request(host, port, path, body=None, headers=None):
    connection = http.client.HTTPConnection(host, port, timeout=600)
    try:
        connection.request("POST" if body is not None else "GET", path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, {key.lower(): value for key, value in response.getheaders()}, response.read()
    finally:
        connection.close()


def driver(pids=()):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"], text=True, capture_output=True)
    report = dict(returncode=result.returncode, output=result.stdout, error=result.stderr)
    if pids:
        command = "$wanted=@(" + ",".join(map(str, pids)) + "); try { @((Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples | Where-Object { $wanted -contains [int]($_.InstanceName.Split('_')[1]) } | Select-Object InstanceName,CookedValue) | ConvertTo-Json -Compress } catch { Write-Error $_; exit 1 }"
        counters = subprocess.run(["powershell", "-NoProfile", "-Command", command], text=True, capture_output=True, timeout=15)
        report["windows_dedicated_bytes"] = dict(returncode=counters.returncode, samples=json.loads(counters.stdout) if counters.stdout.strip() else [], error=counters.stderr)
    return report


def read_log(log):
    raw = log.read_bytes() if log.exists() else b""
    return raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8-sig", errors="replace")


def records(log):
    result = []
    pending = ""
    for line in read_log(log).splitlines():
        if "USVC resource lifecycle " in line:
            pending = line.split("USVC resource lifecycle ", 1)[1]
        elif pending:
            pending += line.strip()
        if pending:
            try:
                result.append(ast.literal_eval(pending))
                pending = ""
            except (ValueError, SyntaxError):
                pass
    return result


def children(pid):
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress"],
                            text=True, capture_output=True, check=True)
    all_processes = json.loads(result.stdout)
    selected = {pid}
    while True:
        found = {item["ProcessId"] for item in all_processes if item["ParentProcessId"] in selected}
        if found <= selected:
            return selected - {pid}
        selected |= found


def alive(pid):
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def server(tmp_path, host="127.0.0.1", port=8000, token=None):
    log = tmp_path / f"server-{port}.log"
    args = [LAUNCHER, "--device", "cuda", "--model-root", ROOT / "models"]
    if host != "127.0.0.1" or port != 8000:
        args += ["--host", host, "--port", port]
    if token is not None:
        args += ["--token", token]
    process = subprocess.Popen(["powershell", "-NoProfile", "-Command", pipeline(args, log)], cwd=ROOT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    descendants = []
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            assert process.poll() is None, "launcher exited before listening"
            try:
                status, _, _ = request(host, port, "/settings")
                if status in (401, 422):
                    break
            except OSError:
                time.sleep(.2)
        else:
            pytest.fail("launcher did not listen within 45 seconds")
        yield host, port, log
    finally:
        if process.poll() is None:
            descendants = children(process.pid)
            process.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
                process.wait(timeout=10)
        deadline = time.monotonic() + 20
        while any(alive(pid) for pid in descendants) and time.monotonic() < deadline:
            time.sleep(.1)
        assert not any(alive(pid) for pid in descendants), "server child process survived shutdown"
        with socket.socket() as connection:
            assert connection.connect_ex((host, port)) != 0, "server port survived shutdown"


def archive(raw, details):
    digest = hashlib.sha256(raw).hexdigest()
    path = ROOT / "tests/output" / f"{digest}.wav"
    if not path.exists():
        path.write_bytes(raw)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    details.update(output_sha256=digest, path=str(path), source_sha256=SOURCE_HASH)
    path.with_suffix(".json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HTTP AUDIO: {path}", flush=True)
    return details


def test_real_launcher_full_matrix_and_python_example(tmp_path):
    assert torch.cuda.is_available() and LAUNCHER.exists()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    evidence = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                    input=dict(sha256=SOURCE_HASH, sample_rate=sf.info(SOURCE).samplerate, duration=sf.info(SOURCE).duration),
                    driver_before=driver(), outputs=[])
    with server(tmp_path) as (host, port, log):
        status, _, raw = request(host, port, "/model")
        assert status == 200, raw
        assert b"\n  {" in raw
        models = json.loads(raw)
        assert len(models) == 12 and sum(item["test_fixture"] for item in models) == 7
        assert models == sorted(models, key=lambda item: (item["backend"], item["id"]))
        evidence["models"] = models
        server_pid = int(re.search(r"Started server process \[(\d+)\]", read_log(log)).group(1))
        evidence["server_pid"] = server_pid
        evidence["driver_after_probes"] = driver([server_pid])
        settings_examples = {}
        for model in models:
            print(f"HTTP MATRIX: {model['id']}", flush=True)
            query = urlencode(dict(backend=model["backend"], model=model["id"]))
            status, _, raw = request(host, port, "/settings?" + query)
            assert status == 200 and b"\n  " in raw
            settings = json.loads(raw)
            assert settings["device"] == "cuda"
            settings_examples[model["id"]] = settings
            choices = [{"speaker": role[key]} for role in model["speakers"] for key in ("id", "name")] if model["multi_speaker"] else [{}]
            if model["backend"] == "rvc" and model["indexes"]:
                choices = [{"index_ratio": ratio} for ratio in (0., .5, 1.)]
            for params in choices:
                status, headers, wav = request(host, port, "/infer?" + urlencode(dict(model=model["id"], params=json.dumps(params, ensure_ascii=False))), SOURCE.read_bytes(), {"Content-Type": "audio/wav"})
                assert status == 200, wav
                assert headers["content-type"] == "audio/wav"
                info = sf.info(io.BytesIO(wav))
                assert info.samplerate == int(headers["x-usvc-sample-rate"])
                assert abs(info.duration - sf.info(SOURCE).duration) < .1
                audio, _ = sf.read(io.BytesIO(wav))
                assert audio.size and np.isfinite(audio).all() and np.max(np.abs(audio)) > 0
                assert (headers.get("x-usvc-test-fixture") == "true") == model["test_fixture"]
                evidence["outputs"].append(archive(wav, dict(model=model["id"], params=params, device="cuda", test_fixture=model["test_fixture"], headers=headers)))
                evidence.setdefault("driver_after_requests", []).append(driver([server_pid]))
        evidence["settings"] = settings_examples
        # Documentation examples must match the actual response, not a planned schema.
        samples = [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", (ROOT / "对接指南.md").read_text(encoding="utf-8"), flags=re.S)]
        for item in samples[0]:
            assert item == next(model for model in models if model["id"] == item["id"])
        assert samples[1] == settings_examples[samples[1]["model"]]
        # Real socket disconnect after a complete upload: GPU teardown must finish.
        previous = len(records(log))
        query = urlencode(dict(model="ddsp6/__test_ddsp65_multi", params='{"speaker":0}'))
        with socket.create_connection((host, port)) as connection:
            body = SOURCE.read_bytes()
            connection.sendall((f"POST /infer?{query} HTTP/1.1\r\nHost: {host}\r\nContent-Type: audio/wav\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
            time.sleep(.1)
        deadline = time.monotonic() + 30
        while len(records(log)) <= previous and time.monotonic() < deadline:
            time.sleep(.2)
        assert len(records(log)) > previous, "disconnected request did not finish resource teardown"
        evidence["driver_after_disconnect"] = driver([server_pid])
        errors = []
        for path, body, expected in [("/settings", None, 422), ("/infer?model=unknown/name", b"bad", 404),
                                      ("/missing-route", None, 404), ("/infer", None, 405),
                                      ("/infer?" + urlencode(dict(model="ddsp6/__test_ddsp65_multi", params='{"speaker":2}')), b"bad", 422),
                                      ("/infer?model=rvc/youzhanv2-xi&params=%7B%22index_ratio%22%3A0.5%7D", SOURCE.read_bytes(), 422),
                                      ("/infer?model=rvc/guanguanV1", b"bad", 422)]:
            status, _, raw = request(host, port, path, body)
            assert status == expected and b"\n  \"error\"" in raw
            errors.append(dict(status=status, body=json.loads(raw)))
        evidence["errors"] = errors
        params_file = tmp_path / "client-params.json"
        params_file.write_text('{"speaker":"测试乙","trans":12}', encoding="utf-8")
        command = [sys.executable, ROOT / "examples/infer_api_client.py", "--model", "sovits/__test_sovits_multi", "--params-file", params_file]
        assert subprocess.run(["powershell", "-NoProfile", "-Command", pipeline(command, tmp_path / "client.log")], cwd=ROOT, timeout=60).returncode == 0
        history = records(log)
        assert len(history) >= len(evidence["outputs"]) + 2
        for record in history:
            assert record["after"]["allocated"] <= record["before"]["allocated"], record
            assert record["after"]["reserved"] <= record["before"]["reserved"], record
        evidence["resource_history"] = history
    evidence["driver_after_shutdown"] = driver([server_pid])
    evidence["source_unchanged"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    assert evidence["source_unchanged"]
    (ROOT / "tests/output/infer-api-m3-http.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


def test_real_launcher_auth_and_errors(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.2", 0))
        port = reservation.getsockname()[1]
    with server(tmp_path, "127.0.0.2", port, "acceptance-token") as (host, port, log):
        for auth in (None, "Basic acceptance-token", "Bearer wrong"):
            status, headers, raw = request(host, port, "/infer?model=rvc/guanguanV1", b"bad", {"Authorization": auth} if auth else {})
            assert status == 401 and headers["www-authenticate"] == "Bearer"
            assert json.loads(raw)["error"]["code"] == "unauthorized"
        assert not records(log)
        status, _, raw = request(host, port, "/settings", headers={"Authorization": "Bearer acceptance-token"})
        assert status == 422 and json.loads(raw)["error"]["field"] == "backend"
        # Correct auth actually reaches conversion, with a real first-use probe.
        query = urlencode(dict(model="diffusion/__test_diffusion_single", params="{}"))
        status, headers, raw = request(host, port, "/infer?" + query, SOURCE.read_bytes(), {"Authorization": "Bearer acceptance-token", "Content-Type": "audio/wav"})
        assert status == 200 and headers["x-usvc-test-fixture"] == "true"
        archive(raw, dict(model="diffusion/__test_diffusion_single", params={}, device="cuda", token_mode=True))
        occupied = subprocess.run(["powershell", "-NoProfile", "-Command", pipeline([LAUNCHER, "--host", host, "--port", port], tmp_path / "occupied.log")], cwd=ROOT, timeout=30)
        assert occupied.returncode != 0
    for name, arguments in [("port", ["--port", "0"]), ("host", ["--host", "invalid-address"]), ("cuda", ["--device", "cuda:999"])]:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", pipeline([LAUNCHER, *arguments], tmp_path / f"invalid-{name}.log")], cwd=ROOT, timeout=30)
        assert result.returncode != 0


def test_documented_json_matches_recorded_http():
    evidence = json.loads((ROOT / "tests/output/infer-api-m3-http.json").read_text(encoding="utf-8"))
    samples = [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", (ROOT / "对接指南.md").read_text(encoding="utf-8"), flags=re.S)]
    for item in samples[0]:
        assert item == next(model for model in evidence["models"] if model["id"] == item["id"])
    assert samples[1] == evidence["settings"][samples[1]["model"]]
    assert samples[2] == evidence["errors"][0]["body"]


def test_all_http_json_errors_are_pretty():
    from fastapi.testclient import TestClient
    from usvc.infer_api.app import create_app
    with TestClient(create_app(ROOT / "models", device="cpu", validator=lambda model: None)) as client:
        for route, expected in (("/missing-route", 404), ("/infer", 405)):
            response = client.get(route)
            assert response.status_code == expected and '\n  "error"' in response.text


def test_local_pretrained_cuda_forwards():
    from usvc.data.feature_config import FeatureConfig
    from usvc.features.f0 import F0Extractor
    from usvc.features.units import UnitsExtractor
    from usvc.hub import local_pretrained_only
    from usvc.infer_api.runtime import caches, encode, release_owned
    from usvc.io.audio import resample_np
    from usvc.nn.vocoder import NSFHiFiGAN
    manifest = json.loads((ROOT / "pretrained/preparation-manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / "pretrained" / item["file"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    wav, sr = sf.read(SOURCE, dtype="float32", always_2d=True)
    wav = wav[:sr, :].mean(1)

    def forward(kind):
        if kind in ("fcpe", "rmvpe"):
            checkpoint = ROOT / "pretrained" / ("fcpe/fcpe.pt" if kind == "fcpe" else "rmvpe/model.pt")
            extractor = F0Extractor(FeatureConfig("f0", dict(extractor=kind, model_ckpt=str(checkpoint))), sr, 512, "cuda")
            output = extractor.extract(wav)
            device = str(extractor.fcpe.tensor_device_marker.device) if kind == "fcpe" else str(next(extractor.rmvpe.model.parameters()).device)
        elif kind in ("contentvec", "hubertsoft"):
            checkpoint = ROOT / "pretrained" / ("contentvec/pytorch_model.bin" if kind == "contentvec" else "hubert/hubert-soft-0d54a1f4.pt")
            extractor = UnitsExtractor(FeatureConfig("units", dict(kind=kind, model_ckpt=str(checkpoint))), sr, 512, "cuda")
            output = extractor.extract(torch.from_numpy(resample_np(wav, sr, 16000)))
            device = str(next(extractor.model.parameters()).device)
            assert output.shape[-1] == (768 if kind == "contentvec" else 256)
        else:
            directory = ROOT / "pretrained/nsf_hifigan"
            vocoder = NSFHiFiGAN(directory / "config.json", directory / "model.ckpt", device="cuda")
            with torch.no_grad():
                output = vocoder.infer(torch.zeros(1, 16, 128, device="cuda"), torch.full((1, 16), 180., device="cuda"))
            device = str(output.device)
            archive(encode(output.detach().cpu().numpy().flatten(), 44100), dict(component="real NSF-HiFiGAN with synthetic mel/F0", device=device))
        assert device == "cuda:0" and output.numel() and torch.isfinite(output).all()
        return dict(component=kind, tensor_device=device, output_shape=list(output.shape))

    evidence = []
    with local_pretrained_only():
        for kind in ("fcpe", "rmvpe", "contentvec", "hubertsoft", "nsf_hifigan"):
            snapshots = [(cache, set(cache)) for cache in caches()]
            before = torch.cuda.memory_allocated()
            try:
                evidence.append(forward(kind))
            finally:
                release_owned(snapshots, "cuda")
            assert torch.cuda.memory_allocated() == before
    (ROOT / "tests/output/infer-api-m3-pretrained.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
