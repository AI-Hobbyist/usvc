"""Speaker selection boundaries and real simulated-corpus CLI lifecycle."""
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from usvc.cli import infer
from usvc.infer.engine import InferenceEngine
from usvc.speakers import load_model_speakers, resolve_speaker
from usvc.train.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


def run_cli(module, args):
    # Every child executes in foreground through the repository PowerShell logging contract.
    code = "import torch,numpy,runpy; torch.manual_seed(123); numpy.random.seed(123); runpy.run_module(" + repr(module) + ",run_name='__main__')"
    argv = [sys.executable, "-u", "-c", code, *map(str, args)]
    quoted = ["'" + a.replace("'", "''") + "'" for a in argv]
    log_dir = ROOT / "tests/output/cli-logs" / uuid.uuid4().hex
    log_dir.mkdir(parents=True)
    log = "'" + str(log_dir / "build.log").replace("'", "''") + "'"
    command = "& " + " ".join(quoted) + " 2>&1 | Tee-Object -FilePath " + log + "\n$buildExitCode = $LASTEXITCODE\nif ($buildExitCode -ne 0) { Get-Content " + log + " -Tail 30; exit $buildExitCode }"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], cwd=ROOT)
    assert result.returncode == 0, f"{module} failed; see build.log"


def archive_audio(path, details):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    out = ROOT / "tests/output" / (digest + ".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_bytes(raw)
    wav, sr = sf.read(path)
    assert wav.size and np.isfinite(wav).all()
    details.update(output_sha256=digest, sample_rate=sr, duration=len(wav)/sr,
                   finite=True, source_sha256=hashlib.sha256((ROOT / "tests/input/1.wav").read_bytes()).hexdigest())
    out.with_suffix(".json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AUDIO: {out}", flush=True)


def test_listing_and_legacy_selection(tmp_path, capsys):
    path = tmp_path / "model.pth"
    config = tmp_path / "config.yaml"
    cfg = {"backend": "ddsp6", "spk": {"歌手 甲": 0, "乙": 1}, "backend_opts": {"num_speaker": 2}}
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    torch.save({"spk_embed.weight": torch.randn(2, 32)}, path)
    assert infer.main(["--config", str(config), "--ckpt", str(path), "--list-speakers"]) == 0
    assert "歌手 甲" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        infer.main(["--config", str(config), "--ckpt", str(path), "--speaker", "乙", "--spk", "0"])
    cfg.pop("spk"); config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    mapping, capacity = load_model_speakers(TrainConfig.from_yaml(config), path)
    assert mapping is None and resolve_speaker(0, mapping, capacity) == 0
    with pytest.raises(ValueError):
        resolve_speaker(None, mapping, capacity)
    cfg["backend_opts"]["num_speaker"] = 1
    torch.save({}, path); config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert resolve_speaker(None, *load_model_speakers(TrainConfig.from_yaml(config), path)) == 0


def test_target_resources_and_enhancer(tmp_path):
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.backend_name, engine.device = "sovits", "cpu"
    engine.speaker_mapping, engine.speaker_capacity = {"甲": 0, "乙": 1}, 2
    engine.cfg = TrainConfig(backend="sovits", backend_opts={"num_speaker": 2})
    units = torch.zeros(3, 4)
    cluster = tmp_path / "cluster.pt"
    torch.save({"kind": "units_cluster", "甲": {"cluster_centers_": torch.ones(1, 4)},
                "乙": {"cluster_centers_": torch.full((1, 4), 2.)}}, cluster)
    for sid in (0, 1):
        result = engine._enhance_units(units, cluster_path=cluster, cluster_ratio=1., spk=sid)
        torch.testing.assert_close(result, torch.full_like(units, sid + 1.))
    index = tmp_path / "index.pt"
    torch.save({"vectors": torch.ones(2, 4), "spk_id": "甲"}, index)
    with pytest.raises(ValueError, match="different speaker"):
        engine._enhance_units(units, index_path=index, index_ratio=1., spk=1)
    engine.reflow_enhancer = SimpleNamespace(speaker_capacity=2, speaker_mapping={"乙": 0, "甲": 1})
    assert engine._resolve_enhancer_speaker(0) == 1
    engine.reflow_enhancer.speaker_mapping = None
    with pytest.raises(ValueError, match="verified"):
        engine._resolve_enhancer_speaker(0)
    engine.reflow_enhancer.speaker_mapping = {"其他": 0, "甲": 1}
    with pytest.raises(ValueError, match="unknown"):
        engine._resolve_enhancer_speaker(1)


def test_api_rejects_two_selectors():
    from fastapi.testclient import TestClient
    from usvc.api.main import create_app
    with TestClient(create_app(token=None)) as client:
        response = client.post("/api/jobs", json={"kind": "infer", "params": {
            "input": "unused.wav", "config": "unused.yaml", "ckpt": "unused.pth",
            "spk": 0, "speaker": "甲"}})
        assert response.status_code == 422


def test_conflicting_checkpoint_mapping(tmp_path):
    path = tmp_path / "model.pth"
    torch.save({"state_dict": {"spk_embed.weight": torch.randn(2, 32)},
                "usvc_speakers": {"version": 1, "capacity": 2, "spk": {"甲": 0, "乙": 1}}}, path)
    cfg = TrainConfig(backend="ddsp6", backend_opts={"num_speaker": 2}, spk={"乙": 0, "甲": 1})
    with pytest.raises(ValueError, match="mapping mismatch"):
        load_model_speakers(cfg, path)


@pytest.mark.parametrize("kind", ["6.1", "6.2", "6.3", "6.5", "sovits"])
def test_simulated_preprocess_train_export_independent_infer(tmp_path, kind):
    from usvc.backends import get_backend
    from usvc.data.meta import load_meta
    backend = "sovits" if kind == "sovits" else "ddsp6"
    names = ["歌手 甲", "歌手乙"]
    source, sr = sf.read(ROOT / "tests/input/1.wav", dtype="float32")
    if source.ndim == 2:
        source = source.mean(1)
    raw = tmp_path / "raw"
    for sid, name in enumerate(names):
        (raw / name).mkdir(parents=True)
        # Real input excerpts under simulated role labels; these are not real distinct speakers.
        for filename, offset in (("train.wav", 3 + sid), ("val.wav", 5 + sid)):
            count = round(.7 * sr)
            excerpt = source[offset * sr:offset * sr + count]
            assert len(excerpt) == count
            sf.write(raw / name / filename, excerpt, sr)
    dataset = tmp_path / "data"
    exp = tmp_path / "experiment"
    opts = dict(num_speaker=2, n_units=768, f0_extractor="rmvpe", units_kind="contentvec" if kind in ("6.1", "sovits") else "contentvec_tta2x")
    if backend == "ddsp6":
        opts.update(version=kind, n_aux_layers=1, n_aux_chans=32, n_layers=1, n_chans=32, use_pitch_aug=False)
    else:
        opts.update(spec_channels=1025, inter_channels=32, hidden_channels=32, filter_channels=64,
                    gin_channels=32, n_layers_enc=1, n_layers_dec=1, n_sdp_layers0=1, n_sdp_layers=1,
                    segment_size=8, use_automatic_f0_prediction=False)
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(dict(backend=backend, dataset_dir=str(dataset), exp_dir=str(exp),
                      spk={n: i for i, n in enumerate(names)}, backend_opts=opts,
                      batch_size=2, max_frames=12, num_workers=0, augment=False, max_steps=1,
                      save_interval=1, val_interval=2, limit_val_batches=0, precision="32-true"),
                      allow_unicode=True), encoding="utf-8")
    run_cli("usvc.cli.preprocess", ["init", dataset, "--src", raw])
    meta = load_meta(dataset)
    meta.splits = {"train": [f"{n}/train.wav" for n in names], "val": [f"{n}/val.wav" for n in names]}
    meta.save()
    run_cli("usvc.cli.preprocess", ["run", dataset, "--backend", backend, "--train-cfg", config, "--device", "cuda"])
    run_cli("usvc.cli.train", ["--config", config, "--device", "cuda"])
    checkpoint = exp / "last.ckpt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["usvc_speakers"]["spk"] == {n: i for i, n in enumerate(names)}
    cfg = TrainConfig.from_yaml(config)
    torch.manual_seed(123)
    initial = get_backend(backend)().build_module(cfg).state_dict()
    embedding_key = "generator.emb_g.weight" if backend == "sovits" else "model.ddsp_model.unit2ctrl.spk_embed.weight"
    assert (saved["state_dict"][embedding_key] - initial[embedding_key]).abs().sum(1).gt(0).all()
    exported = exp / "model.pth"
    run_cli("usvc.cli.export", ["--ckpt", checkpoint, "-o", exported])
    deployed = exported.with_suffix(".yaml")
    input_path = dataset / "audio" / names[0] / "val.wav"
    common = [input_path, "--config", deployed, "--ckpt", exported, "--device", "cuda", "--no-half", "--steps", "2"]
    outputs = []
    for selector, value in (("--spk", "0"), ("--speaker", names[0]), ("--speaker", names[1])):
        path = tmp_path / f"output{len(outputs)}.wav"
        run_cli("usvc.cli.infer", [*common, selector, value, "-o", path])
        archive_audio(path, dict(backend=backend, version=kind, model=str(exported), target_name=names[int(len(outputs)==2)],
                                target_id=int(len(outputs)==2), device="cuda", seed=123, steps=2,
                                input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(), simulated_roles=True))
        outputs.append(sf.read(path)[0])
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[0], outputs[2])
    engine = InferenceEngine(deployed, exported, device="cuda")
    for bad in (None, True, -1, 2, "未知角色"):
        with pytest.raises(ValueError):
            engine.convert("does-not-exist.wav", spk=bad)
    del engine
    torch.cuda.empty_cache()
    if backend == "sovits":
        from fastapi.testclient import TestClient
        from usvc.api.main import create_app
        from usvc.api.jobs import _infer_argv, JobError
        params = dict(input=str(input_path), config=str(deployed), ckpt=str(exported),
                      output=str(tmp_path / "job.wav"), speaker=names[1], device="cuda", no_half=True)
        with pytest.raises(JobError):
            _infer_argv(dict(params, spk=0))
        with TestClient(create_app(token=None)) as client:
            conflict = client.post("/api/jobs", json={"kind": "infer", "params": dict(params, spk=0)})
            assert conflict.status_code == 422
            response = client.post("/api/jobs", json={"kind": "infer", "params": params})
            assert response.status_code == 201, response.text
            job_id = response.json()["id"]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                state = client.get(f"/api/jobs/{job_id}").json()
                if state["state"] in ("succeeded", "failed", "canceled"):
                    break
                time.sleep(.5)
            print(state, flush=True)
            assert state["state"] == "succeeded"
        archive_audio(Path(params["output"]), dict(backend=backend, model=str(exported), target_name=names[1],
                      target_id=1, device="cuda", simulated_roles=True, api_job=job_id))
