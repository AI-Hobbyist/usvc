"""M1 contracts. Fixture files persist; no teardown removes model resources."""
import hashlib
import json
from pathlib import Path
import uuid

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from usvc.infer_api.app import create_app
from usvc.infer_api.catalog import Catalog
from usvc.infer_api.fixture_engine import validate_network
from infer_api_fixtures import MATRIX, generate

ROOT = Path(__file__).resolve().parents[1]


def controlled_root():
    root = ROOT / "models" / "_contract" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def model_copy(root, backend="ddsp6", name="中文 模型"):
    directory = root / backend / name
    directory.mkdir(parents=True)
    source = ROOT / "models" / backend / ("__test_ddsp65_single" if backend == "ddsp6" else "__test_sovits_single")
    for file in source.iterdir():
        if file.is_file():
            (directory / file.name).write_bytes(file.read_bytes())
    return directory


def test_seven_networks_and_idempotent_generation():
    catalog = Catalog(ROOT / "models", lambda model: None)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for b, n, _ in MATRIX for p in (ROOT / "models" / b / n).iterdir() if p.is_file()}
    assert generate(ROOT / "models") == []
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    for backend, name, count in MATRIX:
        model = catalog._read(ROOT / "models" / backend / name, backend)
        outputs = validate_network(model, "cuda")
        assert len(outputs) == count
        assert model.fixture and len(model.speakers) == count
        # Actual network conditioning is observed directly, separately from sine audio.
        from usvc.infer_api.fixture_engine import load_network
        network = load_network(model, "cuda")
        seen = []
        embedding = network.generator.emb_g if backend == "sovits" else getattr(network.model.ddsp_model.unit2ctrl, "spk_embed", None) if backend == "ddsp6" else None
        if embedding is not None:
            handle = embedding.register_forward_pre_hook(lambda module, args: seen.extend(args[0].detach().cpu().flatten().tolist()))
            from usvc.train.dataset import SVCBatch
            for sid in range(count):
                batch = SVCBatch(units=torch.randn(1, 16, 768, device="cuda"), f0=torch.full((1, 16), 180., device="cuda"),
                                 rms=torch.full((1, 16), .1, device="cuda"), uv=torch.ones(1, 16, device="cuda"),
                                 spk_id=torch.tensor([sid], device="cuda"), frame_len=torch.tensor([16], device="cuda"))
                with torch.inference_mode():
                    network.sample(batch) if backend == "sovits" else network.sample(batch, steps=2)
            handle.remove()
            assert set(seen) == set(range(count))
        del network
        if backend == "sovits":
            for sid, output in enumerate(outputs):
                import io
                buf = io.BytesIO()
                sf.write(buf, output.flatten().numpy(), 44100, format="WAV", subtype="PCM_16")
                raw = buf.getvalue()
                digest = hashlib.sha256(raw).hexdigest()
                path = ROOT / "tests/output" / f"{digest}.wav"
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    path.write_bytes(raw)
                print(f"AUDIO: {path} (M1 independent random-weight network, {name}, speaker={sid})", flush=True)
        torch.cuda.empty_cache()


@pytest.mark.parametrize("mapping", [None, {}, {"甲": 0}])
def test_single_speaker_defaults(mapping):
    root = controlled_root()
    directory = model_copy(root)
    cfg = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    if mapping is None:
        cfg.pop("spk")
    else:
        cfg["spk"] = mapping
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    payload = torch.load(directory / "model.ckpt", weights_only=False)
    payload.pop("usvc_speakers")
    torch.save(payload, directory / "model.ckpt")
    catalog = Catalog(root, lambda model: None)
    found = catalog.refresh()
    assert len(found) == 1 and not found[0]["multi_speaker"]
    assert found[0]["speakers"] == [{"id": 0, "name": "甲" if mapping else "default"}]


@pytest.mark.parametrize("fault", ["backend", "mapping", "capacity", "checkpoint_mapping", "config_choice", "checkpoint_choice", "escape", "missing"])
def test_bad_models_filtered(fault):
    root = controlled_root()
    directory = model_copy(root)
    cfgpath, metapath = directory / "config.yaml", directory / "model.json"
    cfg = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
    meta = json.loads(metapath.read_text())
    if fault == "backend":
        cfg["backend"] = "sovits"
    elif fault == "mapping":
        cfg["spk"] = {"甲": 2}
    elif fault == "capacity":
        cfg["backend_opts"]["num_speaker"] = 2
    elif fault == "checkpoint_mapping":
        cfg["spk"] = {"其他": 0}
    elif fault == "config_choice":
        (directory / "other.yml").write_bytes(cfgpath.read_bytes())
        meta.pop("config")
    elif fault == "checkpoint_choice":
        (directory / "other.pth").write_bytes((directory / "model.ckpt").read_bytes())
        meta.pop("checkpoint")
    elif fault == "escape":
        meta["checkpoint"] = "../../../../outside.pt"
    else:
        meta["checkpoint"] = "missing.pth"
    cfgpath.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    metapath.write_text(json.dumps(meta), encoding="utf-8")
    catalog = Catalog(root, lambda model: None)
    assert catalog.refresh() == [] and catalog.diagnostics


@pytest.mark.parametrize("token", [None, "", "test-secret"])
def test_http_and_authentication(token):
    root = controlled_root()
    model_copy(root)
    (root / "reflow_shallow" / "enhancer").mkdir(parents=True)
    calls = []
    app = create_app(root, token=token, validator=lambda model: calls.append(model.id), probe_models=True)
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer test-secret"} if token else {}
        if token:
            for authorization in (None, "Basic test-secret", "Bearer wrong"):
                response = client.get("/model", headers={"Authorization": authorization} if authorization else {})
                assert response.status_code == 401 and response.headers["WWW-Authenticate"] == "Bearer"
            assert not calls
        response = client.get("/model", headers=headers)
        assert response.status_code == 200 and isinstance(response.json(), list) and len(response.json()) == 1
        identity = response.json()[0]["id"]
        assert client.get("/settings", headers=headers).status_code == 422
        assert client.get("/settings", params={"backend": "ddsp6", "model": identity}, headers=headers).status_code == 200
        assert client.get("/settings", params={"backend": "sovits", "model": identity}, headers=headers).status_code == 422
        assert client.get("/settings", params={"backend": "sovits"}, headers=headers).status_code == 404
        assert client.get("/settings", params={"backend": "ddsp6", "model": "missing"}, headers=headers).status_code == 404
        assert len(calls) == 1


@pytest.mark.parametrize("probe_models", [False, True])
def test_discovery_probe_option_and_selected_model_validation(probe_models):
    root = controlled_root()
    directory = model_copy(root)
    calls = []
    app = create_app(root, validator=lambda model: calls.append(model.id),
                     **({"probe_models": True} if probe_models else {}))
    with TestClient(app) as client:
        response = client.get("/model")
        assert response.status_code == 200 and len(response.json()) == 1
        identity = response.json()[0]["id"]
        assert len(calls) == int(probe_models)
        assert client.get("/settings", params={"backend": "ddsp6", "model": identity}).status_code == 200
        assert len(calls) == int(probe_models)
        config = directory / "config.yaml"
        config.write_text(config.read_text(encoding="utf-8") + "\n# resource changed\n", encoding="utf-8")
        assert client.get("/model").status_code == 200
        assert len(calls) == 2 * int(probe_models)
        catalog = app.state.catalog
        catalog.ensure_ready(catalog.candidate(identity))
        assert len(calls) == (2 if probe_models else 1)
        catalog.ensure_ready(catalog.candidate(identity))
        assert len(calls) == (2 if probe_models else 1)


def test_empty_catalog():
    root = controlled_root()
    with TestClient(create_app(root, validator=lambda model: None)) as client:
        assert client.get("/model").json() == []


def test_multiple_resources_and_default_selection():
    root = controlled_root()
    directory = model_copy(root)
    (directory / "extra.yml").write_bytes((directory / "config.yaml").read_bytes())
    (directory / "extra.pth").write_bytes((directory / "model.ckpt").read_bytes())
    for name in ("index甲.pt", "index乙.pt"):
        torch.save({"vectors": torch.ones(2, 768)}, directory / name)
    torch.save({"vectors": torch.ones(2, 256)}, directory / "index坏.pt")
    catalog = Catalog(root, lambda model: None)
    found = catalog.refresh()
    assert len(found) == 1 and found[0]["default_index"] is None
    assert set(found[0]["indexes"]) == {"index甲.pt", "index乙.pt"}
    meta = json.loads((directory / "model.json").read_text())
    meta["default_index"] = "index乙.pt"
    (directory / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    assert catalog.refresh()[0]["default_index"] == "index乙.pt"


def test_multi_selection_and_invalid_defaults():
    root = controlled_root()
    directory = root / "ddsp6" / "多人"
    directory.mkdir(parents=True)
    for file in (ROOT / "models/ddsp6/__test_ddsp65_multi").iterdir():
        (directory / file.name).write_bytes(file.read_bytes())
    catalog = Catalog(root, lambda model: None)
    assert catalog.refresh()[0]["multi_speaker"]
    from usvc.infer_api.settings import validate
    from usvc.infer_api.errors import APIError
    model = catalog.get("ddsp6/多人")
    with pytest.raises(APIError):
        validate(model, {}, "cpu")
    assert validate(model, {"speaker": "测试乙"}, "cpu")["speaker"] == 1
    for bad in (-1, 2, True, "未知"):
        with pytest.raises(APIError):
            validate(model, {"speaker": bad}, "cpu")
    meta = json.loads((directory / "model.json").read_text())
    meta["default_speaker"] = "不存在"
    (directory / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    assert catalog.refresh() == []


@pytest.mark.parametrize("args", [["--port", "0"], ["--port", "65536"], ["--host", "bad host"], ["--model-root", "missing-root"]])
def test_cli_validation(args):
    from usvc.infer_api.cli import main
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
