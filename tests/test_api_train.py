"""P2 测试：train job（argv 映射 / overrides 白名单 / 临时 yaml）+ tfevents 曲线。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from usvc.api.jobs import JobError, _argv_for
from usvc.api.main import create_app

TERMINAL = ("succeeded", "failed", "canceled")


@pytest.fixture()
def client():
    app = create_app(token=None)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def train_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "backend: sovits\n"
        "dataset_dir: data/nahida\n"
        "exp_dir: experiments/nahida_sovits\n"
        "batch_size: 48\n"
        "max_frames: 64\n"
        "max_steps: 15000\n"
        "val_interval: 5000\n"
        "save_interval: 5000\n"
        "precision: bf16-mixed\n",
        encoding="utf-8",
    )
    return p


def _wait_terminal(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["state"] in TERMINAL:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} not terminal within {timeout}s")


# ---------- argv 映射 ----------


def test_train_argv_mapping():
    argv = _argv_for("train", {"config": "c.yaml", "resume": "r.ckpt", "device": "cuda:0"})
    s = " ".join(argv)
    assert "-m usvc.cli.train --config c.yaml" in s
    assert "--resume r.ckpt" in s and "--device cuda:0" in s

    argv = _argv_for("train", {"config": "c.yaml"})
    s = " ".join(argv)
    assert "--resume" not in s and "--device" not in s


def test_train_argv_requires_config():
    with pytest.raises(JobError):
        _argv_for("train", {})
    with pytest.raises(JobError):
        _argv_for("train", {"resume": "r.ckpt"})


# ---------- overrides 白名单 / 临时 yaml ----------


def test_train_overrides_whitelist(client, train_yaml, monkeypatch, tmp_path):
    monkeypatch.setenv("USVC_OUTPUT_DIR", str(tmp_path))
    # batch_size 属于训练语义边界，必须拒绝
    r = client.post("/api/jobs", json={
        "kind": "train",
        "params": {"config": str(train_yaml), "overrides": {"batch_size": 8}},
    })
    assert r.status_code == 422
    assert "batch_size" in r.json()["detail"]

    # 未知字段也拒绝
    r = client.post("/api/jobs", json={
        "kind": "train",
        "params": {"config": str(train_yaml), "overrides": {"foo": 1}},
    })
    assert r.status_code == 422


def test_train_normalize_writes_temp_yaml(client, train_yaml, monkeypatch, tmp_path):
    monkeypatch.setenv("USVC_OUTPUT_DIR", str(tmp_path))
    from usvc.api import jobs as jobs_mod

    # 不真跑训练：替换 argv（生命周期仍走完）
    monkeypatch.setattr(
        jobs_mod, "_argv_for", lambda kind, params: [sys.executable, "-c", "print('ok')"]
    )
    r = client.post("/api/jobs", json={
        "kind": "train",
        "params": {
            "config": str(train_yaml),
            "overrides": {"exp_dir": "/tmp/webui_exp", "max_steps": 500},
        },
    })
    assert r.status_code == 201
    snap = r.json()
    assert snap["gpu"] is True
    tmp_cfg = Path(snap["params"]["config"])
    assert tmp_cfg.is_file()
    assert tmp_cfg.suffix == ".yaml"
    assert str(tmp_path) in str(tmp_cfg)  # 在受控输出目录下
    assert snap["params"]["exp_dir"] == "/tmp/webui_exp"
    # 临时 yaml 已应用 overrides 且保留其余字段
    import yaml

    data = yaml.safe_load(tmp_cfg.read_text(encoding="utf-8"))
    assert data["max_steps"] == 500 and data["batch_size"] == 48
    _wait_terminal(client, snap["id"])


def test_train_config_missing(client, monkeypatch, tmp_path):
    monkeypatch.setenv("USVC_OUTPUT_DIR", str(tmp_path))
    r = client.post("/api/jobs", json={
        "kind": "train",
        "params": {"config": "/nonexistent.yaml"},
    })
    assert r.status_code == 422
    assert "not found" in r.json()["detail"]


# ---------- metrics（tfevents 解析）----------


@pytest.fixture()
def tfevents_exp(tmp_path):
    """合成一个 lightning_logs/version_0 带标量的 exp 目录。"""
    import torch.utils.tensorboard as tb

    exp = tmp_path / "exp"
    v0 = exp / "lightning_logs" / "version_0"
    v0.mkdir(parents=True)
    w = tb.SummaryWriter(log_dir=str(v0))
    for step in range(1, 21):
        w.add_scalar("train/loss", 1.0 / step, step)
        w.add_scalar("train/lr", 0.001, step)
    w.close()
    return exp


def test_metrics_parses_tfevents(client, tfevents_exp, monkeypatch, tmp_path):
    monkeypatch.setenv("USVC_OUTPUT_DIR", str(tmp_path))
    from usvc.api import jobs as jobs_mod

    monkeypatch.setattr(
        jobs_mod, "_argv_for", lambda kind, params: [sys.executable, "-c", "print('ok')"]
    )
    r = client.post("/api/jobs", json={
        "kind": "train",
        "params": {"config": str(tmp_path / "c.yaml"), "overrides": {"exp_dir": str(tfevents_exp)}},
    })
    # config 校验先行：合成 exp 的 config 文件不存在 → 422；直接构造合法提交
    assert r.status_code == 422  # 缺 config yaml（我们没建）

    # 构造一个合法 config 再测
    cfg = tmp_path / "c.yaml"
    cfg.write_text("backend: sovits\nexp_dir: experiments/x\nmax_steps: 100\n", encoding="utf-8")
    r = client.post("/api/jobs", json={
        "kind": "train",
        "params": {"config": str(cfg), "overrides": {"exp_dir": str(tfevents_exp)}},
    })
    assert r.status_code == 201
    jid = r.json()["id"]
    _wait_terminal(client, jid)

    m = client.get(f"/api/jobs/{jid}/metrics")
    assert m.status_code == 200
    data = m.json()
    assert data["version"] == "version_0"
    assert "train/loss" in data["tags"]
    pts = data["tags"]["train/loss"]
    assert len(pts) == 20
    assert pts[0] == [1, 1.0]
    assert abs(pts[-1][1] - 0.05) < 1e-6
    assert len(data["tags"]["train/lr"]) == 20


def test_metrics_job_not_found(client):
    assert client.get("/api/jobs/deadbeef/metrics").status_code == 404


def test_metrics_no_exp_dir(client, monkeypatch, tmp_path):
    monkeypatch.setenv("USVC_OUTPUT_DIR", str(tmp_path))
    from usvc.api import jobs as jobs_mod

    monkeypatch.setattr(
        jobs_mod, "_argv_for", lambda kind, params: [sys.executable, "-c", "print('ok')"]
    )
    r = client.post("/api/jobs", json={"kind": "echo", "params": {}})
    jid = r.json()["id"]
    _wait_terminal(client, jid)
    assert client.get(f"/api/jobs/{jid}/metrics").status_code == 404
