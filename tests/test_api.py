"""usvc-serve API 骨架测试：只读接口 + Job 生命周期 + SSE + 取消 + 鉴权。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from usvc.api.main import create_app

TERMINAL = ("succeeded", "failed", "canceled")


@pytest.fixture()
def client():
    app = create_app(token=None)
    with TestClient(app) as c:
        yield c


def _wait_terminal(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        snap = r.json()
        if snap["state"] in TERMINAL:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} not terminal within {timeout}s")


def _sse_events(client, job_id: str, from_seq: int = 0) -> list[dict]:
    """完整消费一条 SSE 流（直到关闭），返回事件列表。"""
    events: list[dict] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events", params={"from_seq": from_seq}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


# ---------- 只读接口 ----------


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == "usvc-serve"


def test_env(client):
    r = client.get("/api/env")
    assert r.status_code == 200
    body = r.json()
    report = body["report"]
    for key in ("platform", "python", "torch", "torch_cuda", "triton", "gpu", "compile_ok"):
        assert key in report
    assert isinstance(body["verdict"], list)


def test_backends(client):
    r = client.get("/api/backends")
    assert r.status_code == 200
    names = [b["name"] for b in r.json()["backends"]]
    assert "rvc" in names and "sovits" in names and len(names) >= 7


def test_presets(client, monkeypatch, tmp_path):
    (tmp_path / "mypreset.yaml").write_text(
        "backend: rvc\nbatch_size: 8\nmax_steps: 1000\n", encoding="utf-8"
    )
    monkeypatch.setenv("USVC_CONFIGS_DIR", str(tmp_path))
    r = client.get("/api/presets")
    assert r.status_code == 200
    body = r.json()
    assert body["dir"] == str(tmp_path)
    assert len(body["presets"]) == 1
    preset = body["presets"][0]
    assert preset["name"] == "mypreset"
    assert preset["backend"] == "rvc"
    assert preset["batch_size"] == 8


# ---------- Job 生命周期 ----------


def test_job_echo_success(client):
    r = client.post("/api/jobs", json={"kind": "echo"})
    assert r.status_code == 201
    snap = r.json()
    assert snap["state"] in ("pending", "running", "succeeded")
    snap = _wait_terminal(client, snap["id"])
    assert snap["state"] == "succeeded"
    assert snap["exit_code"] == 0


def test_job_checkenv_success(client):
    r = client.post("/api/jobs", json={"kind": "checkenv"})
    assert r.status_code == 201
    snap = _wait_terminal(client, r.json()["id"], timeout=120)
    assert snap["state"] == "succeeded"


def test_job_unknown_kind(client):
    r = client.post("/api/jobs", json={"kind": "nope"})
    assert r.status_code == 422


def test_job_not_found(client):
    assert client.get("/api/jobs/abcdef123456").status_code == 404
    assert client.delete("/api/jobs/abcdef123456").status_code == 404
    assert client.get("/api/jobs/abcdef123456/events").status_code == 404


def test_jobs_list_filter(client):
    client.post("/api/jobs", json={"kind": "echo"})
    client.post("/api/jobs", json={"kind": "echo"})
    r = client.get("/api/jobs")
    assert len(r.json()["jobs"]) >= 2
    r = client.get("/api/jobs", params={"state": "pending"})
    assert all(j["state"] == "pending" for j in r.json()["jobs"])


def test_job_cancel_running(client):
    r = client.post("/api/jobs", json={"kind": "sleep"})
    job_id = r.json()["id"]
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["state"] in ("canceled", "running")
    snap = _wait_terminal(client, job_id, timeout=15)
    assert snap["state"] == "canceled"


# ---------- SSE replay-then-tail ----------


def test_sse_replay_after_done(client):
    job_id = client.post("/api/jobs", json={"kind": "echo"}).json()["id"]
    _wait_terminal(client, job_id)
    events = _sse_events(client, job_id)
    assert events, "SSE 流应回放出事件"
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs), "seq 必须单调递增"
    types = [e["type"] for e in events]
    assert "state" in types
    logs = [e for e in events if e["type"] == "log"]
    assert logs, "应有日志事件"
    assert any(e["payload"].get("stream") == "stdout" for e in logs)
    assert events[-1]["type"] == "state"
    assert events[-1]["payload"]["state"] == "succeeded"
    # 首事件含 running，末事件含终态
    states = [e["payload"]["state"] for e in events if e["type"] == "state"]
    assert "running" in states and "succeeded" in states


def test_sse_from_seq_partial(client):
    job_id = client.post("/api/jobs", json={"kind": "echo"}).json()["id"]
    _wait_terminal(client, job_id)
    events = _sse_events(client, job_id)
    cut = events[len(events) // 2]["seq"]
    tail = _sse_events(client, job_id, from_seq=cut)
    assert tail
    assert all(e["seq"] > cut for e in tail)
    assert tail[-1] == events[-1], "末事件（终态）必须不丢"


def test_sse_live_stream(client):
    """边跑边连：应拿到全部事件且无重复（订阅窗口去重）。"""
    job_id = client.post("/api/jobs", json={"kind": "echo"}).json()["id"]
    # 立即开流（job 还在跑），从 seq=0 回放
    events = _sse_events(client, job_id)
    assert events[-1]["type"] == "state"
    assert events[-1]["payload"]["state"] == "succeeded"
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(set(seqs)), "实时流 + 回放不得重复事件"


# ---------- 鉴权 ----------


def test_token_auth():
    app = create_app(token="sekrit")
    with TestClient(app) as c:
        # health 豁免
        assert c.get("/api/health").status_code == 200
        # 无 token 被拒
        assert c.get("/api/env").status_code == 401
        assert c.post("/api/jobs", json={"kind": "echo"}).status_code == 401
        # 正确 token 通过
        h = {"X-Usvc-Token": "sekrit"}
        assert c.get("/api/env", headers=h).status_code == 200
        r = c.post("/api/jobs", json={"kind": "echo"}, headers=h)
        assert r.status_code == 201
