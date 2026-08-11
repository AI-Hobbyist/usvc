"""P0.5 测试：音频注册/峰值/流 + infer job argv 映射 + 输出规范化。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
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
def wav_file(tmp_path):
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, data, sr)
    return p


def _wait_terminal(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["state"] in TERMINAL:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} not terminal within {timeout}s")


# ---------- 音频注册 / 峰值 / 流 ----------


def test_audio_register_and_peaks(client, wav_file):
    r = client.post("/api/audio", json={"path": str(wav_file)})
    assert r.status_code == 201
    item = r.json()
    assert item["name"] == "tone.wav"
    assert item["size"] > 0

    # 同一路径幂等注册
    r2 = client.post("/api/audio", json={"path": str(wav_file)})
    assert r2.json()["id"] == item["id"]

    # 峰值
    r = client.get(f"/api/audio/{item['id']}/peaks", params={"n": 64})
    assert r.status_code == 200
    peaks = r.json()
    assert peaks["n"] == 64
    assert len(peaks["peaks"]) == 64
    assert all(len(pair) == 2 for pair in peaks["peaks"])
    assert abs(peaks["duration"] - 0.5) < 0.01
    assert peaks["sample_rate"] == 22050
    # 正弦波峰值应接近 ±0.5
    mx = max(p[1] for p in peaks["peaks"])
    assert 0.4 < mx <= 0.5

    # 流式播放
    r = client.get(f"/api/audio/{item['id']}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert len(r.content) == item["size"]

    # 列表
    r = client.get("/api/audio")
    assert any(a["id"] == item["id"] for a in r.json()["items"])


def test_audio_register_rejects_bad_input(client, tmp_path):
    r = client.post("/api/audio", json={"path": str(tmp_path / "nope.wav")})
    assert r.status_code == 422
    txt = tmp_path / "note.txt"
    txt.write_text("hi")
    r = client.post("/api/audio", json={"path": str(txt)})
    assert r.status_code == 422


def test_audio_not_found(client):
    assert client.get("/api/audio/abcdef123456").status_code == 404
    assert client.get("/api/audio/abcdef123456/peaks").status_code == 404


# ---------- infer argv 映射 ----------


def test_infer_argv_full_mapping():
    argv = _argv_for("infer", {
        "input": "in.wav", "config": "c.yaml", "ckpt": "m.ckpt", "output": "o.wav",
        "trans": 2, "protect": 0.33, "noice_scale": 0.5, "f0_method": "fcpe",
        "half": True, "auto_predict_f0": True, "index_path": "idx.pt", "index_ratio": 0.7,
        "shallow_diffusion": True, "k_step": 50, "target_loudness": -18.0,
    })
    s = " ".join(argv)
    assert "-m usvc.cli.infer in.wav --config c.yaml --ckpt m.ckpt" in s
    assert "--output o.wav" in s
    assert "--trans 2" in s and "--protect 0.33" in s and "--noice-scale 0.5" in s
    assert "--f0-method fcpe" in s
    assert "--half" in s and "--auto-predict-f0" in s
    assert "--index-path idx.pt" in s and "--index-ratio 0.7" in s
    assert "--shd" in s and "--k-step 50" in s
    assert "--target-loudness -18.0" in s


def test_infer_argv_omits_none_and_false():
    argv = _argv_for("infer", {"input": "i.wav", "config": "c.yaml", "ckpt": "m.ckpt"})
    s = " ".join(argv)
    assert "--trans" not in s and "--half" not in s and "--output" not in s
    assert "--protect" not in s and "--shd" not in s


def test_infer_argv_requires_params():
    with pytest.raises(JobError):
        _argv_for("infer", {"input": "i.wav"})


# ---------- infer job 提交（输出规范化 + 必填校验）----------


def test_infer_job_requires_params(client):
    r = client.post("/api/jobs", json={"kind": "infer", "params": {"input": "x.wav"}})
    assert r.status_code == 422


def test_infer_job_output_normalization(client, monkeypatch, tmp_path):
    monkeypatch.setenv("USVC_OUTPUT_DIR", str(tmp_path))
    # 替换 argv，避免真的起推理（只验证规范化与生命周期）
    from usvc.api import jobs as jobs_mod

    monkeypatch.setattr(
        jobs_mod, "_argv_for", lambda kind, params: [sys.executable, "-c", "print('ok')"]
    )
    r = client.post("/api/jobs", json={
        "kind": "infer",
        "params": {"input": "/data/纳西妲_demo.wav", "config": "c.yaml", "ckpt": "m.ckpt"},
    })
    assert r.status_code == 201
    snap = r.json()
    assert snap["gpu"] is True
    assert snap["params"]["output"].startswith(str(tmp_path))
    assert "out_纳西妲_demo_" in snap["params"]["output"]
    assert snap["params"]["output"].endswith(".wav")
    _wait_terminal(client, snap["id"])
