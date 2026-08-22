"""usvc-serve: local API for the WebUI / usvc-studio desktop shell (WEBUI_DESIGN §3).

Read-only: /api/health|env|backends|presets. Unified Jobs: /api/jobs CRUD + SSE replay-then-tail.
Optional token auth (X-Usvc-Token); CORS fully open (browser mode). Startup contract (desktop
parses stdout line 1)::    USVC_READY {"port": 9740, "token": null}"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import socket
import time
from pathlib import Path

import uvicorn
import yaml as _yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import _envcheck
from ..backends import BACKENDS
from .audio import AudioError, AudioStore
from .jobs import TERMINAL_STATES, JobError, JobScheduler
from .tfevents import TfEventsStore

DEFAULT_PORT = 9740

_AUDIO_MEDIA = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}

# train overrides whitelist — plan/location fields only; batch_size/max_frames/loss/arch/optimizer
# are frozen training semantics (AGENTS.md) and must never be overridable from the API.
_TRAIN_OVERLAY_KEYS = {
    "dataset_dir", "exp_dir", "resume", "max_steps", "max_epochs",
    "save_interval", "val_interval", "keep_last_n", "num_workers",
}


def _version() -> str:
    try: return importlib.metadata.version("usvc")
    except importlib.metadata.PackageNotFoundError: return "0.0.0"


def _presets_dir() -> Path | None:  # configs/: env var wins, else cwd
    d = os.environ.get("USVC_CONFIGS_DIR")
    if d: return Path(d)
    cwd = Path.cwd() / "configs"
    return cwd if cwd.is_dir() else None


def _outputs_dir() -> Path:
    d = os.environ.get("USVC_OUTPUT_DIR")
    return Path(d) if d else Path.cwd() / "outputs"


def _sse(ev) -> str:
    data = json.dumps(ev.to_dict(), ensure_ascii=False)
    return f"id: {ev.seq}\nevent: {ev.type}\ndata: {data}\n\n"


class JobCreate(BaseModel):
    kind: str
    params: dict = Field(default_factory=dict)


class AudioRegister(BaseModel):
    path: str


def _normalize_train(params: dict, out_dir: Path) -> dict:
    """Load train yaml + apply whitelist overrides -> write temp yaml -> inject exp_dir."""
    cfg_path = Path(params["config"])
    if not cfg_path.is_file(): raise JobError(f"config not found: {cfg_path}")
    try: data = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except _yaml.YAMLError as exc:
        raise JobError(f"invalid yaml {cfg_path}: {exc}")
    if not isinstance(data, dict) or "backend" not in data:
        raise JobError(f"config missing 'backend' key: {cfg_path}")

    overrides = dict(params.get("overrides") or {})
    bad = set(overrides) - _TRAIN_OVERLAY_KEYS
    if bad:
        raise JobError(f"unsupported train overrides: {sorted(bad)} (allowed: {sorted(_TRAIN_OVERLAY_KEYS)})")
    data.update(overrides)

    exp_dir = str(data.get("exp_dir") or "")
    cfg_dir = out_dir / "train_cfgs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg_dir / f"train_{int(time.time() * 1000)}.yaml"
    tmp.write_text(_yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    new_params = dict(params); new_params["config"] = str(tmp); new_params["exp_dir"] = exp_dir
    if "resume" in overrides: new_params["resume"] = overrides["resume"]
    return new_params


def create_app(token: str | None = None) -> FastAPI:
    app = FastAPI(title="usvc-serve", version=_version())
    # outermost CORS: browser mode needs cross-origin; local token guards other pages from hijacking 127.0.0.1
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    if token:
        @app.middleware("http")
        async def _auth(request: Request, call_next):
            if request.method == "OPTIONS" or request.url.path == "/api/health":
                return await call_next(request)
            if request.headers.get("x-usvc-token") != token:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            return await call_next(request)

    scheduler = JobScheduler(); app.state.scheduler = scheduler
    audio_store = AudioStore(); app.state.audio = audio_store
    tfevents = TfEventsStore(); app.state.tfevents = tfevents
    def _on_job_success(job) -> None:
        """infer success: register output audio + artifact event (before terminal state)."""
        if job.kind != "infer": return
        out = job.params.get("output")
        if not out or not Path(out).exists(): return
        try: rec = audio_store.register(out, job_id=job.id, kind="infer_output")
        except AudioError: return
        job.emit("artifact", kind="wav", path=rec["path"], audio_id=rec["id"], name=rec["name"])

    scheduler.on_success = _on_job_success
    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "service": "usvc-serve", "version": _version()}
    @app.get("/api/env")
    async def env() -> dict:
        report = _envcheck.full_report()
        return {"report": report, "verdict": _envcheck._verdict(report)}
    @app.get("/api/backends")
    async def backends() -> dict:
        return {"backends": [{"name": n} for n in BACKENDS]}
    @app.get("/api/presets")
    async def presets() -> dict:
        base = _presets_dir()
        if base is None:
            return {"dir": None, "presets": [],
                    "note": "configs/ 目录未找到（从仓库根启动或设置 USVC_CONFIGS_DIR）"}
        out = []
        for p in sorted(base.glob("*.yaml")):
            try: data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except _yaml.YAMLError: data = {}
            out.append({"name": p.stem, "path": str(p),
                        **{k: data.get(k) for k in
                           ("backend", "dataset_dir", "exp_dir", "batch_size", "max_steps")}})
        return {"dir": str(base), "presets": out}
    @app.get("/api/jobs")
    async def jobs_list(state: str | None = None) -> dict:
        jobs = [j.snapshot() for j in scheduler.list()]
        if state: jobs = [j for j in jobs if j["state"] == state]
        return {"jobs": jobs}
    @app.post("/api/jobs", status_code=201)
    async def jobs_create(body: JobCreate) -> dict:
        params = dict(body.params)
        out_dir = _outputs_dir()
        if body.kind == "infer" and not params.get("output"):
            # route output into the controlled dir for registration/playback
            stem = Path(str(params.get("input", "out"))).stem or "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            params["output"] = str(out_dir / f"out_{stem}_{int(time.time())}.wav")
        elif body.kind == "train":
            try: params = _normalize_train(params, out_dir)
            except JobError as exc: raise HTTPException(status_code=422, detail=str(exc))
        try: job = await scheduler.submit(body.kind, params)
        except JobError as exc: raise HTTPException(status_code=422, detail=str(exc))
        return job.snapshot()
    @app.get("/api/jobs/{job_id}")
    async def jobs_get(job_id: str) -> dict:
        job = scheduler.get(job_id)
        if job is None: raise HTTPException(status_code=404, detail="job not found")
        return job.snapshot()
    @app.delete("/api/jobs/{job_id}")
    async def jobs_cancel(job_id: str) -> dict:
        job = await scheduler.cancel(job_id)
        if job is None: raise HTTPException(status_code=404, detail="job not found")
        return job.snapshot()

    # ---------- audio artifacts ----------
    @app.post("/api/audio", status_code=201)
    async def audio_register(body: AudioRegister) -> dict:
        """Register a server-side audio file (in/out), returns playable audio id."""
        try: return audio_store.register(body.path)
        except AudioError as exc: raise HTTPException(status_code=422, detail=str(exc))
    @app.get("/api/audio")
    async def audio_list() -> dict:
        return {"items": audio_store.list()}
    @app.get("/api/audio/{audio_id}")
    async def audio_stream(audio_id: str) -> FileResponse:
        item = audio_store.get(audio_id)
        if item is None: raise HTTPException(status_code=404, detail="audio not found")
        media = _AUDIO_MEDIA.get(Path(item["path"]).suffix.lower(), "application/octet-stream")
        return FileResponse(item["path"], media_type=media, filename=item["name"])  # Range-capable
    @app.get("/api/audio/{audio_id}/peaks")
    async def audio_peaks(audio_id: str, n: int = Query(800, ge=16, le=8192)) -> dict:
        result = await asyncio.to_thread(audio_store.peaks, audio_id, n)
        if result is None: raise HTTPException(status_code=404, detail="audio not found")
        return result
    @app.get("/api/jobs/{job_id}/metrics")
    async def job_metrics(job_id: str) -> dict:
        job = scheduler.get(job_id)
        if job is None: raise HTTPException(status_code=404, detail="job not found")
        exp_dir = (job.params or {}).get("exp_dir")
        if not exp_dir: raise HTTPException(status_code=404, detail="no exp_dir for this job")
        return await asyncio.to_thread(tfevents.scalars, exp_dir)
    @app.get("/api/jobs/{job_id}/events")
    async def jobs_events(job_id: str, from_seq: int = Query(0, ge=0)) -> StreamingResponse:
        """SSE replay-then-tail: history after from_seq first, then live."""
        job = scheduler.get(job_id)
        if job is None: raise HTTPException(status_code=404, detail="job not found")
        async def gen():
            q: asyncio.Queue = asyncio.Queue(maxsize=4096)
            job.subscribe(q)  # subscribe before replay; window overlap deduped by seq below
            try:
                last = from_seq
                for ev in job.events_after(last):
                    yield _sse(ev); last = ev.seq
                if job.state in TERMINAL_STATES: return
                while True:
                    ev = await q.get()
                    if ev is None: return  # terminal sentinel
                    if ev.seq <= last: continue  # already replayed within subscribe window
                    last = ev.seq
                    yield _sse(ev)
            finally:
                job.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="usvc-serve", description="usvc WebUI/桌面端本地 API 服务")
    p.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口；0=自动选空闲端口")
    p.add_argument("--token", default=None, help="鉴权 token（默认取 USVC_SERVE_TOKEN 环境变量）")
    p.add_argument("--no-token", action="store_true", help="强制关闭鉴权")
    args = p.parse_args(argv)

    token = None if args.no_token else (args.token or os.environ.get("USVC_SERVE_TOKEN"))

    # bind first to learn the actual port (--port 0), hand the socket to uvicorn: no race
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); sock.bind((args.host, args.port))
    actual_port = sock.getsockname()[1]

    print(f"USVC_READY {json.dumps({'port': actual_port, 'token': token}, ensure_ascii=False)}",
          flush=True)  # startup contract: stdout line 1, parsed by desktop/client

    server = uvicorn.Server(uvicorn.Config(create_app(token=token), host=args.host,
                                           port=actual_port, log_level="info"))
    server.config.sockets = [sock]; server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
