"""usvc-serve：usvc WebUI / 桌面端（usvc-studio）的本地 API 服务。

P0 骨架（docs/WEBUI_DESIGN.md §3）：
- 只读接口：/api/health /api/env /api/backends /api/presets
- 统一 Job 抽象：/api/jobs（提交/列表/详情/取消）+ SSE 事件流（replay-then-tail）
- 可选 token 鉴权（X-Usvc-Token）；CORS 全开（浏览器模式需要）

启动契约（桌面端解析 stdout 首行）::

    USVC_READY {"port": 9740, "token": null}
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import socket
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import _envcheck
from ..backends import BACKENDS
from .jobs import TERMINAL_STATES, JobError, JobScheduler

__all__ = ["create_app", "main"]

DEFAULT_PORT = 9740


def _version() -> str:
    try:
        return importlib.metadata.version("usvc")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _presets_dir() -> Path | None:
    """configs/ 目录解析：环境变量优先，其次当前工作目录。"""
    d = os.environ.get("USVC_CONFIGS_DIR")
    if d:
        return Path(d)
    cwd = Path.cwd() / "configs"
    return cwd if cwd.is_dir() else None


def _sse(ev) -> str:
    data = json.dumps(ev.to_dict(), ensure_ascii=False)
    return f"id: {ev.seq}\nevent: {ev.type}\ndata: {data}\n\n"


class JobCreate(BaseModel):
    kind: str
    params: dict = Field(default_factory=dict)


def create_app(token: str | None = None) -> FastAPI:
    app = FastAPI(title="usvc-serve", version=_version())
    # CORS 最外层：浏览器模式需要跨域；本地 token 兜底防其他网页盗连 127.0.0.1
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if token:

        @app.middleware("http")
        async def _auth(request: Request, call_next):
            if request.method == "OPTIONS" or request.url.path == "/api/health":
                return await call_next(request)
            if request.headers.get("x-usvc-token") != token:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            return await call_next(request)

    scheduler = JobScheduler()
    app.state.scheduler = scheduler

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "service": "usvc-serve", "version": _version()}

    @app.get("/api/env")
    async def env() -> dict:
        """环境诊断（复用 usvc-checkenv 的 full_report）。"""
        report = _envcheck.full_report()
        return {"report": report, "verdict": _envcheck._verdict(report)}

    @app.get("/api/backends")
    async def backends() -> dict:
        return {"backends": [{"name": n} for n in BACKENDS]}

    @app.get("/api/presets")
    async def presets() -> dict:
        """预置训练配置（configs/*.yaml）摘要。"""
        base = _presets_dir()
        if base is None:
            return {
                "dir": None,
                "presets": [],
                "note": "configs/ 目录未找到（从仓库根启动或设置 USVC_CONFIGS_DIR）",
            }
        out = []
        for p in sorted(base.glob("*.yaml")):
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                data = {}
            out.append(
                {
                    "name": p.stem,
                    "path": str(p),
                    "backend": data.get("backend"),
                    "dataset_dir": data.get("dataset_dir"),
                    "exp_dir": data.get("exp_dir"),
                    "batch_size": data.get("batch_size"),
                    "max_steps": data.get("max_steps"),
                }
            )
        return {"dir": str(base), "presets": out}

    @app.get("/api/jobs")
    async def jobs_list(state: str | None = None) -> dict:
        jobs = [j.snapshot() for j in scheduler.list()]
        if state:
            jobs = [j for j in jobs if j["state"] == state]
        return {"jobs": jobs}

    @app.post("/api/jobs", status_code=201)
    async def jobs_create(body: JobCreate) -> dict:
        try:
            job = await scheduler.submit(body.kind, body.params)
        except JobError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return job.snapshot()

    @app.get("/api/jobs/{job_id}")
    async def jobs_get(job_id: str) -> dict:
        job = scheduler.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.snapshot()

    @app.delete("/api/jobs/{job_id}")
    async def jobs_cancel(job_id: str) -> dict:
        job = await scheduler.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def jobs_events(job_id: str, from_seq: int = Query(0, ge=0)) -> StreamingResponse:
        """SSE 事件流：replay-then-tail（先回放 from_seq 之后的历史，再续实时流）。"""
        job = scheduler.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def gen():
            # 先订阅再回放：窗口内重复投递的事件按 seq 去重，终态不丢
            q: asyncio.Queue = asyncio.Queue(maxsize=4096)
            job.subscribe(q)
            try:
                last = from_seq
                for ev in job.events_after(last):
                    yield _sse(ev)
                    last = ev.seq
                if job.state in TERMINAL_STATES:
                    return
                while True:
                    ev = await q.get()
                    if ev is None:  # 终态哨兵
                        return
                    if ev.seq <= last:
                        continue  # 订阅窗口内已回放的事件
                    last = ev.seq
                    yield _sse(ev)
            finally:
                job.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="usvc-serve", description="usvc WebUI/桌面端本地 API 服务")
    p.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口；0=自动选空闲端口")
    p.add_argument("--token", default=None, help="鉴权 token（默认取 USVC_SERVE_TOKEN 环境变量）")
    p.add_argument("--no-token", action="store_true", help="强制关闭鉴权")
    args = p.parse_args(argv)

    token = None if args.no_token else (args.token or os.environ.get("USVC_SERVE_TOKEN"))

    # 先绑定 socket 拿实际端口（--port 0 时），再交给 uvicorn 复用，避免端口竞争
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    actual_port = sock.getsockname()[1]

    # 启动契约：stdout 第一行，桌面端/客户端解析 {port, token}
    print(
        f"USVC_READY {json.dumps({'port': actual_port, 'token': token}, ensure_ascii=False)}",
        flush=True,
    )

    app = create_app(token=token)
    config = uvicorn.Config(app, host=args.host, port=actual_port, log_level="info")
    server = uvicorn.Server(config)
    server.config.sockets = [sock]
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
