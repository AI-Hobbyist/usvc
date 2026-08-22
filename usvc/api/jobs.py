"""Unified Job abstraction + scheduler (WEBUI_DESIGN §3): all long ops are Jobs with a monotonic
event seq (SSE replay-then-tail), global GPU mutex, and cancel = kill the whole process tree
(POSIX process group / Windows taskkill /T)."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass

TERMINAL_STATES = frozenset({"succeeded", "failed", "canceled"})

BUFFER_SIZE = 2000  # event replay window; oldest dropped beyond this

# gpu=True jobs serialize through one global lock (train+infer would fight for VRAM)
KINDS: dict[str, dict] = {
    "echo": {"gpu": False, "desc": "验证用：每秒输出一行日志（20 行后退出）"},
    "checkenv": {"gpu": False, "desc": "运行 usvc-checkenv 环境诊断（流式输出日志）"},
    "sleep": {"gpu": False, "desc": "验证用：挂起 1 小时（用于测试取消）"},
    "infer": {"gpu": True, "desc": "单文件歌声转换（usvc-infer）"},
    "train": {"gpu": True, "desc": "训练（usvc-train）"},
}

_ECHO_SCRIPT = (
    "import sys, time\n"
    "for i in range(20):\n"
    "    print(f'echo line {i + 1}/20', flush=True)\n"
    "    time.sleep(0.1)\n"
)
_SLEEP_SCRIPT = "import time; time.sleep(3600)"


class JobError(Exception):
    """Job parameter error (unknown kind, missing params, ...)."""


#: infer param -> usvc-infer CLI value flag (None = omit, CLI default applies)
_INFER_VALUE_FLAGS = {
    "vocoder_dir": "--vocoder-dir", "spk": "--spk", "trans": "--trans", "steps": "--steps",
    "resample_sr": "--resample-sr", "noice_scale": "--noice-scale",
    "formant_shift": "--formant-shift", "vocal_register_shift": "--vocal-register-shift",
    "f0_method": "--f0-method", "device": "--device", "index_path": "--index-path",
    "index_ratio": "--index-ratio", "cluster_path": "--cluster-path",
    "cluster_ratio": "--cluster-ratio", "rms_mix_rate": "--rms-mix-rate", "protect": "--protect",
    "diffusion_ckpt": "--diffusion-ckpt", "diffusion_config": "--diffusion-config",
    "k_step": "--k-step", "diffusion_method": "--diffusion-method",
    "diffusion_speedup": "--diffusion-speedup", "reflow_ckpt": "--reflow-ckpt",
    "reflow_config": "--reflow-config", "t_start": "--t-start", "ds_cfg": "--ds-cfg",
    "spk_cfg": "--spk-cfg", "skip_cfg": "--skip-cfg", "target_loudness": "--target-loudness",
}
#: infer param -> usvc-infer CLI switch (True = append)
_INFER_BOOL_FLAGS = {
    "half": "--half", "no_half": "--no-half", "auto_predict_f0": "--auto-predict-f0",
    "compile_graph": "--compile-graph", "shallow_diffusion": "--shd",
    "no_restore_loudness": "--no-restore-loudness",
}


def _train_argv(params):
    if not params.get("config"): raise JobError("train job requires params: config")
    argv = [sys.executable, "-m", "usvc.cli.train", "--config", str(params["config"])]
    if params.get("resume"): argv += ["--resume", str(params["resume"])]
    if params.get("device"): argv += ["--device", str(params["device"])]
    return argv


def _infer_argv(params):
    missing = [k for k in ("input", "config", "ckpt") if not params.get(k)]
    if missing: raise JobError(f"infer job requires params: {', '.join(missing)}")
    argv = [sys.executable, "-m", "usvc.cli.infer", str(params["input"]),
            "--config", str(params["config"]), "--ckpt", str(params["ckpt"])]
    if params.get("output"): argv += ["--output", str(params["output"])]
    for key, flag in _INFER_VALUE_FLAGS.items():
        if params.get(key) is not None: argv += [flag, str(params[key])]
    for key, flag in _INFER_BOOL_FLAGS.items():
        if params.get(key): argv.append(flag)
    return argv


def _argv_for(kind, params=None) -> list:
    p = params or {}
    if kind == "echo": return [sys.executable, "-c", _ECHO_SCRIPT]
    if kind == "checkenv": return [sys.executable, "-m", "usvc._envcheck"]
    if kind == "sleep": return [sys.executable, "-c", _SLEEP_SCRIPT]
    if kind == "infer": return _infer_argv(p)
    if kind == "train": return _train_argv(p)
    raise JobError(f"unknown job kind: {kind!r}")


@dataclass
class JobEvent:
    seq: int
    ts: float
    type: str  # log | state | progress | artifact | scalar
    payload: dict
    def to_dict(self) -> dict:
        return {"seq": self.seq, "ts": self.ts, "type": self.type, "payload": self.payload}


class Job:  # event buffer (replay) + subscribers (SSE) + lifecycle
    def __init__(self, kind, params=None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.params = params or {}
        self.gpu = bool(KINDS[kind]["gpu"])
        self.state = "pending"; self.exit_code: int | None = None; self.error: str | None = None
        self.created_at = time.time(); self.started_at: float | None = None; self.finished_at: float | None = None
        self._seq = 0
        self._buffer: deque[JobEvent] = deque(maxlen=BUFFER_SIZE)
        self._subs: set[asyncio.Queue] = set()
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel_requested = False

    # ---- events ----
    def emit(self, type_, **payload) -> JobEvent:  # buffer + fan out to SSE subscribers
        self._seq += 1
        ev = JobEvent(seq=self._seq, ts=time.time(), type=type_, payload=payload)
        self._buffer.append(ev)
        for q in list(self._subs):
            try: q.put_nowait(ev)
            except asyncio.QueueFull: pass  # slow subscriber drops live copy (buffer keeps full)
        return ev
    def events_after(self, seq) -> list: return [e for e in self._buffer if e.seq > seq]
    def subscribe(self, q) -> None: self._subs.add(q)
    def unsubscribe(self, q) -> None: self._subs.discard(q)

    # ---- lifecycle ----
    def mark_running(self) -> None:
        if self.state != "pending": return
        self.state = "running"; self.started_at = time.time()
        self.emit("state", state="running")
    def mark_terminal(self, state, exit_code=None, error=None) -> None:
        if self.state in TERMINAL_STATES: return
        self.state = state; self.exit_code = exit_code; self.error = error
        self.finished_at = time.time()
        self.emit("state", state=state, exit_code=exit_code, error=error)
        for q in list(self._subs):
            try: q.put_nowait(None)  # terminal sentinel: SSE stream closes
            except asyncio.QueueFull: pass
    def snapshot(self) -> dict:
        return {"id": self.id, "kind": self.kind, "params": self.params, "gpu": self.gpu,
                "state": self.state, "exit_code": self.exit_code, "error": self.error,
                "created_at": self.created_at, "started_at": self.started_at,
                "finished_at": self.finished_at}


class JobScheduler:  # registration, subprocess exec with streamed logs, GPU mutex, cancel
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._gpu_lock = asyncio.Lock()
        self._runners: set[asyncio.Task] = set()
        self.on_success = None  # injected by main.py (artifact registration etc.)
    def list(self) -> list: return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
    def get(self, job_id) -> Job | None: return self._jobs.get(job_id)
    async def submit(self, kind, params=None) -> Job:
        if kind not in KINDS: raise JobError(f"unknown job kind: {kind!r}; available: {sorted(KINDS)}")
        if kind == "infer":
            missing = [k for k in ("input", "config", "ckpt") if not (params or {}).get(k)]
            if missing: raise JobError(f"infer job requires params: {', '.join(missing)}")
        if kind == "train" and not (params or {}).get("config"):
            raise JobError("train job requires params: config")
        job = Job(kind, params)
        self._jobs[job.id] = job
        task = asyncio.create_task(self._run(job))
        self._runners.add(task)
        task.add_done_callback(self._runners.discard)
        return job
    async def _run(self, job) -> None:
        job.mark_running()
        try:
            if job.gpu: await self._gpu_lock.acquire()
            try:
                if job.state in TERMINAL_STATES or job._cancel_requested:
                    job.mark_terminal("canceled")  # canceled while queued
                    return
                await self._execute(job)
            finally:
                if job.gpu: self._gpu_lock.release()
        except Exception as exc:  # catch-all: anything uncaught lands as failed
            job.mark_terminal("failed", error=f"{type(exc).__name__}: {exc}")
    async def _execute(self, job) -> None:
        argv = _argv_for(job.kind, job.params)
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=(os.name != "nt"))  # POSIX: own process group -> kill whole tree
        job._proc = proc
        job.emit("log", stream="system", line=f"[job] pid={proc.pid} argv={' '.join(argv)}")

        if job._cancel_requested:  # cancel raced between mark_running and process spawn
            await _kill_tree(proc)
        async def pump(stream, name):
            while True:
                line = await stream.readline()
                if not line: return
                job.emit("log", stream=name, line=line.decode("utf-8", "replace").rstrip("\n"))

        await asyncio.gather(pump(proc.stdout, "stdout"), pump(proc.stderr, "stderr"))
        code = await proc.wait()
        job.exit_code = code
        if job._cancel_requested:
            job.mark_terminal("canceled", exit_code=code)
        elif code == 0:
            # callback first: artifact events must precede terminal or SSE subs miss them
            if self.on_success is not None:
                try: self.on_success(job)
                except Exception as exc:  # callback failure must not block finalization
                    job.emit("log", stream="system", line=f"[job] on_success error: {exc!r}")
            job.mark_terminal("succeeded", exit_code=code)
        else:
            job.mark_terminal("failed", exit_code=code, error=f"exit code {code}")
    async def cancel(self, job_id) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None or job.state in TERMINAL_STATES: return job
        job._cancel_requested = True
        if job.state == "pending":
            job.mark_terminal("canceled")  # runner skips execution on startup
            return job
        if job._proc is not None and job._proc.returncode is None: await _kill_tree(job._proc)
        return job


async def _kill_tree(proc) -> None:  # POSIX pgid TERM->KILL; Windows taskkill /T /F
    if os.name == "nt":
        await asyncio.to_thread(subprocess.run, ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                capture_output=True, check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        return
    try: pgid = os.getpgid(proc.pid)
    except ProcessLookupError: return
    try: os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError: return
    try: await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        try: os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError: pass
