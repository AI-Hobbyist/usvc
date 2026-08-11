"""统一长任务抽象（Job）与调度器。

设计（docs/WEBUI_DESIGN.md §3）：
- 所有长操作（预处理/训练/建索引/导出/下载/推理）统一为 Job。
- 事件流带单调递增 seq，SSE 支持 replay-then-tail（先补发历史事件再续实时流）。
- GPU 作业全局互斥（训练与推理并发会争显存）。
- 取消 = 杀整棵进程树（POSIX 进程组 / Windows taskkill /T）。

P0 只内置验证用 kind（echo/checkenv/sleep），真实作业 kind 在 P1+ 接入。
"""
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

__all__ = ["KINDS", "TERMINAL_STATES", "Job", "JobError", "JobScheduler"]

TERMINAL_STATES = frozenset({"succeeded", "failed", "canceled"})

#: 事件环形缓冲上限（replay 窗口，超过后最早的被丢弃）
BUFFER_SIZE = 2000

#: Job 类型注册表；gpu=True 的作业由调度器全局串行。
KINDS: dict[str, dict] = {
    "echo": {"gpu": False, "desc": "验证用：每秒输出一行日志（20 行后退出）"},
    "checkenv": {"gpu": False, "desc": "运行 usvc-checkenv 环境诊断（流式输出日志）"},
    "sleep": {"gpu": False, "desc": "验证用：挂起 1 小时（用于测试取消）"},
    "infer": {"gpu": True, "desc": "单文件歌声转换（usvc-infer）"},
}

_ECHO_SCRIPT = (
    "import sys, time\n"
    "for i in range(20):\n"
    "    print(f'echo line {i + 1}/20', flush=True)\n"
    "    time.sleep(0.1)\n"
)
_SLEEP_SCRIPT = "import time; time.sleep(3600)"


class JobError(Exception):
    """Job 参数错误（kind 不存在等）。"""


def _argv_for(kind: str, params: dict | None = None) -> list[str]:
    if kind == "echo":
        return [sys.executable, "-c", _ECHO_SCRIPT]
    if kind == "checkenv":
        return [sys.executable, "-m", "usvc._envcheck"]
    if kind == "sleep":
        return [sys.executable, "-c", _SLEEP_SCRIPT]
    if kind == "infer":
        return _infer_argv(params or {})
    raise JobError(f"unknown job kind: {kind!r}")


#: infer 参数 → usvc-infer CLI 值参数映射（None 即省略，CLI 用默认值）
_INFER_VALUE_FLAGS = {
    "vocoder_dir": "--vocoder-dir",
    "spk": "--spk",
    "trans": "--trans",
    "steps": "--steps",
    "resample_sr": "--resample-sr",
    "noice_scale": "--noice-scale",
    "formant_shift": "--formant-shift",
    "vocal_register_shift": "--vocal-register-shift",
    "f0_method": "--f0-method",
    "device": "--device",
    "index_path": "--index-path",
    "index_ratio": "--index-ratio",
    "cluster_path": "--cluster-path",
    "cluster_ratio": "--cluster-ratio",
    "rms_mix_rate": "--rms-mix-rate",
    "protect": "--protect",
    "diffusion_ckpt": "--diffusion-ckpt",
    "diffusion_config": "--diffusion-config",
    "k_step": "--k-step",
    "diffusion_method": "--diffusion-method",
    "diffusion_speedup": "--diffusion-speedup",
    "reflow_ckpt": "--reflow-ckpt",
    "reflow_config": "--reflow-config",
    "t_start": "--t-start",
    "ds_cfg": "--ds-cfg",
    "spk_cfg": "--spk-cfg",
    "skip_cfg": "--skip-cfg",
    "target_loudness": "--target-loudness",
}

#: infer 参数 → usvc-infer CLI 开关参数映射（True 即追加）
_INFER_BOOL_FLAGS = {
    "half": "--half",
    "no_half": "--no-half",
    "auto_predict_f0": "--auto-predict-f0",
    "compile_graph": "--compile-graph",
    "shallow_diffusion": "--shd",
    "no_restore_loudness": "--no-restore-loudness",
}


def _infer_argv(params: dict) -> list[str]:
    input_path = params.get("input")
    config = params.get("config")
    ckpt = params.get("ckpt")
    if not input_path or not config or not ckpt:
        raise JobError("infer job requires params: input, config, ckpt")
    argv = [
        sys.executable, "-m", "usvc.cli.infer", str(input_path),
        "--config", str(config), "--ckpt", str(ckpt),
    ]
    if params.get("output"):
        argv += ["--output", str(params["output"])]
    for key, flag in _INFER_VALUE_FLAGS.items():
        v = params.get(key)
        if v is not None:
            argv += [flag, str(v)]
    for key, flag in _INFER_BOOL_FLAGS.items():
        if params.get(key):
            argv.append(flag)
    return argv


@dataclass
class JobEvent:
    seq: int
    ts: float
    type: str  # log | state | progress | artifact | scalar
    payload: dict

    def to_dict(self) -> dict:
        return {"seq": self.seq, "ts": self.ts, "type": self.type, "payload": self.payload}


class Job:
    """单个长任务：事件缓冲（replay 用）+ 订阅者（SSE 用）+ 生命周期。"""

    def __init__(self, kind: str, params: dict | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.params = params or {}
        self.gpu = bool(KINDS[kind]["gpu"])
        self.state = "pending"
        self.exit_code: int | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._seq = 0
        self._buffer: deque[JobEvent] = deque(maxlen=BUFFER_SIZE)
        self._subs: set[asyncio.Queue] = set()
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel_requested = False

    # ---- 事件 ----
    def emit(self, type_: str, **payload: object) -> JobEvent:
        """记录事件并推送给所有 SSE 订阅者（缓冲与实时投递同时进行）。"""
        self._seq += 1
        ev = JobEvent(seq=self._seq, ts=time.time(), type=type_, payload=payload)
        self._buffer.append(ev)
        for q in list(self._subs):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass  # 订阅者消费慢则丢弃实时投递（缓冲里仍有完整副本）
        return ev

    def events_after(self, seq: int) -> list[JobEvent]:
        return [e for e in self._buffer if e.seq > seq]

    def subscribe(self, q: asyncio.Queue) -> None:
        self._subs.add(q)

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    # ---- 生命周期 ----
    def mark_running(self) -> None:
        if self.state != "pending":
            return
        self.state = "running"
        self.started_at = time.time()
        self.emit("state", state="running")

    def mark_terminal(
        self, state: str, exit_code: int | None = None, error: str | None = None
    ) -> None:
        if self.state in TERMINAL_STATES:
            return
        self.state = state
        self.exit_code = exit_code
        self.error = error
        self.finished_at = time.time()
        self.emit("state", state=state, exit_code=exit_code, error=error)
        for q in list(self._subs):
            try:
                q.put_nowait(None)  # 终态哨兵：通知 SSE 流关闭
            except asyncio.QueueFull:
                pass

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "gpu": self.gpu,
            "state": self.state,
            "exit_code": self.exit_code,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobScheduler:
    """Job 注册、执行（子进程 + 流式日志）、GPU 互斥与取消。"""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._gpu_lock = asyncio.Lock()
        self._runners: set[asyncio.Task] = set()
        #: 成功收尾回调（如注册音频产物、发 artifact 事件），由 main.py 注入
        self.on_success = None

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def submit(self, kind: str, params: dict | None = None) -> Job:
        if kind not in KINDS:
            raise JobError(f"unknown job kind: {kind!r}; available: {sorted(KINDS)}")
        if kind == "infer":
            missing = [k for k in ("input", "config", "ckpt") if not (params or {}).get(k)]
            if missing:
                raise JobError(f"infer job requires params: {', '.join(missing)}")
        job = Job(kind, params)
        self._jobs[job.id] = job
        task = asyncio.create_task(self._run(job))
        self._runners.add(task)
        task.add_done_callback(self._runners.discard)
        return job

    async def _run(self, job: Job) -> None:
        job.mark_running()
        try:
            if job.gpu:
                await self._gpu_lock.acquire()
            try:
                if job.state in TERMINAL_STATES or job._cancel_requested:
                    job.mark_terminal("canceled")  # 排队期间被取消
                    return
                await self._execute(job)
            finally:
                if job.gpu:
                    self._gpu_lock.release()
        except Exception as exc:  # noqa: BLE001  # 兜底：任何未捕获异常都落成 failed
            job.mark_terminal("failed", error=f"{type(exc).__name__}: {exc}")

    async def _execute(self, job: Job) -> None:
        argv = _argv_for(job.kind, job.params)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),  # POSIX: 独立进程组，便于杀整树
        )
        job._proc = proc
        job.emit("log", stream="system", line=f"[job] pid={proc.pid} argv={' '.join(argv)}")

        if job._cancel_requested:
            # 取消在「mark_running 之后、进程创建完成之前」到达时会走到这里
            # （cancel() 看到 _proc 为 None 无法杀）——立刻终止，下面统一收尾。
            await _kill_tree(proc)

        async def pump(stream: asyncio.StreamReader, name: str) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    return
                job.emit("log", stream=name, line=line.decode("utf-8", "replace").rstrip("\n"))

        await asyncio.gather(pump(proc.stdout, "stdout"), pump(proc.stderr, "stderr"))
        code = await proc.wait()
        job.exit_code = code
        if job._cancel_requested:
            job.mark_terminal("canceled", exit_code=code)
        elif code == 0:
            # 先回调（artifact 事件要先于终态发出，否则 SSE 订阅者会错过），再落终态
            if self.on_success is not None:
                try:
                    self.on_success(job)
                except Exception as exc:  # noqa: BLE001  # 回调失败不阻塞任务收尾
                    job.emit("log", stream="system", line=f"[job] on_success error: {exc!r}")
            job.mark_terminal("succeeded", exit_code=code)
        else:
            job.mark_terminal("failed", exit_code=code, error=f"exit code {code}")

    async def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None or job.state in TERMINAL_STATES:
            return job
        job._cancel_requested = True
        if job.state == "pending":
            job.mark_terminal("canceled")  # 排队中：直接终止，runner 启动时会跳过执行
            return job
        proc = job._proc
        if proc is not None and proc.returncode is None:
            await _kill_tree(proc)
        return job


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """杀整棵进程树：POSIX 用进程组 SIGTERM→SIGKILL，Windows 用 taskkill /T。"""
    if os.name == "nt":
        await asyncio.to_thread(
            subprocess.run,
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
