"""usvc-profile CLI: nsys 采集 + 解析 + 渲染 HTML 报告。

Usage:
    usvc-profile infer --backend rvc --config configs/rvc.yaml --ckpt model.ckpt \
        --input /tmp/input_600s.wav -o docs/prof/rvc_infer/

产出:
    <output>/report.html         可视化(plotly 单文件)
    <output>/report.json         结构化(agent 读)
    <output>/<name>.nsys-rep     Perfetto/Nsight UI 深挖
    <output>/<name>.sqlite       原始数据

实现: nsys 需包裹 python 推理子进程, 本脚本通过 USVC_PROFILE_MODE 环境变量
区分"调度模式"(启动 nsys)和"运行模式"(被 nsys 包裹做推理)。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # 运行模式: 被 nsys 包裹, 执行真实推理
    if os.environ.get("USVC_PROFILE_MODE") == "run":
        return _run_inference()

    p = argparse.ArgumentParser(
        prog="usvc-profile",
        description="Profile usvc inference/train with nsys, render HTML report.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("infer", help="profile inference")
    pi.add_argument("--config", "-c", required=True, help="training config yaml")
    pi.add_argument("--ckpt", required=True, help="model checkpoint")
    pi.add_argument("--input", default="/tmp/input_600s.wav",
                    help="input wav (default: 600s standard clip)")
    pi.add_argument("--output", "-o", default="docs/prof/infer", help="output dir")
    pi.add_argument("--name", default=None, help="run name (default: <backend>_infer)")
    pi.add_argument("--spk", type=int, default=0)
    pi.add_argument("--trans", type=int, default=0)
    pi.add_argument("--vocoder-dir", default=None)
    pi.add_argument("--extra-arg", action="append", default=[],
                    help="extra args passed to usvc-infer (e.g. --extra-arg=--half)")

    args = p.parse_args(argv)

    if args.cmd == "infer":
        return _profile_infer(args)
    return 1


def _profile_infer(args) -> int:
    nsys = _find_nsys()
    if nsys is None:
        print("ERROR: nsys 未找到。装: apt install nsight-systems-2024.2.3", file=sys.stderr)
        return 2

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    name = args.name or _backend_from_config(args.config) + "_infer"
    prefix = str(outdir / name)

    # 构造推理命令(用本脚本的 run 模式, 复用 usvc-infer 的引擎)
    infer_args = ["--config", args.config, "--ckpt", args.ckpt,
                  "--input", args.input, "--spk", str(args.spk),
                  "--trans", str(args.trans)]
    if args.vocoder_dir:
        infer_args += ["--vocoder-dir", args.vocoder_dir]
    infer_args += args.extra_arg

    env = dict(os.environ)
    env["USVC_PROFILE_MODE"] = "run"
    env["USVC_PROFILE_ARGS"] = json.dumps(infer_args)

    cmd = [nsys, "profile",
           "--trace=cuda,nvtx",
           "--force-overwrite=true",
           "--export=sqlite",
           "-o", prefix,
           sys.executable, "-m", "usvc.cli.profile"]
    # 重要: 不用 capture_output。nsys 的数据采集会被 stdout/stderr 管道干扰
    # (实测 capture_output=True 导致 kernel 表为空)。输出重定向到日志文件。
    log_path = outdir / f"{name}.nsys.log"
    print(f"[profile] 启动 nsys → {prefix} (日志 {log_path})")
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"ERROR: nsys 失败 (rc={r.returncode}), 见 {log_path}", file=sys.stderr)
        return r.returncode

    # 解析 + 渲染
    sqlite_path = prefix + ".sqlite"
    if not Path(sqlite_path).exists():
        print(f"ERROR: nsys 未产出 {sqlite_path}, 见 {log_path}", file=sys.stderr)
        return 3

    from usvc._profiler.nsys_parser import parse_sqlite
    from usvc._profiler.render import render_html

    print(f"[profile] 解析 {sqlite_path}...")
    rep = parse_sqlite(sqlite_path)
    rep_dict = _report_to_dict(rep)

    html_path = outdir / "report.html"
    json_path = outdir / "report.json"
    print(f"[profile] 渲染 {html_path}...")
    render_html(rep, html_path, title=f"{name} profile",
                input_duration_s=_wav_duration(args.input))
    json_path.write_text(json.dumps(rep_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✓ 报告: {html_path} (浏览器打开)")
    print(f"✓ 数据: {json_path}")
    print(f"✓ 深挖: {prefix}.nsys-rep (Nsight Systems UI 打开)")
    # 打印关键结论
    _print_summary(rep)
    return 0


def _run_inference() -> int:
    """被 nsys 包裹运行: warmup 1 次 + profile 1 次推理。"""
    args = json.loads(os.environ["USVC_PROFILE_ARGS"])
    from usvc.infer.engine import InferenceEngine
    cfg, ckpt = args[args.index("--config") + 1], args[args.index("--ckpt") + 1]
    inp = args[args.index("--input") + 1]
    spk = int(args[args.index("--spk") + 1])
    trans = int(args[args.index("--trans") + 1])
    kw = {}
    if "--vocoder-dir" in args:
        kw["vocoder_dir"] = args[args.index("--vocoder-dir") + 1]
    eng = InferenceEngine(cfg, ckpt, device="cuda", **kw)
    _ = eng.convert(inp, spk=spk, trans=trans)  # warmup
    out = eng.convert(inp, spk=spk, trans=trans)  # profile
    import numpy as np
    print(f"[run] 输出 {len(out)} samples, max {np.abs(out).max():.3f}")
    return 0


def _find_nsys() -> str | None:
    for c in ("/opt/nvidia/nsight-systems/2024.2.3/bin/nsys",
              "/opt/nvidia/nsight-systems/*/bin/nsys"):
        import glob
        for p in glob.glob(c):
            if Path(p).exists():
                return p
    from shutil import which
    return which("nsys")


def _backend_from_config(cfg_path: str) -> str:
    import yaml
    with open(cfg_path) as f:
        return str(yaml.safe_load(f).get("backend", "unknown"))


def _wav_duration(path: str) -> float:
    try:
        import soundfile as sf
        w, sr = sf.read(path)
        return len(w) / sr
    except Exception:
        return 0.0


def _report_to_dict(rep) -> dict:
    return {
        "hardware": {"name": rep.hardware.name, "sm_count": rep.hardware.sm_count,
                     "memory_bandwidth_gbs": rep.hardware.memory_bandwidth_gbs},
        "total_gpu_ms": rep.total_gpu_ns / 1e6,
        "kernels": [{"name": k.name[:100], "ms": k.total_ns / 1e6,
                     "count": k.count, "regs": k.registers_per_thread}
                    for k in rep.kernels[:30]],
        "memcpy": {"count": rep.memcpy.count, "bytes": rep.memcpy.total_bytes,
                   "ms": rep.memcpy.total_ns / 1e6},
    }


def _print_summary(rep) -> None:
    from usvc._profiler._kernel_flops import classify_kernel, ELEMENTWISE_REDUCE, CUDNN_CUBLAS
    from usvc._profiler.occupancy import theoretical_occupancy
    total = rep.total_kernel_ns / 1e6
    glue = sum(k.total_ns for k in rep.kernels
               if classify_kernel(k.name) in (ELEMENTWISE_REDUCE, "other")) / 1e6
    conv = sum(k.total_ns for k in rep.kernels
               if classify_kernel(k.name) == CUDNN_CUBLAS) / 1e6
    print(f"\n--- 关键结论 ---")
    print(f"GPU 总 {total:.0f}ms | conv/gemm {conv:.0f}ms ({100*conv/total:.0f}%) | "
          f"胶水 {glue:.0f}ms ({100*glue/total:.0f}%) | "
          f"memcpy {rep.memcpy.total_ns/1e6:.0f}ms ({100*rep.memcpy.total_ns/1e6/total:.0f}%)")
    # 加权 occupancy
    occ_vals = [(k.total_ns/1e6, theoretical_occupancy(int(k.registers_per_thread), k.block, (8,6)))
                for k in rep.kernels if k.registers_per_thread > 0]
    if occ_vals:
        wocc = sum(m*o for m,o in occ_vals if o>=0) / sum(m for m,o in occ_vals if o>=0)
        print(f"加权 occupancy: {wocc:.0f}% (理论)")


if __name__ == "__main__":
    sys.exit(main())
