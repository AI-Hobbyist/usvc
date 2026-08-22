"""usvc-profile CLI: nsys capture + parse + HTML report (plotly 单文件) + report.json +
<name>.nsys-rep/.sqlite。nsys 包裹 python 推理子进程：USVC_PROFILE_MODE 区分调度/运行模式。

Usage: usvc-profile infer -c configs/rvc.yaml --ckpt m.ckpt [-o docs/prof/infer]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

def main(argv=None) -> int:
    if os.environ.get("USVC_PROFILE_MODE") == "run": return _run_inference()  # wrapped by nsys

    p = argparse.ArgumentParser(prog="usvc-profile",
                                description="Profile usvc inference/train with nsys, render HTML report.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("infer", help="profile inference")
    pi.add_argument("--config", "-c", required=True, help="training config yaml")
    pi.add_argument("--ckpt", required=True, help="model checkpoint")
    pi.add_argument("--input", default="/tmp/input_600s.wav", help="input wav (default: 600s standard clip)")
    pi.add_argument("--output", "-o", default="docs/prof/infer", help="output dir"); pi.add_argument("--name", default=None, help="run name (default: <backend>_infer)")
    pi.add_argument("--spk", type=int, default=0); pi.add_argument("--trans", type=int, default=0)
    pi.add_argument("--vocoder-dir", default=None)
    pi.add_argument("--extra-arg", action="append", default=[],
                    help="extra usvc-infer args (e.g. --extra-arg=--half)")
    pi.add_argument("--half", action="store_true", help="FP16 autocast (rvc 默认开)")
    pi.add_argument("--bf16", action="store_true", help="BF16 autocast"); pi.add_argument("--no-half", action="store_true", help="强制 FP32 (覆盖 rvc 默认)")
    args = p.parse_args(argv)
    return _profile_infer(args) if args.cmd == "infer" else 1

def _profile_infer(args) -> int:
    nsys = _find_nsys()
    if nsys is None:
        print("ERROR: nsys 未找到。装: apt install nsight-systems-2024.2.3", file=sys.stderr); return 2

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    name = args.name or _backend_from_config(args.config) + "_infer"
    prefix = str(outdir / name)

    infer_args = ["--config", args.config, "--ckpt", args.ckpt, "--input", args.input,
                  "--spk", str(args.spk), "--trans", str(args.trans)]
    # precision: --no-half > --bf16 > --half > backend default (rvc half)
    if args.no_half: infer_args += ["--precision", "fp32"]
    elif args.bf16: infer_args += ["--precision", "bf16"]
    elif args.half: infer_args += ["--precision", "fp16"]
    if args.vocoder_dir: infer_args += ["--vocoder-dir", args.vocoder_dir]
    infer_args += args.extra_arg

    env = dict(os.environ)
    env["USVC_PROFILE_MODE"] = "run"
    env["USVC_PROFILE_ARGS"] = json.dumps(infer_args)
    shape_path = outdir / f"{name}.shapes.json"
    env["USVC_PROFILE_SHAPES"] = str(shape_path)

    cmd = [nsys, "profile", "--trace=cuda,nvtx", "--force-overwrite=true",
           "--export=sqlite", "-o", prefix, sys.executable, "-m", "usvc.cli.profile"]
    # 不用 capture_output：管道会干扰 nsys 采集（实测 kernel 表变空），输出重定向到日志文件
    log_path = outdir / f"{name}.nsys.log"
    print(f"[profile] 启动 nsys → {prefix} (日志 {log_path})")
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"ERROR: nsys 失败 (rc={r.returncode}), 见 {log_path}", file=sys.stderr); return r.returncode

    sqlite_path = prefix + ".sqlite"
    if not Path(sqlite_path).exists():
        print(f"ERROR: nsys 未产出 {sqlite_path}, 见 {log_path}", file=sys.stderr); return 3

    from usvc._profiler.nsys_parser import parse_sqlite
    from usvc._profiler.render import render_html

    print(f"[profile] 解析 {sqlite_path}...")
    rep = parse_sqlite(sqlite_path, nsys_rep_path=prefix + ".nsys-rep", shapes_path=shape_path)

    html_path, json_path = outdir / "report.html", outdir / "report.json"
    print(f"[profile] 渲染 {html_path}...")
    render_html(rep, html_path, title=f"{name} profile", input_duration_s=_wav_duration(args.input))
    json_path.write_text(json.dumps(_report_to_dict(rep), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✓ 报告: {html_path} (浏览器打开)\n✓ 数据: {json_path}")
    print(f"✓ 深挖: {prefix}.nsys-rep (Nsight Systems UI 打开)")
    _print_summary(rep)
    return 0

def _run_inference() -> int:
    """被 nsys 包裹运行: warmup 1 次 + profile 1 次推理（ShapeCapture 记录 conv/gemm shapes）。"""
    import torch
    from usvc.infer.engine import InferenceEngine

    args = json.loads(os.environ["USVC_PROFILE_ARGS"])
    at = lambda flag: args[args.index(flag) + 1]
    kw = {"vocoder_dir": at("--vocoder-dir")} if "--vocoder-dir" in args else {}
    eng = InferenceEngine(at("--config"), at("--ckpt"), device="cuda", **kw)
    # 精度: --precision fp32/bf16/fp16 (默认 rvc 自动 half)
    precision = at("--precision") if "--precision" in args else "auto"
    eng.half, eng._autocast_dtype = {
        "fp32": (False, None), "fp16": (True, torch.float16), "bf16": (True, torch.bfloat16),
    }.get(precision, (eng.half, eng._autocast_dtype))
    inp, spk, trans = at("--input"), int(at("--spk")), int(at("--trans"))
    _ = eng.convert(inp, spk=spk, trans=trans)  # warmup
    from usvc._profiler.shape_capture import ShapeCapture
    cap = ShapeCapture()
    with cap: out = eng.convert(inp, spk=spk, trans=trans)  # profile
    shape_out = os.environ.get("USVC_PROFILE_SHAPES")
    if shape_out:
        ops_data = [{"op": o.op, "shapes": [list(s) for s in o.shapes], "count": o.count,
                     "flops_per_call": o.flops_per_call}
                    for o in cap.compute_ops if o.flops_per_call]
        Path(shape_out).write_text(json.dumps(ops_data, indent=2), encoding="utf-8")
        print(f"[run] 记录 {len(ops_data)} 个 conv/gemm shape")
    import numpy as np
    print(f"[run] 输出 {len(out)} samples, max {np.abs(out).max():.3f}")
    return 0

def _find_nsys() -> str | None:
    for pat in ("/opt/nvidia/nsight-systems/2024.2.3/bin/nsys", "/opt/nvidia/nsight-systems/*/bin/nsys"):
        for hit in glob.glob(pat):
            if Path(hit).exists(): return hit
    from shutil import which
    return which("nsys")

def _backend_from_config(cfg_path) -> str:
    import yaml
    with open(cfg_path) as f: return str(yaml.safe_load(f).get("backend", "unknown"))

def _wav_duration(path) -> float:
    try:
        import soundfile as sf
        w, sr = sf.read(path); return len(w) / sr
    except Exception:
        return 0.0

def _report_to_dict(rep) -> dict:
    hw = {"name": rep.hardware.name, "sm_count": rep.hardware.sm_count,
          "memory_bandwidth_gbs": rep.hardware.memory_bandwidth_gbs}
    return {"hardware": hw, "total_gpu_ms": rep.total_gpu_ns / 1e6,
            "kernels": [{"name": k.name[:100], "ms": k.total_ns / 1e6, "count": k.count,
                         "regs": k.registers_per_thread} for k in rep.kernels[:30]],
            "memcpy": {"count": rep.memcpy.count, "bytes": rep.memcpy.total_bytes,
                       "ms": rep.memcpy.total_ns / 1e6},
            "stages": [{"name": s.name, "instances": s.instances, "total_ms": s.total_ms,
                        "top_kernels": [{"name": k.name[:80], "ms": k.total_ns / 1e6, "count": k.count}
                                        for k in s.kernels[:5]]} for s in (rep.stages or [])]}

def _print_summary(rep) -> None:
    from usvc._profiler._kernel_flops import CUDNN_CUBLAS, ELEMENTWISE_REDUCE, classify_kernel
    from usvc._profiler.occupancy import theoretical_occupancy

    total = rep.total_kernel_ns / 1e6
    glue = sum(k.total_ns for k in rep.kernels
               if classify_kernel(k.name) in (ELEMENTWISE_REDUCE, "other")) / 1e6
    conv = sum(k.total_ns for k in rep.kernels if classify_kernel(k.name) == CUDNN_CUBLAS) / 1e6
    memcpy_ms = rep.memcpy.total_ns / 1e6
    print("\n--- 关键结论 ---")
    print(f"GPU 总 {total:.0f}ms | conv/gemm {conv:.0f}ms ({100 * conv / total:.0f}%) | "
          f"胶水 {glue:.0f}ms ({100 * glue / total:.0f}%) | "
          f"memcpy {memcpy_ms:.0f}ms ({100 * memcpy_ms / total:.0f}%)")
    occ_vals = [(k.total_ns / 1e6, theoretical_occupancy(int(k.registers_per_thread), k.block, (8, 6)))
                for k in rep.kernels if k.registers_per_thread > 0]
    if occ_vals:
        wocc = sum(m * o for m, o in occ_vals if o >= 0) / sum(m for m, o in occ_vals if o >= 0)
        print(f"加权 occupancy: {wocc:.0f}% (理论)")
    if rep.stages:  # NVTX 阶段拆解
        st_total = sum(s.total_ms for s in rep.stages)
        print("阶段拆解 (NVTX):")
        for s in rep.stages:
            print(f"  {s.name:12s} {s.total_ms:7.0f}ms ({100 * s.total_ms / st_total:2.0f}%) x{s.instances}")

if __name__ == "__main__":
    sys.exit(main())
