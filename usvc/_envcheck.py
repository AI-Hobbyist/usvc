"""环境诊断：检查 torch CUDA / triton / GPU，给出加速可用性判断。

- ``import usvc`` 时调用 :func:`check_once`，triton 缺失时打一次友好提示（不报错）。
- ``usvc-checkenv`` 命令调用 :func:`main`，输出完整诊断表。

设计原则：triton 是**可选加速**，缺失时框架仍可用（torch.compile 回退 eager、
_triton/ kernel 回退 PyTorch 原生），只损失速度不损失正确性。
"""
from __future__ import annotations

import platform
import sys

_warned = False


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def check_once() -> None:
    """import usvc 时调用：triton 缺失则提示一次（跨平台差异化建议）。"""
    global _warned
    if _warned or _has_triton():
        return
    _warned = True
    plat = platform.system()
    if plat == "Windows":
        sys.stderr.write(
            "\n[usvc] 未检测到 triton —— torch.compile 与手写 kernel 将回退到 "
            "PyTorch 原生（速度变慢，功能正常）。\n"
            "       Windows 上需安装 triton-windows（版本须与 torch 对应）：\n"
            "         pip install triton-windows\n"
            "       对应关系: torch 2.10→triton 3.6 / 2.9→3.5 / 2.8→3.4 / "
            "2.7→3.3 / 2.6→3.2 / 2.4-2.5→3.1\n"
            "       或直接: pip install usvc[gui]\n\n"
        )
    elif plat == "Linux":
        sys.stderr.write(
            "\n[usvc] 未检测到 triton —— torch.compile 与手写 kernel 将回退到 "
            "PyTorch 原生（速度变慢，功能正常）。\n"
            "       Linux 安装: pip install triton\n"
            "       （若用 CPU 版 torch 则不会自动带 triton；建议装 CUDA 版 torch）\n\n"
        )


def full_report() -> dict:
    """返回完整诊断 dict（供 main 和程序化调用）。"""
    import torch
    info: dict = {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.cuda.is_available(),
        "triton": _has_triton(),
    }
    if info["torch_cuda"]:
        cap = torch.cuda.get_device_capability()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        # channels_last 需要 sm80+（Ampere/Hopper/Ada/Blackwell）
        info["channels_last_ok"] = cap[0] >= 8
    else:
        info["gpu"] = None
        info["compute_capability"] = None
        info["channels_last_ok"] = False
    # torch.compile 可用性
    info["compile_ok"] = info["triton"] and info["torch_cuda"]
    return info


def _verdict(info: dict) -> list[str]:
    """根据诊断给出加速可用性结论。"""
    lines: list[str] = []
    if not info["torch_cuda"]:
        lines.append("✗ torch CUDA 不可用 —— 无法 GPU 加速（仅 CPU 回退）")
        lines.append("  安装 CUDA 版 torch: https://pytorch.org/get-started/locally/")
        return lines
    if not info["triton"]:
        lines.append("△ triton 缺失 —— torch.compile/手写 kernel 回退（GPU 加速部分生效）")
        if info["platform"] == "Windows":
            lines.append("  修复: pip install usvc[gui]  (或 pip install triton-windows)")
        else:
            lines.append("  修复: pip install triton")
    else:
        lines.append("✓ triton 可用 —— torch.compile + 手写 kernel 全速")
    if not info["channels_last_ok"]:
        lines.append(
            f"△ GPU sm{info['compute_capability']} < 8.0 —— "
            "channels_last(dec_cl/vocoder_cl)降级关闭（TensorCore 仍可用）"
        )
    else:
        lines.append("✓ GPU sm≥8.0 —— channels_last 全速")
    return lines


def main() -> int:
    """usvc-checkenv 入口。"""
    info = full_report()
    print("=" * 56)
    print("usvc 环境诊断")
    print("=" * 56)
    print(f"  平台:          {info['platform']}")
    print(f"  Python:        {info['python']}")
    print(f"  torch:         {info['torch']}")
    print(f"  torch CUDA:    {'✓' if info['torch_cuda'] else '✗'}")
    if info["gpu"]:
        print(f"  GPU:           {info['gpu']}")
        print(f"  计算能力:      sm{info['compute_capability']}")
    print(f"  triton:        {'✓' if info['triton'] else '✗ 缺失'}")
    print(f"  torch.compile: {'✓' if info['compile_ok'] else '✗ 回退 eager'}")
    print(f"  channels_last: {'✓' if info['channels_last_ok'] else '△ 降级'}")
    print("-" * 56)
    print("加速可用性:")
    for line in _verdict(info):
        print(f"  {line}")
    print("=" * 56)
    return 0 if (info["torch_cuda"] and info["triton"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
