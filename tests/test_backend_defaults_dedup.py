"""E6/E7 anchors: from_ckpt strip helper + shared backend defaults tables.

- strip_prefix: mechanical strip = old inline dict-comprehensions (prefix keys
  dropped, rest deleted; the "keep-bare-keys" variant is caller-side only)
- _defaults.py: torch & mlx modules alias the SAME objects (identity asserts);
  behavioral spot-checks for resolve_ddsp6_opts + RVC_UPSAMPLE invariants.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

import pytest


# ---------------------------------------------------------------------------
# E6: strip_prefix (from_ckpt 骨架收敛)
# ---------------------------------------------------------------------------

def test_strip_prefix_mechanics():
    from usvc.nn.mlx._convert import strip_prefix

    sd = {"model.a": 1, "model.b": 2, "c": 3}
    assert strip_prefix(sd, "model.") == {"a": 1, "b": 2}
    assert strip_prefix(sd, "model.") != sd  # 非匹配键被丢弃(与 sovits 保底分支无关)
    assert strip_prefix(sd, "") == sd
    assert strip_prefix(sd, "module.") == {}  # 全不匹配 -> 空(调用方自行回退)
    assert strip_prefix({}, "model.") == {}


def test_strip_prefix_matches_old_inline_form():
    import random
    from usvc.nn.mlx._convert import strip_prefix

    rng = random.Random(0)
    keys = [f"{p}{i}.w" for i in range(20) for p in ("model.", "module.", "")]
    sd = {k: rng.random() for k in keys}
    for pre in ("model.", "module.", "", "model"):
        old = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        assert strip_prefix(sd, pre) == old


# ---------------------------------------------------------------------------
# E7: _defaults.py 共享表
# ---------------------------------------------------------------------------

def test_ddsp6_opts_single_source_of_truth():
    from usvc.backends._defaults import resolve_ddsp6_opts
    from usvc.backends.ddsp6.module import _resolve_opts

    assert _resolve_opts is resolve_ddsp6_opts  # torch 模块 = 共享函数
    pytest.importorskip("mlx.core")  # mlx 别名断言放独立测试(可跳过)

    o = resolve_ddsp6_opts({"version": "6.1"})
    assert o["velocity_net"] == "LYNXNet" and o["n_aux_layers"] == 3 and o["t_start"] == 0.7
    assert resolve_ddsp6_opts({})["velocity_net"] == "LYNXNet2"  # default 6.3
    merged = resolve_ddsp6_opts({"version": "6.1", "n_chans": 768, "n_units": 512})
    assert merged["n_chans"] == 768 and merged["n_units"] == 512  # user wins + 透传
    with pytest.raises(ValueError, match="ddsp6 version must be one of"):
        resolve_ddsp6_opts({"version": "6.4"})


def test_mlx_ddsp6_opts_same_object():
    pytest.importorskip("mlx.core")
    import usvc.train.mlx.modules.ddsp6 as m

    from usvc.backends._defaults import resolve_ddsp6_opts

    assert m._resolve_opts is resolve_ddsp6_opts


def test_rvc_upsample_table_single_source_of_truth():
    from usvc.backends._defaults import RVC_UPSAMPLE
    from usvc.backends.rvc.module import _RVC_UPSAMPLE

    assert _RVC_UPSAMPLE is RVC_UPSAMPLE  # torch 别名 = 同一对象
    # 不变式:prod(upsample_rates) == hop == sr/100,sr 覆盖 32k/40k/48k
    from math import prod

    assert set(RVC_UPSAMPLE) == {
        ("v1", 32000), ("v1", 40000), ("v1", 48000), ("v2", 32000), ("v2", 48000),
    }
    for (ver, sr), (rates, kernels) in RVC_UPSAMPLE.items():
        assert prod(rates) == sr // 100, (ver, sr)
        assert len(rates) == len(kernels) and rates[0] < kernels[0]  # hifigan 惯例
    # v2 32000 具体值(回归锚)
    assert RVC_UPSAMPLE[("v2", 32000)] == ([10, 8, 2, 2], [20, 16, 4, 4])
    assert RVC_UPSAMPLE[("v2", 48000)] == ([12, 10, 2, 2], [24, 20, 4, 4])