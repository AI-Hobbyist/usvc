"""E8 anchor: BaseSVCModule._unpack == manual unpack + mel-energy volume fallback.

The helper must be bit-identical to the inlined per-backend code it replaces
(ddsp6/diffusion/reflow_vae forward_loss + sample).
"""
from __future__ import annotations

import torch

from usvc.train.dataset import SVCBatch
from usvc.train.module import BaseSVCModule


def _batch(rms: bool, aug: bool) -> SVCBatch:
    B, T, D, M = 2, 7, 768, 128
    kw = dict(
        units=torch.randn(B, T, D),
        f0=torch.rand(B, T) * 400,
        mel=torch.rand(B, T, M) * 3 - 1,
        spk_id=torch.randint(0, 2, (B,)),
        aug_shift=torch.randn(B, 1, 1) if aug else None,
    )
    if rms:
        kw["rms"] = torch.rand(B, T)
    return SVCBatch(**kw)


def test_unpack_fields_are_same_objects():
    b = _batch(rms=True, aug=True)
    u, f, v, s, m, a = BaseSVCModule._unpack(b)
    assert u is b.units and f is b.f0 and s is b.spk_id
    assert m is b.mel and a is b.aug_shift
    assert v is b.rms  # rms present -> no fallback computed


def test_unpack_volume_fallback_bit_equal_to_manual():
    b = _batch(rms=False, aug=False)
    _, _, v, _, _, _ = BaseSVCModule._unpack(b)
    expect = (b.mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)
    assert torch.equal(v, expect)
    assert v.shape == (b.units.shape[0], b.units.shape[1])