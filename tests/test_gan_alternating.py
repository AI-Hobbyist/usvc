"""GAN backends (sovits/rvc): strict alternating G/D training must not raise.

Regression test for the autograd version-check RuntimeError
("one of the variables needed for gradient computation has been modified by
an inplace operation: [1,1,1,1] ...").

Root cause (verified): it is NOT a weight_norm API bug. AdamW's step() with
default weight_decay=0.01 mutates each weight_norm parameter in-place twice
(``p.mul_(1 - lr*wd)`` + ``p.addcdiv_(...)``), bumping ``_version`` by 2. If a
second loss graph was built before that step (e.g. forwarding both D and G
losses, then stepping D, then backwarding G), the G graph's saved version
(expected at forward time) mismatches the mutated version at backward ->
RuntimeError. The same failure occurs with the modern
``parametrizations.weight_norm``, so the fix is the strict alternating order:

    D fwd -> D bwd -> D opt.step() -> G fwd -> G bwd -> G opt.step()

usvc's training path (BaseGANModule.training_step) and bench_epochs.py already
follow this order; these tests lock it in.
"""
from __future__ import annotations

import pytest
import torch

from usvc.backends.rvc.module import RVCModule
from usvc.backends.sovits.module import SoVITSModule
from usvc.train.config import TrainConfig


def _make_cfg(backend, opts):
    cfg = TrainConfig(backend=backend)
    cfg.backend_opts = opts
    return cfg


def _make_batch(B, T, gen, spec_ch=1025, hop=512, rvc=False):
    """Synthetic SVCBatch on CUDA (mirrors benchmarks/bench_epochs.py)."""
    from usvc.train.dataset import SVCBatch

    import dataclasses

    wave_len = T * hop
    fields = {
        "units": torch.randn(B, T, 768, generator=gen),
        "f0": torch.rand(B, T, generator=gen) * 400 + 100,
        "mel": torch.randn(B, spec_ch, T, generator=gen),
        "wav": torch.randn(B, 1, wave_len, generator=gen),
        "rms": torch.rand(B, T, generator=gen),
        "spk_id": torch.zeros(B, dtype=torch.long),
        "frame_len": torch.full((B,), T),
        "uv": torch.rand(B, T, generator=gen),
        "aug_shift": None,
        "pitch_token": None,
        "key": None,
    }
    if rvc:
        fields["pitch_token"] = torch.randint(1, 256, (B, T), generator=gen)
    b = SVCBatch(**fields)
    return SVCBatch(
        **{
            f.name: (
                getattr(b, f.name).cuda()
                if torch.is_tensor(getattr(b, f.name))
                else getattr(b, f.name)
            )
            for f in dataclasses.fields(b)
        }
    )


def _run_alternating(module_cls, cfg, B, T, seg, hop, rvc, n_steps=8):
    torch.manual_seed(42)
    m = module_cls(cfg).cuda()
    opt_g = torch.optim.AdamW(m.generator.parameters(), lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(m.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99))
    gen = torch.Generator().manual_seed(1)
    losses_g = []
    for _ in range(n_steps):
        b = _make_batch(B, T, gen, hop=hop, rvc=rvc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_d, _ = m.discriminator_loss(b)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_g, _ = m.generator_loss(b)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        losses_g.append(loss_g.item())
    return losses_g


@pytest.mark.parametrize(
    "module_cls,cfg_opts,B,seg,hop,rvc",
    [
        pytest.param(
            SoVITSModule,
            {"segment_size": 32, "num_speaker": 200, "gin_channels": 256,
             "n_units": 768, "spec_channels": 1025},
            2, 32, 512, False,
            id="sovits",
        ),
        pytest.param(
            RVCModule,
            {"segment_size": 36, "num_speaker": 200, "spk_embed_dim": 200,
             "n_units": 768, "spec_channels": 1025, "sr": 48000, "hop": 480},
            2, 36, 480, True,
            id="rvc",
        ),
    ],
)
def test_gan_strict_alternating_no_inplace_error(
    module_cls, cfg_opts, B, seg, hop, rvc
):
    """8 steps of strict alternating G/D training must not raise."""
    cfg = _make_cfg("sovits" if not rvc else "rvc", cfg_opts)
    losses_g = _run_alternating(
        module_cls, cfg, B, B and 64, seg, hop, rvc, n_steps=8
    )
    assert len(losses_g) == 8
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_g)


def test_adamw_step_bumps_version_by_two():
    """Documents why building both graphs before a step fails (root cause)."""
    p = torch.nn.Parameter(torch.ones(1, 1, 1, 1).cuda())
    opt = torch.optim.AdamW([p], lr=2e-4)  # default weight_decay=0.01
    loss = (p * 2).sum()
    loss.backward()
    v0 = p._version
    opt.step()
    assert p._version == v0 + 2  # mul_ (decay) + addcdiv_ (update)
