"""ddsp6 backend: LYNXNet (6.1/6.2) vs LYNXNet2 (6.3) velocity net selection.

These tests run without the original DDSP-SVC checkouts (the diff=0 numerical
validation against the original projects is done in usvc/train scripts and
documented in TRAINING_VALIDATION_REPORT.md).
"""
from __future__ import annotations

import pytest
import torch

from usvc.backends.ddsp6.model import DDSP6Model, LYNXNet, LYNXNet2
from usvc.backends.ddsp6.module import _resolve_opts


def _make_model(velocity_net="LYNXNet2"):
    torch.manual_seed(0)
    return DDSP6Model(
        n_units=768, n_spk=1, n_aux_layers=3, n_aux_chans=256,
        n_layers=2, n_chans=512, velocity_net=velocity_net,
    ).cuda()


def _data():
    B, T = 2, 64
    return (
        torch.randn(B, T, 768).cuda(),
        torch.rand(B, T).cuda() * 400 + 100,
        torch.rand(B, T).cuda(),
        torch.zeros(B, dtype=torch.long).cuda(),
        torch.randn(B, T, 128).cuda(),
    )


def test_velocity_net_types():
    assert isinstance(_make_model("LYNXNet").reflow_model.velocity_fn, LYNXNet)
    assert isinstance(_make_model("LYNXNet2").reflow_model.velocity_fn, LYNXNet2)
    # default must stay 6.3-compatible
    assert isinstance(_make_model().reflow_model.velocity_fn, LYNXNet2)


def test_invalid_velocity_net():
    with pytest.raises(ValueError):
        _make_model("bogus")


def test_forward_backward_both_nets():
    units, f0, vol, sid, spec = _data()
    for name, kw in [("LYNXNet", {"velocity_net": "LYNXNet"}), ("LYNXNet2", {})]:
        m = DDSP6Model(n_units=768, n_spk=1, n_aux_layers=3, n_aux_chans=256,
                       n_layers=2, n_chans=512, **kw).cuda()
        loss, sub = m(units, f0, vol, sid, spec)
        assert loss.requires_grad
        assert set(sub) == {"ddsp_loss", "reflow_loss"}
        loss.backward()
        # both sub-models must receive gradients
        ddsp_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in m.ddsp_model.parameters())
        # velocity output projection is zero-init: only it gets grad on step 0
        rf_grad = m.reflow_model.velocity_fn.output_projection.weight.grad is not None
        assert ddsp_grad and rf_grad, name


def test_version_defaults():
    o = _resolve_opts({"version": "6.1"})
    assert o["velocity_net"] == "LYNXNet"
    assert o["n_aux_layers"] == 3 and o["n_aux_chans"] == 256
    assert o["n_chans"] == 512 and o["t_start"] == 0.7 and o["use_norm"] is False

    o = _resolve_opts({"version": "6.2"})
    assert o["velocity_net"] == "LYNXNet"
    assert o["n_aux_layers"] == 6 and o["n_aux_chans"] == 512
    assert o["n_chans"] == 1024 and o["t_start"] == 0.0 and o["use_norm"] is True

    o = _resolve_opts({})  # default 6.3
    assert o["velocity_net"] == "LYNXNet2"

    # explicit opts override version defaults
    o = _resolve_opts({"version": "6.1", "n_chans": 768})
    assert o["n_chans"] == 768 and o["velocity_net"] == "LYNXNet"

    with pytest.raises(ValueError):
        _resolve_opts({"version": "6.4"})


def test_sample_both_nets():
    units, f0, vol, _, _ = _data()
    sid = torch.zeros(units.size(0), dtype=torch.long).cuda()
    for name, kw in [("LYNXNet", {"velocity_net": "LYNXNet", "t_start": 0.7}),
                     ("LYNXNet2", {})]:
        m = DDSP6Model(n_units=768, n_spk=1, n_aux_layers=3, n_aux_chans=256,
                       n_layers=2, n_chans=512, **kw).cuda().eval()
        with torch.no_grad():
            mel = m.sample(units, f0, vol, sid, infer_step=4, method="euler", use_tqdm=False)
        assert mel.shape == (units.shape[0], units.shape[1], 128)
        assert torch.isfinite(mel).all(), name
