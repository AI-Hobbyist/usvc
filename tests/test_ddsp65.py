"""DDSP 6.5 integration and optional numerical parity with a local DDSP-SVC checkout.

The standalone tests run in CI. Original-source tests use USVC_DDSP65_DIR or
<repo>/DDSP-SVC and skip explicitly when that external reference is unavailable.
"""
from __future__ import annotations

import ast
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from usvc.backends._defaults import resolve_ddsp6_opts
from usvc.backends.ddsp6 import DDSP6Backend
from usvc.backends.ddsp6.model import DDSP6Model
from usvc.backends.ddsp6.module import DDSPModule
from usvc.backends.ddsp6.v65 import LYNXNet2_65, RectifiedFlow65
from usvc.train.config import TrainConfig
from usvc.train.dataset import SVCBatch

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _model(glu_type="softsign_glu", speakers=1):
    return DDSP6Model(n_units=16, n_spk=speakers, n_aux_layers=1, n_aux_chans=32,
                      n_layers=1, n_chans=32, velocity_net="LYNXNet2_65",
                      glu_type=glu_type).to(DEVICE)


def _data():
    return (torch.randn(2, 8, 16, device=DEVICE),
            torch.rand(2, 8, device=DEVICE) * 300 + 100,
            torch.rand(2, 8, device=DEVICE),
            torch.tensor([0, 1], device=DEVICE),
            torch.randn(2, 8, 128, device=DEVICE))


def test_config_template_and_defaults():
    cfg = TrainConfig.from_yaml(str(ROOT / "configs/ddsp65.yaml"))
    opts = resolve_ddsp6_opts(cfg.backend_opts)
    assert cfg.backend == "ddsp6"
    assert opts["velocity_net"] == "LYNXNet2_65"
    assert opts["glu_type"] == "softsign_glu"
    assert (opts["n_aux_layers"], opts["n_aux_chans"], opts["n_chans"]) == (6, 512, 1024)
    spec = DDSP6Backend.feature_spec(cfg.backend_opts)
    assert spec.units.params["kind"] == "contentvec_tta2x"
    assert spec.f0.params["extractor"] == "rmvpe"
    assert spec.aug_mel is not None and spec.aug_shift is not None
    assert resolve_ddsp6_opts({})["velocity_net"] == "LYNXNet2"
    assert resolve_ddsp6_opts({"version": "6.5", "glu_type": "atanglu"})["glu_type"] == "atanglu"
    module = DDSPModule(cfg)
    assert isinstance(module.model.reflow_model.velocity_fn, LYNXNet2_65)
    assert len(module.model.ddsp_model.unit2ctrl.residual_layers) == 6
    assert module.model.reflow_model.velocity_fn.input_projection.out_features == 1024


@pytest.mark.parametrize("glu", ["atanglu", "softsign_glu", "double_softsign_glu"])
def test_training_sampling_and_checkpoint(glu, tmp_path):
    torch.manual_seed(5)
    model = _model(glu, speakers=2)
    units, f0, volume, sid, target = _data()
    loss, sub = model(units, f0, volume, sid, target,
                      aug_shift=torch.tensor([[[2.]], [[-1.]]], device=DEVICE),
                      frame_len=torch.tensor([8, 6]))
    assert torch.isfinite(loss) and set(sub) == {"ddsp_loss", "reflow_loss"}
    loss.backward()
    assert model.ddsp_model.unit2ctrl.dense_out.weight.grad.abs().sum() > 0
    assert model.reflow_model.velocity_fn.output_projection.weight.grad.abs().sum() > 0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    model.eval()
    for method in ("euler", "rk4"):
        with torch.no_grad():
            mel = model.sample(units, f0, volume, sid, infer_step=2, method=method)
        assert mel.shape == target.shape and torch.isfinite(mel).all()
    path = tmp_path / "v65.pt"
    torch.save(model.state_dict(), path)
    restored = _model(glu, speakers=2)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    legacy = DDSP6Model(n_units=16, n_spk=2, n_aux_layers=1, n_aux_chans=32,
                        n_layers=1, n_chans=32)
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(model.state_dict(), strict=True)


def test_lightning_module_and_inference_loader(tmp_path):
    opts = dict(version="6.5", n_units=16, num_speaker=2, n_aux_layers=1,
                n_aux_chans=32, n_layers=1, n_chans=32, use_pitch_aug=False)
    cfg = TrainConfig(backend="ddsp6", backend_opts=opts)
    module = DDSPModule(cfg).to(DEVICE)
    assert isinstance(module.model.reflow_model, RectifiedFlow65)
    assert module.model.ddsp_model.unit2ctrl.aug_shift_embed is None
    units, f0, rms, sid, mel = _data()
    batch = SVCBatch(units=units, f0=f0, rms=rms, spk_id=sid, mel=mel,
                     frame_len=torch.tensor([8, 8]))
    loss, _ = module.forward_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(module.sample(batch, steps=2)).all()
    # Exercise the actual inference engine remapping of original model.* keys.
    from usvc.infer.engine import InferenceEngine
    path = tmp_path / "original.pt"
    torch.save({"model": module.model.state_dict()}, path)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.ckpt_path, engine.backend_name = str(path), "ddsp6"
    engine._load_weights(DDSPModule(cfg))


def test_dual_time_mask_is_not_padding_mask():
    class Probe(nn.Module):
        def forward(self, x, t1, cond, t2=None, mask=None):
            self.inputs = x, t1, t2, mask
            return x * 0

    velocity = Probe()
    flow = RectifiedFlow65(velocity, out_dims=3)
    x1 = torch.ones(2, 1, 3, 4)
    t1, t2 = torch.full((2, 1), .2), torch.full((2, 1), .8)
    time_mask = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]])
    valid = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]])
    torch.manual_seed(7)
    noise = torch.randn_like(x1)
    t = t1 + (t2 - t1) * time_mask
    weights = .398942 / t / (1 - t) * torch.exp(-.5 * torch.log(t / (1 - t)) ** 2)
    expected = ((x1 - noise).square() * weights[:, None, None] * valid[:, None, None]).sum() / (valid.sum() * 3)
    torch.manual_seed(7)
    actual = flow.reflow_loss(x1, t1, torch.zeros(2, 3, 4), t2, time_mask, valid_mask=valid)
    torch.testing.assert_close(actual, expected)
    xt, first, second, selected = velocity.inputs
    torch.testing.assert_close(xt, noise + t[:, None, None] * (x1 - noise))
    assert torch.equal(first, t1 * 1000) and torch.equal(second, t2 * 1000)
    assert torch.equal(selected, time_mask)


@pytest.fixture
def original():
    root = Path(os.environ.get("USVC_DDSP65_DIR", ROOT / "refs/ddsp65"))
    if not (root / "reflow/lynxnet2.py").exists():
        pytest.skip("local DDSP-SVC 6.5 reference unavailable")
    # Load only the original model definitions, avoiding unrelated pitch-extractor
    # dependencies imported at module scope by upstream ddsp/vocoder.py.
    ns = dict(torch=torch, nn=nn, F=F, np=np, math=__import__("math"),
              weight_norm=nn.utils.weight_norm, tqdm=lambda x, **kw: x)
    for relative, names in [
        ("reflow/lynxnet2.py", None),
        ("ddsp/unit2control.py", {"split_to_dict", "Unit2Control"}),
        ("ddsp/vocoder.py", {"CombSubSuperFast"}),
        ("reflow/reflow.py", {"RectifiedFlow"}),
        ("reflow/vocoder.py", {"Unit2Wav"}),
        ("nsf_hifigan/nvSTFT.py", {"STFT", "dynamic_range_compression_torch"}),
    ]:
        source = ast.parse((root / relative).read_text(encoding="utf-8"))
        source.body = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                       and (names is None or n.name in names)]
        exec(compile(source, str(root / relative), "exec"), ns)
    from librosa.filters import mel
    ns["librosa_mel_fn"] = mel
    return ns


@pytest.mark.parametrize("glu", ["atanglu", "softsign_glu", "double_softsign_glu"])
@pytest.mark.parametrize("amp", [False, pytest.param(True, marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA BF16 required"))])
def test_original_full_model_loss_gradients_and_sampling(original, glu, amp):
    torch.manual_seed(10)
    reference = original["Unit2Wav"](44100, 512, 2048, 16, 2, True,
                                      128, 1, 32, 1, 32, glu_type=glu).to(DEVICE)
    if amp:
        # Raw upstream hits unsupported BComplex32 on this PyTorch. Apply only
        # the same FP32 spectral promotion used by usvc; leave its core intact.
        reference.ddsp_model.unit2ctrl.register_forward_hook(
            lambda module, args, out: ({k: v.float() for k, v in out[0].items()}, out[1]))
    # Avoid trivial zero-init velocity output masking architectural differences.
    nn.init.normal_(reference.reflow_model.velocity_fn.output_projection.weight, std=.01)
    model = _model(glu, speakers=2)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert list(model.state_dict()) == list(reference.state_dict())
    units, f0, volume, sid, target = _data()
    shifts = torch.tensor([[[2.]], [[-1.]]], device=DEVICE)
    torch.manual_seed(12)
    with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
        ref_ddsp, ref_flow = reference(units, f0[..., None], volume[..., None],
                                      spk_id=sid[:, None] + 1, aug_shift=shifts,
                                      vocoder=model.mel_extractor, gt_spec=target, infer=False)
        ref_total = ref_ddsp + ref_flow
    ref_total.backward()
    torch.manual_seed(12)
    with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
        total, sub = model(units, f0, volume, sid, target, aug_shift=shifts)
    total.backward()
    torch.testing.assert_close(total, ref_total, rtol=0, atol=0)
    assert sub["ddsp_loss"] == ref_ddsp.item() and sub["reflow_loss"] == ref_flow.item()
    for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        if ref_param.grad is not None:
            # CUDA reflection-pad backward has no deterministic implementation.
            # Under AMP its roundoff can become one BF16 unit, so bound both
            # maximum error and relative L2 error by BF16 machine epsilon.
            if amp:
                epsilon = torch.finfo(torch.bfloat16).eps
                diff = param.grad - ref_param.grad
                assert diff.norm() <= epsilon * ref_param.grad.norm() + 2e-7, name
                atol = epsilon * ref_param.grad.abs().max().item() + 2e-7
                torch.testing.assert_close(param.grad, ref_param.grad, rtol=0, atol=atol,
                                           msg=lambda detail: name + ": " + detail)
            else:
                torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-5, atol=2e-7,
                                           msg=lambda detail: name + ": " + detail)
    model.eval()
    reference.eval()
    for method in ("euler", "rk4"):
        torch.manual_seed(13)
        with torch.no_grad():
            ref_mel = reference(units, f0[..., None], volume[..., None],
                                spk_id=sid[:, None] + 1, vocoder=model.mel_extractor,
                                infer_step=3, method=method, t_start=.3, use_tqdm=False)
        torch.manual_seed(13)
        with torch.no_grad():
            mel = model.sample(units, f0, volume, sid, infer_step=3, method=method, t_start=.3)
        torch.testing.assert_close(mel, ref_mel, rtol=0, atol=0)


def test_original_mel_frontend(original):
    from usvc.backends.ddsp6.model import MelExtractor
    reference = original["STFT"](sr=44100, n_mels=128, n_fft=2048,
                                  win_size=2048, hop_length=512, fmin=40, fmax=16000)
    current = MelExtractor()
    wav = torch.randn(2, 4096, device=DEVICE, requires_grad=True)
    ref_mel = reference.get_mel(wav).transpose(1, 2)
    mel = current.extract(wav)
    torch.testing.assert_close(mel, ref_mel, rtol=0, atol=0)
    actual_grad, = torch.autograd.grad(mel.sum(), wav, retain_graph=True)
    ref_grad, = torch.autograd.grad(ref_mel.sum(), wav)
    torch.testing.assert_close(actual_grad, ref_grad, rtol=1e-5, atol=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 required")
def test_template_lightning_training(tmp_path):
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader

    cfg = TrainConfig.from_yaml(str(ROOT / "configs/ddsp65.yaml"))
    module = DDSPModule(cfg)
    batch = SVCBatch(units=torch.randn(2, 8, 768), f0=torch.full((2, 8), 220.),
                     mel=torch.randn(2, 8, 128), rms=torch.rand(2, 8),
                     spk_id=torch.zeros(2, dtype=torch.long),
                     aug_shift=torch.zeros(2, 1, 1), frame_len=torch.tensor([8, 8]))
    loader = DataLoader([batch, batch], batch_size=None, num_workers=0)
    trainer = Trainer(accelerator="gpu", devices=1, precision=cfg.precision,
                      max_steps=2, limit_val_batches=0, logger=False,
                      enable_checkpointing=False, default_root_dir=str(tmp_path),
                      enable_model_summary=False, enable_progress_bar=False)
    before = module.model.reflow_model.velocity_fn.output_projection.weight.detach().clone()
    trainer.fit(module, train_dataloaders=loader)
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["train/loss"])
    assert not torch.equal(before, module.model.reflow_model.velocity_fn.output_projection.weight.detach().cpu())
