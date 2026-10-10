"""Real model optimization and Lightning lifecycle matrix; no feature/audio mocks."""
import gc

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from usvc.backends.ddsp6.module import DDSPModule
from usvc.backends.sovits.module import SoVITSModule
from usvc.cli import export
from usvc.infer.engine import InferenceEngine
from usvc.train.config import TrainConfig
from usvc.train.dataset import SVCBatch


def make_cfg(kind, speakers, directory):
    opts = dict(num_speaker=speakers, n_units=16)
    if kind == "sovits":
        opts.update(spec_channels=1025, inter_channels=32, hidden_channels=32,
                    filter_channels=64, gin_channels=32, n_layers_enc=1, n_layers_dec=1,
                    n_sdp_layers0=1, n_sdp_layers=1, segment_size=8,
                    use_automatic_f0_prediction=False)
    else:
        opts.update(version=kind, n_aux_layers=1, n_aux_chans=32, n_layers=1, n_chans=32)
    return TrainConfig(backend="sovits" if kind == "sovits" else "ddsp6", backend_opts=opts,
                       spk={n: i for i, n in enumerate(["歌手 甲", "歌手乙"][:speakers])},
                       exp_dir=str(directory), max_steps=1, precision="32-true", grad_clip=None)


def make_batch(kind, speakers):
    b, t = 2, 12
    batch = SVCBatch(units=torch.randn(b, t, 16), f0=torch.full((b, t), 220.),
                     rms=torch.rand(b, t), spk_id=torch.arange(b) % speakers,
                     frame_len=torch.full((b,), t))
    batch.mel = torch.randn(b, 1025, t) if kind == "sovits" else torch.randn(b, t, 128)
    if kind == "sovits":
        batch.wav = torch.randn(b, 1, t * 512) * .05
        batch.uv = torch.ones(b, t)
    return batch


def trainer(path, steps):
    return Trainer(accelerator="gpu", devices=1, precision="32-true", max_steps=steps,
                   logger=False, enable_checkpointing=False, limit_val_batches=0,
                   enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(path))


@pytest.mark.parametrize("kind", ["6.1", "6.2", "6.3", "6.5", "sovits"])
@pytest.mark.parametrize("speakers", [1, 2])
def test_real_training_resume_export_matrix(tmp_path, kind, speakers):
    torch.manual_seed(100)
    cfg = make_cfg(kind, speakers, tmp_path)
    cls = SoVITSModule if kind == "sovits" else DDSPModule
    module = cls(cfg)
    embedding = module.generator.emb_g if kind == "sovits" else getattr(module.model.ddsp_model.unit2ctrl, "spk_embed", None)
    before = embedding.weight.detach().clone() if embedding is not None else None
    loader = DataLoader([make_batch(kind, speakers)], batch_size=None)
    first = trainer(tmp_path, 1)
    first.fit(module, train_dataloaders=loader)
    if embedding is not None:
        assert torch.isfinite(embedding.weight).all()
        assert torch.isfinite(embedding.weight.grad).all()
        assert (embedding.weight.grad.abs().sum(1) > 0).all()
        assert (embedding.weight.detach().cpu() - before).abs().sum(1).gt(0).all()
    for key, value in first.callback_metrics.items():
        if "loss" in key:
            assert torch.isfinite(value).all()
    path = tmp_path / "last.ckpt"
    first.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["usvc_speakers"]["spk"] == cfg.spk
    assert checkpoint["global_step"] > 0 and all(o["state"] for o in checkpoint["optimizer_states"])
    changed = make_cfg(kind, speakers, tmp_path)
    changed.backend_opts = dict(changed.backend_opts, n_units=32)
    with pytest.raises(ValueError, match="resume architecture"):
        cls(changed).on_load_checkpoint(checkpoint)
    # Restore without taking another step: state, optimizer moments and global step must match.
    restored = cls(cfg)
    second = trainer(tmp_path, first.global_step)
    second.fit(restored, train_dataloaders=loader, ckpt_path=str(path))
    assert second.global_step == first.global_step
    for key, value in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    for loaded_opt, saved_opt in zip(second.optimizers, checkpoint["optimizer_states"]):
        loaded = loaded_opt.state_dict()
        assert loaded["param_groups"] == saved_opt["param_groups"]
        for pid, states in saved_opt["state"].items():
            for name, value in states.items():
                torch.testing.assert_close(loaded["state"][pid][name].cpu(), value.cpu(), rtol=0, atol=0)
    if speakers == 2:
        changed = make_cfg(kind, 2, tmp_path)
        changed.spk = {"歌手乙": 0, "歌手 甲": 1}
        with pytest.raises(ValueError, match="resume speaker"):
            cls(changed).on_load_checkpoint(checkpoint)
        changed = make_cfg(kind, 2, tmp_path)
        changed.spk["新增"] = 2; changed.backend_opts["num_speaker"] = 3
        with pytest.raises(ValueError, match="resume speaker"):
            cls(changed).on_load_checkpoint(checkpoint)
    output = tmp_path / "model.pth"
    assert export.main(["--ckpt", str(path), "-o", str(output)]) == 0
    result = torch.load(output, weights_only=True)
    assert result["usvc_speakers"] == checkpoint["usvc_speakers"]
    assert output.with_suffix(".yaml").exists()
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.ckpt_path, engine.backend_name = str(output), cfg.backend
    engine._load_weights(cls(cfg))
    # Existing bare tensors still load; capacity mismatches must never leave random embedding rows.
    torch.save(result["state_dict"], tmp_path / "bare.pth")
    engine.ckpt_path = str(tmp_path / "bare.pth")
    engine._load_weights(cls(cfg))
    if speakers == 2:
        with pytest.raises(RuntimeError):
            engine._load_weights(cls(make_cfg(kind, 3, tmp_path)))
    del module, restored, first, second
    gc.collect(); torch.cuda.empty_cache()


@pytest.mark.parametrize("kind", ["6.1", "6.2", "6.3", "6.5"])
def test_ddsp_batch_condition_matches_each_sample(tmp_path, kind):
    cfg = make_cfg(kind, 2, tmp_path)
    control = DDSPModule(cfg).model.ddsp_model.unit2ctrl.eval().cuda()
    args = [torch.randn(2, 7, 16, device="cuda"), torch.randn(2, 7, 512, device="cuda"),
            torch.randn(2, 7, 512, device="cuda"), torch.rand(2, 7, 1, device="cuda")]
    ids = torch.tensor([0, 1], device="cuda")
    with torch.no_grad():
        _, batch = control(*args, spk_id=ids)
        for i in range(2):
            _, one = control(*(a[i:i+1] for a in args), spk_id=ids[i:i+1])
            torch.testing.assert_close(batch[i:i+1], one, rtol=1e-5, atol=2e-6)
        _, swapped = control(*args, spk_id=ids.flip(0))
    assert not torch.equal(batch, swapped)


@pytest.mark.parametrize("version", ["6.1", "6.2", "6.3"])
def test_original_ddsp_condition_and_gradients(tmp_path, version):
    import ast
    from pathlib import Path
    import numpy as np
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np,
              weight_norm=torch.nn.utils.weight_norm)
    root = Path("refs") / ("ddsp" + version.replace(".", ""))
    for name in ("model_conformer_naive.py", "unit2control.py"):
        source = ast.parse((root / "ddsp" / name).read_text(encoding="utf-8"))
        source.body = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(source, str(root / name), "exec"), ns)
    current = DDSPModule(make_cfg(version, 2, tmp_path)).model.ddsp_model.unit2ctrl.cuda()
    reference = ns["Unit2Control"](16, 512, 2, current.output_splits, num_layers=1,
                                    dim_model=32, use_norm=version != "6.1",
                                    use_pitch_aug=True).cuda()
    reference.load_state_dict(current.state_dict(), strict=True)
    args = [torch.randn(2, 7, 16, device="cuda"), torch.randn(2, 7, 512, device="cuda"),
            torch.randn(2, 7, 512, device="cuda"), torch.rand(2, 7, 1, device="cuda")]
    ids = torch.tensor([0, 1], device="cuda")
    controls, hidden = current(*args, spk_id=ids)
    expected, expected_hidden = reference(*args, spk_id=ids[:, None] + 1)
    torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
    for key in controls:
        torch.testing.assert_close(controls[key], expected[key], rtol=0, atol=0)
    sum(c.square().mean() for c in controls.values()).backward()
    sum(c.square().mean() for c in expected.values()).backward()
    torch.testing.assert_close(current.spk_embed.weight.grad, reference.spk_embed.weight.grad,
                               rtol=1e-5, atol=2e-7)
