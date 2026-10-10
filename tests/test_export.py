"""usvc-export 测试：剥离训练专用权重 + --half 转换。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from usvc.cli import export


def _fake_ckpt(path: Path) -> None:
    sd = {
        "generator.enc.0.weight": torch.randn(4, 4),
        "generator.dec.0.bias": torch.randn(4),
        "model.layers.0.weight": torch.randn(4, 4),
        "discriminator.d.0.weight": torch.randn(4, 4),
        "optimizer.step": torch.randn(4),
        "criterion.weight": torch.randn(4),
        "loss.weight": torch.randn(4),
    }
    torch.save({"state_dict": sd, "optimizer_states": [{}], "epoch": 3}, path)


def test_export_strips_training_keys(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    _fake_ckpt(ckpt)
    out = tmp_path / "model.pth"
    rc = export.main(["--ckpt", str(ckpt), "--output", str(out)])
    assert rc == 0
    assert out.exists()
    loaded = torch.load(out, map_location="cpu", weights_only=False)
    assert "generator.enc.0.weight" in loaded
    assert "model.layers.0.weight" in loaded
    assert "discriminator.d.0.weight" not in loaded
    assert "optimizer.step" not in loaded
    assert "criterion.weight" not in loaded
    assert "loss.weight" not in loaded


def test_export_half(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    _fake_ckpt(ckpt)
    out = tmp_path / "model_half.pth"
    rc = export.main(["--ckpt", str(ckpt), "--output", str(out), "--half"])
    assert rc == 0
    loaded = torch.load(out, map_location="cpu", weights_only=False)
    assert loaded["generator.enc.0.weight"].dtype == torch.float16


def test_export_missing_ckpt(tmp_path):
    rc = export.main(
        ["--ckpt", str(tmp_path / "nope.ckpt"), "--output", str(tmp_path / "x.pth")]
    )
    assert rc == 1


def test_export_speaker_metadata_half(tmp_path):
    from usvc.train.config import TrainConfig
    from usvc.train.speaker_checkpoint import config_payload, speaker_metadata
    cfg = TrainConfig(backend="ddsp6", backend_opts={"num_speaker": 2}, spk={"歌手 甲": 0, "乙": 1})
    path = tmp_path / "last.ckpt"
    _fake_ckpt(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.update(usvc_speakers=speaker_metadata(cfg), usvc_config=config_payload(cfg))
    torch.save(checkpoint, path)
    output = tmp_path / "model.pth"
    assert export.main(["--ckpt", str(path), "-o", str(output), "--half"]) == 0
    payload = torch.load(output, weights_only=True)
    assert payload["usvc_speakers"]["spk"] == cfg.spk
    assert payload["state_dict"]["model.layers.0.weight"].dtype == torch.float16
    restored = TrainConfig.from_yaml(output.with_suffix(".yaml"))
    assert restored.spk == cfg.spk
