"""Shallow diffusion (sovits -shd) tests.

Covers:
  - DiffusionEnhancer pipeline (mel extract → diffusion sample → vocoder)
  - mel/units frame alignment (trim T+1 → T, pad short mel)
  - rms=None volume fallback
  - engine wiring (shallow_diffusion toggle, k_step/method/speedup passthrough)
  - error paths (shallow_diffusion without enhancer, non-sovits backend)

Uses tiny random-weight models + a dummy vocoder; no pretrained weights needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"


class _DummyVocoder:
    """Shape-only vocoder: returns zeros of length T*hop."""

    hop = 64

    def __init__(self, *args, **kwargs):
        pass

    def infer(self, mel, f0):
        T = mel.shape[1]
        return torch.zeros(mel.shape[0], 1, T * self.hop, device=mel.device)


def _diffusion_cfg(path: Path, out_dims: int = 8, k_step: int = 300, **over) -> dict:
    cfg = {
        "n_units": 16,
        "n_spk": 1,
        "out_dims": out_dims,
        "n_hidden": 32,
        "timesteps": 1000,
        "k_step": k_step,
        "spec_min": -12,
        "spec_max": 2,
        "denoiser": "convnext",
        "cn_layers": 2,
        "cn_chans": 16,
        "mlp_factor": 2,
        "wn_layers": 2,
        "wn_chans": 16,
        "units_kind": "hubertsoft",
        "mel": {
            "n_fft": 256,
            "win_length": 256,
            "n_mels": out_dims,
            "fmin": 40.0,
            "fmax": 8000.0,
            "hop": 64,
            "sampling_rate": 16000,
        },
    }
    cfg.update(over)
    import yaml

    path.write_text(yaml.safe_dump(cfg))
    return cfg


def _diffusion_ckpt(path: Path, cfg_path: Path) -> None:
    """Save a tiny DiffusionModel ckpt with lightning-style ``model.`` prefix."""
    import yaml

    from usvc.backends.diffusion.model import DiffusionModel

    cfg = yaml.safe_load(cfg_path.read_text())
    m = DiffusionModel(
        n_units=cfg["n_units"], n_spk=cfg["n_spk"], out_dims=cfg["out_dims"],
        n_hidden=cfg["n_hidden"], timesteps=cfg["timesteps"],
        k_step=cfg["k_step"], spec_min=cfg["spec_min"], spec_max=cfg["spec_max"],
        denoiser=cfg["denoiser"], cn_layers=cfg["cn_layers"],
        cn_chans=cfg["cn_chans"], mlp_factor=cfg["mlp_factor"],
        wn_layers=cfg["wn_layers"], wn_chans=cfg["wn_chans"],
    )
    sd = {"model." + k: v for k, v in m.state_dict().items()}
    torch.save({"state_dict": sd, "epoch": 0, "global_step": 0}, path)


def _build_enhancer(tmp_path: Path, monkeypatch, **cfg_over):
    """Build a DiffusionEnhancer with a dummy vocoder + tiny diffusion model.

    Standalone-enhancer tests use sr=16000/hop=64 (config overrides); engine
    tests must match the engine's sr/hop (44100/512) instead.
    """
    from usvc.infer.diffusion_enhancer import DiffusionEnhancer

    monkeypatch.setattr("usvc.infer.diffusion_enhancer.NSFHiFiGAN", _DummyVocoder)
    cfg_path = tmp_path / "diff.yaml"
    _diffusion_cfg(cfg_path, **cfg_over)
    ckpt_path = tmp_path / "diff.ckpt"
    _diffusion_ckpt(ckpt_path, cfg_path)
    return DiffusionEnhancer(
        ckpt_path, cfg_path, tmp_path, device=DEVICE, sr=16000, hop=64
    )


class TestDiffusionEnhancer:
    def test_enhancer_pipeline_shapes(self, tmp_path, monkeypatch):
        enh = _build_enhancer(tmp_path, monkeypatch)
        T = 40
        units = torch.randn(T, 16, device=DEVICE)
        f0 = torch.rand(T, device=DEVICE) * 200 + 100
        rms = torch.rand(T, device=DEVICE) + 0.1
        wav = torch.randn(T * 64, device=DEVICE) * 0.1
        out = enh.enhance(units, f0, rms, 0, wav, k_step=50)
        assert out.shape == (T * 64,)
        assert np.isfinite(out).all()

    def test_enhancer_rms_none_fallback(self, tmp_path, monkeypatch):
        enh = _build_enhancer(tmp_path, monkeypatch)
        T = 30
        units = torch.randn(T, 16, device=DEVICE)
        f0 = torch.rand(T, device=DEVICE) * 200 + 100
        wav = torch.randn(T * 64, device=DEVICE) * 0.1
        out = enh.enhance(units, f0, None, 0, wav, k_step=50)
        assert out.shape == (T * 64,)
        assert np.isfinite(out).all()

    def test_enhancer_mel_align_trim_and_pad(self, tmp_path, monkeypatch):
        """mel has T+1 frames (extractor) → trim; short wav → pad."""
        enh = _build_enhancer(tmp_path, monkeypatch)
        T = 33
        units = torch.randn(T, 16, device=DEVICE)
        f0 = torch.rand(T, device=DEVICE) * 200 + 100
        # 正常: wav = T*hop → mel T+1 帧, 裁剪到 T
        out = enh.enhance(units, f0, None, 0, torch.randn(T * 64, device=DEVICE), k_step=30)
        assert out.shape == (T * 64,)
        # 短 wav: 少于 T*hop → mel 帧数不足, pad 到 T
        out2 = enh.enhance(units, f0, None, 0, torch.randn(T * 64 - 128, device=DEVICE), k_step=30)
        assert out2.shape == (T * 64,)
        assert np.isfinite(out2).all()

    def test_enhancer_kstep_passthrough(self, tmp_path, monkeypatch):
        """k_step / infer_speedup / method 必须透传给 diffusion model."""
        from usvc.backends.diffusion.model import DiffusionModel
        from usvc.infer import diffusion_enhancer as de_mod

        monkeypatch.setattr("usvc.infer.diffusion_enhancer.NSFHiFiGAN", _DummyVocoder)
        cfg_path = tmp_path / "diff.yaml"
        _diffusion_cfg(cfg_path)
        ckpt_path = tmp_path / "diff.ckpt"
        _diffusion_ckpt(ckpt_path, cfg_path)

        received = {}

        class _SpyModel(DiffusionModel):
            def sample(self, *args, **kwargs):
                received["k_step"] = kwargs.get("k_step")
                received["method"] = kwargs.get("method")
                received["speedup"] = kwargs.get("infer_speedup")
                received["gt_spec"] = kwargs.get("gt_spec")
                return torch.randn(1, args[0].shape[1], self.out_dims, device=args[0].device)

        monkeypatch.setattr(de_mod, "DiffusionModel", _SpyModel)
        enh = de_mod.DiffusionEnhancer(
            ckpt_path, cfg_path, tmp_path, device=DEVICE, sr=16000, hop=64
        )
        T = 20
        units = torch.randn(T, 16, device=DEVICE)
        f0 = torch.rand(T, device=DEVICE) * 200 + 100
        enh.enhance(units, f0, None, 0, torch.randn(T * 64, device=DEVICE),
                    k_step=17, infer_speedup=7, method="ddpm")
        assert received["k_step"] == 17
        assert received["speedup"] == 7
        assert received["method"] == "ddpm"
        assert received["gt_spec"] is not None
        assert received["gt_spec"].shape == (1, T, 8)  # 对齐后 mel

    def test_load_rejects_bad_ckpt(self, tmp_path, monkeypatch):
        from usvc.infer.diffusion_enhancer import DiffusionEnhancer

        monkeypatch.setattr("usvc.infer.diffusion_enhancer.NSFHiFiGAN", _DummyVocoder)
        cfg_path = tmp_path / "diff.yaml"
        _diffusion_cfg(cfg_path)
        bad = tmp_path / "bad.ckpt"
        torch.save({"state_dict": {"foo.bar": torch.zeros(1)}}, bad)
        with pytest.raises(RuntimeError, match="checkpoint keys"):
            DiffusionEnhancer(bad, cfg_path, tmp_path, device=DEVICE, sr=16000, hop=64)


# 引擎测试用的扩散配置: mel 参数必须匹配引擎 sr/hop (44100/512)
_ENGINE_MEL = {
    "n_fft": 1024, "win_length": 1024, "n_mels": 8,
    "fmin": 40.0, "fmax": 8000.0, "hop": 512, "sampling_rate": 44100,
}


def _sovits_module_ckpt(path: Path) -> None:
    """Tiny sovits module ckpt (lightning state_dict with generator.*)."""
    from usvc.backends import get_backend
    from usvc.train.config import TrainConfig

    cfg = TrainConfig(backend="sovits")
    # n_units 必须 768: 推理特征来自 contentvec (768 维)
    cfg.backend_opts = {
        "n_units": 768, "spec_channels": 1025, "num_speaker": 1,
        "gin_channels": 64, "segment_size": 8, "inter_channels": 32,
        "hidden_channels": 32, "filter_channels": 64, "units_kind": "contentvec",
    }
    module = get_backend("sovits")().build_module(cfg)
    torch.save({"state_dict": module.state_dict(), "epoch": 0, "global_step": 0}, path)


def _sovits_cfg(path: Path) -> None:
    lines = [
        "backend: sovits", "dataset_dir: /tmp", "exp_dir: /tmp/exp",
        "backend_opts:",
        "  n_units: 768", "  spec_channels: 1025", "  num_speaker: 1",
        "  gin_channels: 64", "  segment_size: 8", "  inter_channels: 32",
        "  hidden_channels: 32", "  filter_channels: 64",
        "  units_kind: contentvec",
    ]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(not CUDA, reason="CUDA required")
class TestEngineShallowDiffusion:
    def test_convert_raises_without_enhancer(self, tmp_path):
        from usvc.infer.engine import InferenceEngine
        from usvc.infer.features import synth_test_wav

        ckpt = tmp_path / "sovits.ckpt"
        cfg = tmp_path / "sovits.yaml"
        _sovits_module_ckpt(ckpt)
        _sovits_cfg(cfg)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=44100, duration=0.5, f0_base=220.0)
        engine = InferenceEngine(cfg, ckpt, device="cuda")
        with pytest.raises(ValueError, match="requires diffusion_ckpt"):
            engine.convert(str(wav_in), spk=0, shallow_diffusion=True)

    def test_non_sovits_backend_rejected(self, tmp_path):
        from usvc.backends import get_backend
        from usvc.infer.engine import InferenceEngine
        from usvc.train.config import TrainConfig

        # 真实 tiny diffusion module (mel-domain backend, 不支持浅扩散)
        # 用 diffusion 作为不支持浅扩散的代表
        from usvc.train.config import TrainConfig
        cfg = TrainConfig(backend="diffusion")
        cfg.backend_opts = {
            "n_units": 64, "num_speakers": 1, "out_dims": 128,
            "cn_layers": 2, "cn_chans": 32, "n_hidden": 32,
        }
        from usvc.backends import get_backend
        module = get_backend("diffusion")().build_module(cfg)
        ckpt = tmp_path / "diffusion.ckpt"
        torch.save({"state_dict": module.state_dict()}, ckpt)
        cfgp = tmp_path / "diffusion.yaml"
        cfgp.write_text(
            "backend: diffusion\ndataset_dir: /tmp\nexp_dir: /tmp/exp\nbackend_opts:\n"
            "  n_units: 64\n  num_speakers: 1\n  out_dims: 128\n"
            "  cn_layers: 2\n  cn_chans: 32\n  n_hidden: 32\n"
        )
        from usvc.infer.engine import InferenceEngine
        with pytest.raises(ValueError, match="only supported for the"):
            InferenceEngine(cfgp, ckpt, device="cuda",
                            diffusion_ckpt="d.ckpt", diffusion_config="d.yaml")

    def test_convert_shallow_diffusion_end_to_end(self, tmp_path, monkeypatch):
        """sovits + 真实 enhancer (dummy vocoder): 形状正确、有限值。"""
        from usvc.infer.engine import InferenceEngine
        from usvc.infer.features import synth_test_wav

        monkeypatch.setattr("usvc.infer.diffusion_enhancer.NSFHiFiGAN", _DummyVocoder)
        ckpt = tmp_path / "sovits.ckpt"
        cfg = tmp_path / "sovits.yaml"
        _sovits_module_ckpt(ckpt)
        _sovits_cfg(cfg)
        dcfg = tmp_path / "diff.yaml"
        _diffusion_cfg(dcfg, n_units=768, mel=_ENGINE_MEL)
        dckpt = tmp_path / "diff.ckpt"
        _diffusion_ckpt(dckpt, dcfg)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=44100, duration=0.5, f0_base=220.0)

        monkeypatch.setattr(_DummyVocoder, "hop", 512)
        engine = InferenceEngine(cfg, ckpt, device="cuda",
                                 diffusion_ckpt=dckpt, diffusion_config=dcfg)
        # 普通路径
        out_plain = engine.convert(str(wav_in), spk=0)
        # 浅扩散路径
        out_shd = engine.convert(str(wav_in), spk=0, shallow_diffusion=True, k_step=20)
        assert not np.isnan(out_shd).any()
        assert np.isfinite(out_shd).all()
        assert out_shd.shape == out_plain.shape
        assert 0.4 < out_shd.shape[0] / 44100 < 0.7

    def test_convert_short_audio_shallow_diffusion(self, tmp_path, monkeypatch):
        """短音频 (< max_chunk) 也走浅扩散 (T <= max_chunk_frames 分支)。"""
        from usvc.infer.engine import InferenceEngine
        from usvc.infer.features import synth_test_wav

        monkeypatch.setattr("usvc.infer.diffusion_enhancer.NSFHiFiGAN", _DummyVocoder)
        ckpt = tmp_path / "sovits.ckpt"
        cfg = tmp_path / "sovits.yaml"
        _sovits_module_ckpt(ckpt)
        _sovits_cfg(cfg)
        dcfg = tmp_path / "diff.yaml"
        _diffusion_cfg(dcfg, n_units=768, mel=_ENGINE_MEL)
        dckpt = tmp_path / "diff.ckpt"
        _diffusion_ckpt(dckpt, dcfg)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=44100, duration=1.5, f0_base=220.0)

        monkeypatch.setattr(_DummyVocoder, "hop", 512)
        engine = InferenceEngine(cfg, ckpt, device="cuda",
                                 diffusion_ckpt=dckpt, diffusion_config=dcfg)
        out = engine.convert(str(wav_in), spk=0, shallow_diffusion=True, k_step=20)
        assert np.isfinite(out).all()
        assert 1.2 < out.shape[0] / 44100 < 1.8
