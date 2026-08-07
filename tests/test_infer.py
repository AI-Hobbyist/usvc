"""Inference tests: shared vocoder, sovits/rvc infer, engine end-to-end."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CUDA = torch.cuda.is_available()
VOCODER_DIR = ROOT / "pretrained" / "nsf_hifigan"
HAVE_VOCODER = (VOCODER_DIR / "config.json").exists() and (VOCODER_DIR / "model.ckpt").exists()

# openvpi PC-NSF-HiFiGAN 2025.02 (mini_nsf=True) ckpt, optional in CI.
PC_VOCODER_CKPT = Path("/tmp/vocoder_dl/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt")
HAVE_PC_VOCODER = PC_VOCODER_CKPT.exists()

_PC_VOCODER_CONFIG = {
    "resblock": "1", "num_mels": 128, "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4, 4], "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "sampling_rate": 44100, "hop_size": 512, "n_fft": 2048, "win_size": 2048,
    "fmin": 40, "fmax": 16000,
}


def _make_module_ckpt(backend: str, opts: dict, path: Path):
    from usvc.backends import get_backend
    from usvc.train.config import TrainConfig

    cfg = TrainConfig(backend=backend)
    cfg.backend_opts = opts
    module = get_backend(backend)().build_module(cfg)
    torch.save({"state_dict": module.state_dict(), "epoch": 0, "global_step": 0}, path)
    return path


def _write_cfg(backend: str, opts: dict, path: Path):
    lines = [f"backend: {backend}", "dataset_dir: /tmp", "exp_dir: /tmp/exp",
             "backend_opts:"]
    for k, v in opts.items():
        lines.append(f"  {k}: {v}")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.skipif(not HAVE_VOCODER, reason="NSF-HiFiGAN weights missing")
class TestVocoder:
    def test_vocoder_loads_and_runs(self):
        from usvc.nn.vocoder import NSFHiFiGAN

        voc = NSFHiFiGAN(VOCODER_DIR / "config.json", VOCODER_DIR / "model.ckpt",
                         device="cuda")
        mel = torch.randn(1, 100, 128, device="cuda")
        f0 = (torch.randn(1, 100, device="cuda").abs() * 200 + 100)
        wav = voc.infer(mel, f0)
        assert wav.dim() == 3 and wav.shape[0] == 1 and wav.shape[1] == 1
        assert wav.shape[2] == 100 * 512
        assert not torch.isnan(wav).any()

    def test_vocoder_matches_ddsp_reference(self):
        """usvc vocoder == DDSP-SVC-6.3 nsf_hifigan Generator (bitwise).

        Only valid when pretrained/nsf_hifigan/model.ckpt is the legacy
        choihkk-format ckpt (top-level 'generator' key, mini_nsf=False).
        The deployed openvpi PC ckpt (state_dict/generator.*, mini_nsf=True)
        is a different architecture and is covered by the dedicated
        test_vocoder_pc_matches_openvpi_original instead.
        """
        import torch as _t
        try:
            _ck = _t.load(str(VOCODER_DIR / "model.ckpt"), map_location="cpu", weights_only=False)
        except Exception:
            pytest.skip("model.ckpt not loadable")
        if not (isinstance(_ck, dict) and "generator" in _ck):
            pytest.skip("legacy choihkk-format ckpt not present (PC ckpt deployed)")
        import sys as _sys

        _sys.path.insert(0, "/root/DDSP-SVC-6.3")
        from nsf_hifigan.models import load_model as ddsp_load

        from usvc.nn.vocoder import NSFHiFiGAN

        voc = NSFHiFiGAN(VOCODER_DIR / "config.json", VOCODER_DIR / "model.ckpt",
                         device="cuda")
        ddsp_gen, _ = ddsp_load(str(VOCODER_DIR / "model.ckpt"), device="cuda")
        torch.manual_seed(42)
        c = torch.randn(1, 128, 100, device="cuda")
        f0 = (torch.randn(1, 100, device="cuda").abs() * 200 + 100)
        torch.manual_seed(0)
        w1 = voc.generator(c, f0)
        torch.manual_seed(0)
        w2 = ddsp_gen(c, f0)
        assert (w1 - w2).abs().max().item() < 1e-4

    def test_vocoder_half(self):
        from usvc.nn.vocoder import NSFHiFiGAN

        voc = NSFHiFiGAN(VOCODER_DIR / "config.json", VOCODER_DIR / "model.ckpt",
                         device="cuda", is_half=True)
        mel = torch.randn(1, 50, 128, device="cuda", dtype=torch.float16)
        f0 = (torch.randn(1, 50, device="cuda", dtype=torch.float16).abs() * 200 + 100)
        wav = voc.infer(mel, f0)
        assert wav.dtype == torch.float16 and wav.shape[2] == 50 * 512

    @pytest.mark.skipif(not HAVE_PC_VOCODER, reason="openvpi PC ckpt missing")
    def test_vocoder_pc_mini_nsf_detection_and_run(self):
        """openvpi PC-NSF-HiFiGAN 2025.02: auto-detect mini_nsf=True, infer."""
        from usvc.nn.vocoder import NSFHiFiGAN

        voc = NSFHiFiGAN(_PC_VOCODER_CONFIG, PC_VOCODER_CKPT, device="cuda")
        assert voc.h.mini_nsf is True
        mel = torch.randn(1, 100, 128, device="cuda")
        f0 = (torch.randn(1, 100, device="cuda").abs() * 200 + 100)
        wav = voc.infer(mel, f0)
        assert wav.shape == (1, 1, 100 * 512)
        assert not torch.isnan(wav).any()

    @pytest.mark.skipif(not HAVE_PC_VOCODER, reason="openvpi PC ckpt missing")
    def test_vocoder_pc_half(self):
        from usvc.nn.vocoder import NSFHiFiGAN

        voc = NSFHiFiGAN(_PC_VOCODER_CONFIG, PC_VOCODER_CKPT, device="cuda",
                         is_half=True)
        assert voc.h.mini_nsf is True
        mel = torch.randn(1, 100, 128, device="cuda", dtype=torch.float16)
        f0 = torch.rand(1, 100, device="cuda", dtype=torch.float16) * 400 + 100
        wav = voc.infer(mel, f0)
        assert wav.dtype == torch.float16 and wav.shape == (1, 1, 100 * 512)
        assert not torch.isnan(wav).any()

    @pytest.mark.skipif(not HAVE_PC_VOCODER, reason="openvpi PC ckpt missing")
    def test_vocoder_pc_matches_openvpi_original(self):
        """usvc Generator (mini_nsf) == openvpi SingingVocoders original."""
        import sys as _sys

        _sys.path.insert(0, "/tmp")
        import pc_models

        from usvc.nn.vocoder import nsf_hifigan as uh

        h = uh.AttrDict({**_PC_VOCODER_CONFIG, "mini_nsf": True, "noise_sigma": 0.0})
        torch.manual_seed(42)
        g_usvc = uh.Generator(h).cuda().eval()
        torch.manual_seed(42)
        g_ref = pc_models.Generator(h).cuda().eval()
        ck = torch.load(PC_VOCODER_CKPT, map_location="cpu", weights_only=False)
        sd = {k.replace("generator.", ""): v for k, v in ck["state_dict"].items()
              if k.startswith("generator.")}
        m1, u1 = g_usvc.load_state_dict(sd, strict=False)
        m2, u2 = g_ref.load_state_dict(sd, strict=False)
        assert not m1 and not u1 and not m2 and not u2
        g_usvc.remove_weight_norm()
        g_ref.remove_weight_norm()
        for seed in (0, 42, 1234):
            torch.manual_seed(seed)
            c = torch.randn(1, 128, 200, device="cuda")
            f0 = torch.rand(1, 200, device="cuda") * 400 + 100
            with torch.no_grad():
                w1 = g_usvc(c, f0)
                w2 = g_ref(c, f0)
            assert (w1 - w2).abs().max().item() < 1e-5, f"seed {seed}"


@pytest.mark.skipif(not CUDA, reason="CUDA required")
class TestSovitsRVCInfer:
    def test_sovits_infer_runs(self):
        from usvc.backends.sovits.model import SoVITSModel

        torch.manual_seed(42)
        m = SoVITSModel(n_units=768, spec_channels=1025, segment_size=20,
                        inter_channels=192, hidden_channels=192, filter_channels=768,
                        n_heads=2, n_layers=6, gin_channels=768, n_speakers=200,
                        sampling_rate=44100).cuda()
        B, T = 1, 100
        c = torch.randn(B, 768, T).cuda()
        f0 = (torch.rand(B, T).cuda() * 300 + 100)
        uv = (f0 > 0).float()
        g = torch.zeros(B, 1, dtype=torch.long).cuda()
        out, f0_out = m.infer(c, f0, uv, g)
        assert out.shape == (B, 1, T * 512)
        assert not torch.isnan(out).any()

    def test_sovits_infer_matches_original(self):
        """usvc SoVITSModel.infer == so-vits NetGSvc.infer (bitwise)."""
        import sys as _sys

        _sys.path.insert(0, "/root/so-vits-svc")
        import models as so_models

        from usvc.backends.sovits.model import SoVITSModel

        kwargs = dict(
            spec_channels=1025, segment_size=20, inter_channels=192,
            hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
            kernel_size=3, p_dropout=0.1, resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2, 2], upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4, 4], gin_channels=768,
            ssl_dim=768, n_speakers=200, sampling_rate=44100,
        )
        torch.manual_seed(42)
        orig = so_models.SynthesizerTrn(**kwargs).cuda()
        torch.manual_seed(42)
        usvc = SoVITSModel(n_units=768, **{k: v for k, v in kwargs.items() if k != "ssl_dim"}).cuda()
        missing, unexpected = usvc.load_state_dict(orig.state_dict(), strict=False)
        assert not missing and not unexpected

        for seed in [0, 123]:
            B, T = 1, 120
            torch.manual_seed(seed + 1000)
            c = torch.randn(B, 768, T).cuda()
            f0 = (torch.rand(B, T).cuda() * 300 + 100)
            uv = (f0 > 0).float()
            g = torch.zeros(B, 1, dtype=torch.long).cuda()
            torch.manual_seed(seed)
            o1, _ = orig.infer(c, f0, uv, g=g)
            torch.manual_seed(seed)
            o2, _ = usvc.infer(c, f0, uv, g=g)
            assert (o1 - o2).abs().max().item() < 1e-5

    def test_rvc_infer_runs_and_matches(self):
        import os
        if not os.path.isdir("/tmp/rvc_src"):
            pytest.skip("/tmp/rvc_src (original RVC source) not present")
        import sys as _sys

        _sys.path.insert(0, "/tmp/rvc_src")
        from infer.module.models import SynthesizerTrnMs768NSFsid as OrigRVC

        from usvc.backends.rvc.model import RVCModel

        kwargs = dict(
            spec_channels=1025, segment_size=20, inter_channels=192,
            hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
            kernel_size=3, p_dropout=0.1, resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[12, 10, 2, 2], upsample_initial_channel=512,
            upsample_kernel_sizes=[24, 20, 4, 4], spk_embed_dim=200,
            gin_channels=256, sr=48000,
        )
        torch.manual_seed(42)
        orig = OrigRVC(**kwargs, is_half=False).cuda()
        torch.manual_seed(42)
        usvc = RVCModel(n_units=768, **kwargs).cuda()
        missing, unexpected = usvc.load_state_dict(orig.state_dict(), strict=False)
        assert not missing and not unexpected

        B, T = 1, 120
        torch.manual_seed(7)
        phone = torch.randn(B, T, 768).cuda()
        pitch = torch.randint(1, 255, (B, T)).cuda()
        nsff0 = (torch.rand(B, T).cuda() * 300 + 100)
        lengths = torch.tensor([T], device="cuda")
        sid = torch.zeros(B, dtype=torch.long).cuda()
        torch.manual_seed(0)
        o1, *_ = orig.infer(phone, lengths, pitch, nsff0, sid)
        torch.manual_seed(0)
        o2, *_ = usvc.infer(phone, lengths, pitch, nsff0, sid)
        assert (o1 - o2).abs().max().item() < 1e-5
        assert o1.shape == (B, 1, T * 480)

    def test_rvc_f0_to_coarse(self):
        from usvc.backends.rvc.model import RVCModel

        f0 = torch.tensor([[0.0, 100.0, 500.0, 1100.0]])
        coarse = RVCModel.f0_to_coarse(f0)
        assert coarse.shape == (1, 4)
        assert coarse[0, 0] == 1  # unvoiced -> 1
        assert coarse[0, -1] == 255  # max clamp


@pytest.mark.skipif(not CUDA, reason="CUDA required")
class TestEngine:
    @pytest.mark.parametrize("backend,opts,sr", [
        ("rift_dit", {"dim": 256, "depth": 4, "head_dim": 64, "cvec_dim": 768,
                      "n_mel_channels": 128, "num_speaker": 1, "eval_sample_steps": 20}, 44100),
        ("ddsp6", {"n_units": 768, "num_speaker": 1, "n_aux_layers": 2,
                   "n_aux_chans": 128, "n_layers": 2, "n_chans": 128, "out_dims": 128}, 44100),
        ("diffusion", {"n_units": 768, "num_speaker": 1, "out_dims": 128,
                       "n_hidden": 128, "cn_layers": 2, "cn_chans": 128,
                       "wn_layers": 2, "wn_chans": 128, "k_step": 1000}, 44100),
        ("reflow_vae", {"n_units": 768, "num_speaker": 1, "out_dims": 128,
                        "n_layers": 2, "n_chans": 128, "n_hidden": 128,
                        "back_bone": "wavenet", "use_attention": False,
                        "win_length": 2048, "sampling_rate": 44100,
                        "block_size": 512}, 44100),
    ])
    def test_engine_mel_backend(self, backend, opts, sr, tmp_path):
        if backend in ("rift_dit", "ddsp6", "diffusion", "reflow_vae") and not HAVE_VOCODER:
            pytest.skip("NSF-HiFiGAN weights missing")
        from usvc.infer.engine import InferenceEngine
        from usvc.infer.features import synth_test_wav

        ckpt = tmp_path / f"{backend}.ckpt"
        cfg = tmp_path / f"{backend}.yaml"
        _make_module_ckpt(backend, opts, ckpt)
        _write_cfg(backend, opts, cfg)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=sr, duration=1.0, f0_base=220.0)

        engine = InferenceEngine(cfg, ckpt, device="cuda")
        out = engine.convert(str(wav_in), spk=0, trans=0)
        assert out.dtype == np.float32 and out.ndim == 1
        assert not np.isnan(out).any()
        dur = out.shape[0] / sr
        assert 0.7 < dur < 1.4, f"duration {dur}"

    def test_engine_sovits(self, tmp_path):
        from usvc.infer.engine import InferenceEngine
        from usvc.infer.features import synth_test_wav

        backend, sr = "sovits", 44100
        opts = {"n_units": 768, "spec_channels": 1025, "num_speaker": 1,
                "gin_channels": 768, "segment_size": 20, "inter_channels": 64,
                "hidden_channels": 64, "filter_channels": 128,
                "units_kind": "contentvec"}
        ckpt = tmp_path / "sovits.ckpt"
        cfg = tmp_path / "sovits.yaml"
        _make_module_ckpt(backend, opts, ckpt)
        _write_cfg(backend, opts, cfg)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=sr, duration=1.0, f0_base=220.0)
        engine = InferenceEngine(cfg, ckpt, device="cuda")
        out = engine.convert(str(wav_in), spk=0, trans=0)
        assert not np.isnan(out).any()
        assert 0.7 < out.shape[0] / sr < 1.4

    def test_engine_rvc(self, tmp_path):
        from usvc.infer.engine import InferenceEngine
        from usvc.infer.features import synth_test_wav

        backend, sr = "rvc", 48000
        opts = {"n_units": 768, "num_speaker": 1, "sr": 48000, "segment_size": 20,
                "inter_channels": 64, "hidden_channels": 64, "filter_channels": 128}
        ckpt = tmp_path / "rvc.ckpt"
        cfg = tmp_path / "rvc.yaml"
        _make_module_ckpt(backend, opts, ckpt)
        _write_cfg(backend, opts, cfg)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=sr, duration=1.0, f0_base=220.0)
        engine = InferenceEngine(cfg, ckpt, device="cuda")
        out = engine.convert(str(wav_in), spk=0, trans=0)
        assert not np.isnan(out).any()
        assert 0.7 < out.shape[0] / sr < 1.4

    def test_engine_rvc_original_weights_format(self, tmp_path):
        """原版 RVC 权重格式 (顶层 model 键 + 裸 generator key) 能被 engine 加载.

        同时验证: emb_g shape 不匹配 (num_speaker 不同) 时跳过而非报错,
        enc_q 缺失 (发布版剔除) 时容忍.
        """
        import torch
        from usvc.backends import get_backend
        from usvc.train.config import TrainConfig
        from usvc.infer.engine import InferenceEngine
        from usvc.infer.features import synth_test_wav

        backend, sr = "rvc", 48000
        opts = {"n_units": 768, "num_speaker": 1, "sr": 48000, "segment_size": 20,
                "inter_channels": 64, "hidden_channels": 64, "filter_channels": 128}
        # 构建 module 拿到 generator state_dict, 剥离 generator. 前缀模拟原版格式
        cfg = TrainConfig(backend=backend)
        cfg.backend_opts = opts
        module = get_backend(backend)().build_module(cfg)
        gen_sd = module.generator.state_dict()
        bare = {k.removeprefix("generator."): v for k, v in gen_sd.items()}
        # emb_g 用 109 说话人 (shape 不匹配 num_speaker=1)
        bare["emb_g.weight"] = torch.randn(109, 256)
        # 剔除 enc_q (原版发布版做法)
        bare = {k: v for k, v in bare.items() if not k.startswith("enc_q.")}
        torch.save({"model": bare, "iteration": 0, "learning_rate": 1e-4},
                   tmp_path / "orig_rvc.pth")

        cfg_path = tmp_path / "rvc_orig.yaml"
        _write_cfg(backend, opts, cfg_path)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=sr, duration=1.0, f0_base=220.0)
        engine = InferenceEngine(cfg_path, tmp_path / "orig_rvc.pth", device="cuda")
        out = engine.convert(str(wav_in), spk=0, trans=0)
        assert not np.isnan(out).any()
        assert 0.7 < out.shape[0] / sr < 1.4

        # 恢复 enc_q 后也应能加载 (完整版原版权重)
        bare2 = {k.removeprefix("generator."): v for k, v in gen_sd.items()}
        bare2["emb_g.weight"] = torch.randn(109, 256)
        torch.save({"model": bare2}, tmp_path / "orig_rvc_full.pth")
        engine2 = InferenceEngine(cfg_path, tmp_path / "orig_rvc_full.pth", device="cuda")
        assert engine2.module.generator is not None


class TestLoudnessNorm:
    """LUFS loudness normalization (RIFT-SVC style)."""

    def test_helpers_normalize_restore(self):
        import numpy as np
        from usvc.infer.engine import (
            _lufs_meter, _normalize_loudness, _restore_loudness,
        )

        sr = 44100
        t = np.arange(sr) / sr
        wav = (0.05 * np.sin(2 * np.pi * 440 * t)
               + 0.02 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        meter = _lufs_meter(sr)
        l0 = meter.integrated_loudness(wav)

        # normalize to -18 LUFS, capture original
        wav_n, orig = _normalize_loudness(wav, sr, -18.0)
        assert wav_n.dtype == np.float32
        assert abs(orig - l0) < 1e-3
        assert abs(meter.integrated_loudness(wav_n) - (-18.0)) < 0.3
        assert float(np.max(np.abs(wav_n))) <= 1.0  # peak protection

        # restore back to original loudness
        wav_r = _restore_loudness(wav_n, sr, orig)
        assert abs(meter.integrated_loudness(wav_r) - l0) < 0.5
        assert float(np.max(np.abs(wav_r))) <= 1.0

    def test_helpers_silence_and_noise_floor(self):
        import numpy as np
        from usvc.infer.engine import _normalize_loudness

        sr = 44100
        # digital silence: -inf loudness must not blow up
        sil = np.zeros(sr, dtype=np.float32)
        sil_n, orig_sil = _normalize_loudness(sil, sr, -18.0)
        assert not np.isfinite(orig_sil)
        assert not sil_n.any()
        # near-silence noise floor: skipped (returned unchanged)
        noise = (1e-6 * np.random.randn(sr)).astype(np.float32)
        noise_n, _ = _normalize_loudness(noise, sr, -18.0)
        assert np.allclose(noise_n, noise)

    def test_cli_args_present(self):
        import contextlib
        import io
        from usvc.cli import infer as infer_cli

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                infer_cli.main(["--help"])
            except SystemExit as e:
                assert e.code == 0
        help_text = buf.getvalue()
        assert "--target-loudness" in help_text
        assert "--no-restore-loudness" in help_text

    def test_engine_convert_accepts_target_loudness(self, tmp_path):
        """convert() accepts target_loudness/restore_loudness and output stays
        finite; when the restore peak-protection is not triggered, the output
        loudness is restored to the input's original loudness."""
        import numpy as np
        from usvc.infer.engine import InferenceEngine, _lufs_meter
        from usvc.infer.features import synth_test_wav
        from usvc.io.audio import load_audio

        backend, sr = "rvc", 48000
        opts = {"n_units": 768, "num_speaker": 1, "sr": 48000, "segment_size": 20,
                "inter_channels": 64, "hidden_channels": 64, "filter_channels": 128}
        ckpt = tmp_path / "rvc.ckpt"
        cfg = tmp_path / "rvc.yaml"
        _make_module_ckpt(backend, opts, ckpt)
        _write_cfg(backend, opts, cfg)
        wav_in = tmp_path / "in.wav"
        synth_test_wav(wav_in, sr=sr, duration=1.0, f0_base=220.0)

        engine = InferenceEngine(cfg, ckpt, device="cuda")
        # baseline without normalization
        out_base = engine.convert(str(wav_in), spk=0, trans=0)
        # with loudness normalization (restore on by default)
        out_norm = engine.convert(str(wav_in), spk=0, trans=0, target_loudness=-18.0)
        # with restore disabled
        out_norestore = engine.convert(
            str(wav_in), spk=0, trans=0, target_loudness=-18.0, restore_loudness=False
        )
        for o in (out_base, out_norm, out_norestore):
            assert o.dtype == np.float32 and o.ndim == 1
            assert not np.isnan(o).any()
            assert 0.7 < o.shape[0] / sr < 1.4
        # restore on -> output loudness matches the input's original loudness
        # (unless the 0.99-peak protection fired, which only happens with
        # un-trained random weights whose output peaks far exceed 1.0)
        meter = _lufs_meter(sr)
        l_in = meter.integrated_loudness(load_audio(str(wav_in), sr=sr))
        l_norm = meter.integrated_loudness(out_norm)
        if float(np.abs(out_norm).max()) <= 1.0:
            assert abs(l_norm - l_in) < 1.5, f"restore target {l_in:.2f}, got {l_norm:.2f}"
        else:
            # peak protection fired: loudness is lower than the target
            assert l_norm < l_in + 0.5
