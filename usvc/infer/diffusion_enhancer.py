"""Shallow diffusion enhancer for sovits (so-vits-svc ``-shd`` 浅扩散).

数据流 (与 so-vits-svc 4.1 inference/infer_tool.py 一致):
    sovits wav ──MelExtractor──▶ gt_spec (log-dB 域 mel)
    gt_spec + units/f0/rms/spk ──DiffusionModel.sample(k_step)──▶ 去噪后 mel
    去噪后 mel + f0 ──NSF-HiFiGAN──▶ 最终 wav

关键点:
  - DiffusionModel.sample(gt_spec=, k_step=) 内部做 q_sample 部分加噪到
    k_step-1 再反向采样, 输出 denormalized (dB 域) mel [B, T, M]。
  - mel 帧数与 units 帧数对齐 (sovits 输出 wav 长度 = T * hop)。
  - 扩散采样保持 fp32 (数值敏感, 与 diffusion 后端推理一致)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from ..backends.diffusion.model import DiffusionModel
from ..data.feature_config import FeatureConfig
from ..features.mel import MelExtractor
from ..nn.vocoder import NSFHiFiGAN

__all__ = ["DiffusionEnhancer"]


def _load_ckpt_state(ckpt_path: str) -> dict:
    """Load state dict from a torch ckpt (lightning or bare)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        # lightning: {"state_dict": {...}, ...}; so-vits diffusion: {"model": {...}}
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f"cannot interpret checkpoint: {ckpt_path}")
    return sd


class DiffusionEnhancer:
    """Shallow diffusion enhancer for sovits.

    Loads an independent diffusion checkpoint + NSF-HiFiGAN vocoder.
    Given sovits output wav, extracts mel, runs k-step diffusion denoising,
    and resynthesizes via the diffusion vocoder.
    """

    def __init__(
        self,
        diffusion_ckpt: str | Path,
        diffusion_config: str | Path,
        vocoder_dir: str | Path,
        device: str = "cuda",
        half: bool = False,
        sr: int = 44100,
        hop: int = 512,
        compile_sample: bool = False,
    ) -> None:
        self.device = device
        self.sr = sr
        self.hop = hop
        self.half = half
        self.compile_sample = compile_sample

        with open(diffusion_config) as f:
            cfg = yaml.safe_load(f) or {}
        # 扩散模型参数在 backend_opts 下 (usvc train config 格式) 或顶层
        opts = cfg.get("backend_opts", cfg)
        mel_cfg = cfg.get("mel") or {}
        self.diffusion_n_units = int(opts.get("n_units", cfg.get("n_units", 768)))
        self.diffusion_units_kind = opts.get("units_kind", cfg.get("units_kind", "contentvec"))

        # ---- 扩散模型 (与 usvc diffusion 后端同构) ----
        self.diffusion_model = DiffusionModel(
            n_units=int(opts.get("n_units", cfg.get("n_units", 768))),
            n_spk=int(opts.get("n_spk", cfg.get("n_spk", 1))),
            out_dims=int(opts.get("out_dims", cfg.get("out_dims", 128))),
            n_hidden=int(opts.get("n_hidden", cfg.get("n_hidden", 256))),
            use_pitch_aug=bool(opts.get("use_pitch_aug", cfg.get("use_pitch_aug", False))),
            timesteps=int(opts.get("timesteps", cfg.get("timesteps", 1000))),
            max_beta=float(opts.get("max_beta", cfg.get("max_beta", 0.02))),
            spec_min=float(opts.get("spec_min", cfg.get("spec_min", -12))),
            spec_max=float(opts.get("spec_max", cfg.get("spec_max", 2))),
            cn_layers=int(opts.get("cn_layers", cfg.get("cn_layers", 20))),
            cn_chans=int(opts.get("cn_chans", cfg.get("cn_chans", 384))),
            mlp_factor=int(opts.get("mlp_factor", cfg.get("mlp_factor", 4))),
            gradient_checkpointing=False,
            k_step=int(opts.get("k_step", cfg.get("k_step", 1000))),
            denoiser=str(opts.get("denoiser", cfg.get("denoiser", "convnext"))),
            wn_layers=int(opts.get("wn_layers", cfg.get("wn_layers", 20))),
            wn_chans=int(opts.get("wn_chans", cfg.get("wn_chans", 384))),
        )
        self._load_diffusion_weights(diffusion_ckpt)
        self.diffusion_model.eval()
        self.diffusion_model.to(device)
        # 推理加速: WaveNet denoiser 走 channels_last 数据流（训练验证等价）。
        from ..backends.diffusion.wavenet import WaveNet as _WN

        for _mod in self.diffusion_model.decoder.denoise_fn.modules():
            if isinstance(_mod, _WN):
                _mod._use_cl = True
        # 扩散采样保持 fp32 (DDPM/PLMS 迭代数值敏感, half 易 NaN)

        # ---- mel 提取 (必须与扩散训练时的 mel 参数一致) ----
        self.mel_extractor = MelExtractor(
            FeatureConfig(
                "mel",
                {
                    "n_fft": int(mel_cfg.get("n_fft", 2048)),
                    "win_length": int(mel_cfg.get("win_length", 2048)),
                    "n_mels": int(mel_cfg.get("n_mels", cfg.get("out_dims", 128))),
                    "fmin": float(mel_cfg.get("fmin", 40.0)),
                    "fmax": None if mel_cfg.get("fmax") is None else float(mel_cfg["fmax"]),
                    "center": bool(mel_cfg.get("center", False)),
                },
            ),
            sr=int(mel_cfg.get("sampling_rate", sr)),
            hop=int(mel_cfg.get("hop", hop)),
        ).to(device)

        # ---- contentvec units 提取器 (浅扩散可能用不同编码器) ----
        # sovits 主模型用 hubertsoft(256维), 扩散模型可能用 contentvec(768维)。
        # 从 sovits 输出 wav 重新提取扩散模型需要的 units (second_encoding 简化版)。
        self.units_extractor = None
        if self.diffusion_units_kind != "hubertsoft":
            from ..features.units import UnitsExtractor
            units_fc = FeatureConfig(
                "units",
                {
                    "kind": self.diffusion_units_kind,
                    "sr_in": 16000,
                    "hop_native": 320,
                    "align_mode": "gather",
                    "model_ckpt": cfg.get("units_ckpt", "pretrained/contentvec/pytorch_model.bin"),
                },
            )
            self.units_extractor = UnitsExtractor(
                units_fc, sr, hop, device, dtype=torch.float32
            )

        # ---- 声码器 (fp32, 数值安全) ----
        vocoder_dir = Path(vocoder_dir)
        self.vocoder = NSFHiFiGAN(
            vocoder_dir / "config.json",
            vocoder_dir / "model.ckpt",
            device=device,
            is_half=False,  # 浅扩散全程 fp32 (采样数值敏感)
        )
        # 推理加速: channels_last 数据流（conv1d→conv2d(1,k) 等价变换）。
        # 数值等价已验证（SNR ≥ 64dB）；autocast fp16 下消除逐层布局转换。
        if hasattr(self.vocoder, "generator"):  # 测试用 _DummyVocoder 无此属性
            self.vocoder.generator._use_cl = True
        if compile_sample:
            # 整条采样循环进 CUDA graph（launch-bound 主瓶颈），实测 -8ms。
            self.diffusion_model.sample = torch.compile(
                self.diffusion_model.sample, mode="reduce-overhead", dynamic=False
            )

    def _load_diffusion_weights(self, ckpt_path: str | Path) -> None:
        """Load diffusion state dict, stripping ``model.``/``module.`` prefixes."""
        sd = _load_ckpt_state(str(ckpt_path))
        if all(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items()}
        elif all(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        missing, unexpected = self.diffusion_model.load_state_dict(sd, strict=False)
        # strict=False: 容忍非持久 buffer (sqrt_alphas_cumprod 等) 缺失;
        # unexpected 只允许优化器/调度器残留。
        if unexpected:
            tolerable = [k for k in unexpected if k.startswith(("optimizer", "scheduler", "trainer", "callbacks", "hyper_parameters"))]
            bad = [k for k in unexpected if k not in tolerable]
            if bad:
                raise RuntimeError(
                    f"diffusion checkpoint keys don't match model: {bad[:8]}"
                )
        if missing:
            # 非持久 buffer 会缺失, 其余才是真缺
            real_missing = [k for k in missing if not k.startswith(
                ("sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod")
            )]
            if real_missing:
                raise RuntimeError(f"diffusion checkpoint missing keys: {real_missing[:8]}")

    @torch.no_grad()
    def enhance(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        rms: torch.Tensor | None,
        spk_id: int,
        sovits_wav: torch.Tensor | np.ndarray,
        k_step: int = 100,
        infer_speedup: int = 10,
        method: str = "ddim",
    ) -> np.ndarray:
        """sovits_wav [L] float32 (target sr) → enhanced_wav [L] float32.

        Args:
            units: [T, D] content features (device tensor, any dtype).
            f0: [T] continuous f0 (Hz, pitch-shifted already).
            rms: [T] volume or None (fallback computed from mel).
            spk_id: 0-based speaker id (usvc convention).
            sovits_wav: sovits 输出的 wav (长度为 T * hop)。
        """
        T = units.shape[0]

        # 0) 如果扩散模型用不同编码器 (如 contentvec 768维), 从 sovits 输出 wav
        # 重新提取 units (sovits 用 hubertsoft 256维, 维度不匹配会崩)。
        if self.units_extractor is not None:
            import soundfile as sf_resample  # noqa: F401 (not used directly)
            from ..io.audio import load_audio
            wav_np = sovits_wav if isinstance(sovits_wav, np.ndarray) else sovits_wav.cpu().numpy()
            # UnitsExtractor 需要 16k wav, 从目标 sr 重采样
            import librosa
            wav16k = librosa.resample(wav_np.astype(np.float32), orig_sr=self.sr, target_sr=16000)
            wav16k_t = torch.from_numpy(wav16k).float().to(self.device)
            units = self.units_extractor.extract(wav16k_t)  # [T, D]
            units = units[:T]
            if units.shape[0] < T:
                units = torch.nn.functional.pad(units, (0, 0, 0, T - units.shape[0]))

        # 1) sovits 输出 wav → gt_spec (log-dB 域, 与扩散训练一致)
        mel = self.mel_extractor.extract(sovits_wav)  # [T', M] float32 cpu
        mel = mel[:T]
        if mel.shape[0] < T:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, T - mel.shape[0]))
        mel = mel.unsqueeze(0).to(self.device)  # [1, T, M]

        # 2) 条件特征 (fp32)
        units_b = units.float().unsqueeze(0).to(self.device)  # [1, T, D]
        f0_b = f0.float().unsqueeze(0).to(self.device)        # [1, T]
        if rms is not None:
            vol_b = rms.float().unsqueeze(0).to(self.device)  # [1, T]
        else:
            # fallback: 从 mel 能量估计 volume (与 diffusion/module.py 一致)
            vol_b = (mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)
        spk_b = torch.tensor([spk_id], dtype=torch.long, device=self.device)

        # 3) 浅扩散: gt_spec 部分加噪到 k_step-1, 反向采样去噪。
        # half 时 sample 走 fp16（实测无 NaN 且稳定；wavenet conv 走 tensor core）。
        _sample_ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.half else torch.autocast("cuda", enabled=False)
        )
        with _sample_ctx:
            denoised = self.diffusion_model.sample(
                units_b, f0_b, vol_b, spk_b,
                gt_spec=mel,
                k_step=k_step,
                infer_speedup=infer_speedup,
                method=method,
            )  # [1, T, M] denormalized dB

        # 4) 去噪 mel + f0 → 声码器合成 (autocast fp16: conv 跑 tensor core)
        with torch.autocast("cuda", dtype=torch.float16):
            wav = self.vocoder.infer(denoised.float(), f0_b)
        return wav[0, 0].float().cpu().numpy()
