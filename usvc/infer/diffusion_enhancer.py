"""Shallow-diffusion enhancer (so-vits -shd): sovits wav -> mel -> k-step DDPM denoise -> vocoder wav.

数据流与 so-vits-svc 4.1 infer_tool 一致: gt_spec + units/f0/vol/spk -> q_sample 加噪到
k_step-1 -> 反向采样 (fp32, DDPM/PLMS 数值敏感, half 易 NaN) -> NSF-HiFiGAN。
"""
from __future__ import annotations

import torch
import yaml

from ..backends.diffusion.model import DiffusionModel
from ..nn.vocoder import NSFHiFiGAN
from ._enh_common import mel_ext, pack, reextract, tpad, units_ext, vocoder, vol_batch

__all__ = ["DiffusionEnhancer"]


class DiffusionEnhancer:
    def __init__(self, diffusion_ckpt, diffusion_config, vocoder_dir, device="cuda",
                 half=False, sr=44100, hop=512, compile_sample=False):
        self.device, self.sr, self.hop, self.half = device, sr, hop, half
        cfg = yaml.safe_load(open(diffusion_config)) or {}
        opts = cfg.get("backend_opts", cfg)  # 扩散参数在 backend_opts (usvc 格式) 或顶层
        mel_cfg = cfg.get("mel") or {}
        g = lambda k, d: opts.get(k, cfg.get(k, d))  # backend_opts -> 顶层 -> 默认
        self.diffusion_n_units = int(g("n_units", 768))
        self.diffusion_units_kind = g("units_kind", "contentvec")
        self.diffusion_model = DiffusionModel(
            n_units=int(g("n_units", 768)), n_spk=int(g("n_spk", 1)),
            out_dims=int(g("out_dims", 128)), n_hidden=int(g("n_hidden", 256)),
            use_pitch_aug=bool(g("use_pitch_aug", False)),
            timesteps=int(g("timesteps", 1000)), max_beta=float(g("max_beta", 0.02)),
            spec_min=float(g("spec_min", -12)), spec_max=float(g("spec_max", 2)),
            cn_layers=int(g("cn_layers", 20)), cn_chans=int(g("cn_chans", 384)),
            mlp_factor=int(g("mlp_factor", 4)), gradient_checkpointing=False,
            k_step=int(g("k_step", 1000)), denoiser=str(g("denoiser", "convnext")),
            wn_layers=int(g("wn_layers", 20)), wn_chans=int(g("wn_chans", 384)))
        self._load_diffusion_weights(diffusion_ckpt)
        self.diffusion_model.eval().to(device)
        from ..backends.diffusion.wavenet import WaveNet as _WN
        for _mod in self.diffusion_model.decoder.denoise_fn.modules():
            if isinstance(_mod, _WN):
                _mod._use_cl = True  # wavenet denoiser channels_last (训练验证等价)
        self.mel_extractor = mel_ext(mel_cfg, sr, hop, device,
                                     n_mels=cfg.get("out_dims", 128), fmax=None)
        self.units_extractor = units_ext(
            self.diffusion_units_kind,
            cfg.get("units_ckpt", "pretrained/contentvec/pytorch_model.bin"),
            sr, hop, device) if self.diffusion_units_kind != "hubertsoft" else None
        self.vocoder = vocoder(NSFHiFiGAN, vocoder_dir, device, compile_gen=compile_sample)
        if compile_sample:
            # 整条采样循环进 CUDA graph (launch-bound 主瓶颈, 实测 -8ms)
            self.diffusion_model.sample = torch.compile(
                self.diffusion_model.sample, mode="reduce-overhead", dynamic=False)

    def _load_diffusion_weights(self, ckpt_path):
        from ..io.ckpt import load_ckpt_state
        sd = load_ckpt_state(str(ckpt_path))
        if all(k.startswith("model.") for k in sd):
            sd = {k[6:]: v for k, v in sd.items()}
        elif all(k.startswith("module.") for k in sd):
            sd = {k[7:]: v for k, v in sd.items()}
        missing, unexpected = self.diffusion_model.load_state_dict(sd, strict=False)
        # strict=False: 容忍非持久 buffer (sqrt_alphas_cumprod) + 优化器残留
        bad = [k for k in unexpected
               if not k.startswith(("optimizer", "scheduler", "trainer", "callbacks", "hyper_parameters"))]
        if bad:
            raise RuntimeError(f"diffusion checkpoint keys don't match model: {bad[:8]}")
        real = [k for k in missing
                if not k.startswith(("sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"))]
        if real:
            raise RuntimeError(f"diffusion checkpoint missing keys: {real[:8]}")

    @torch.no_grad()
    def enhance(self, units, f0, rms, spk_id, sovits_wav, k_step=100,
                infer_speedup=10, method="ddim"):
        T = units.shape[0]
        u2 = reextract(self.units_extractor, sovits_wav, T, self.sr, self.device)
        if u2 is not None:
            units = u2
        mel = tpad(self.mel_extractor.extract(sovits_wav), T).unsqueeze(0).to(self.device)  # [1,T,M] dB
        units_b, f0_b, vol_b, spk_b = pack(units, f0, vol_batch(rms, mel, self.device),
                                           spk_id, self.device)
        _ctx = (torch.autocast("cuda", dtype=torch.float16)
                if self.half else torch.autocast("cuda", enabled=False))
        with _ctx:
            denoised = self.diffusion_model.sample(
                units_b, f0_b, vol_b, spk_b, gt_spec=mel, k_step=k_step,
                infer_speedup=infer_speedup, method=method)  # [1,T,M] denormalized dB
        with torch.autocast("cuda", dtype=torch.float16):  # conv 跑 tensor core
            wav = self.vocoder.infer(denoised.float(), f0_b)
        return wav[0, 0].float().cpu().numpy()