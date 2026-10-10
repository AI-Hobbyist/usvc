"""Shallow reflow enhancer: sovits/rvc wav -> mel -> t_start rectified-flow -> vocoder wav.

两种模式: "shallow" = ReflowShallowModel (条件 = sovits/rvc mel + 嵌入);
"ddsp6" = DDSP6Model (条件 = DDSP 合成 mel, 目标 = 外部 mel)。ODE fp32 数值敏感。
"""
from __future__ import annotations

import torch
import yaml

from ..backends.ddsp6.model import DDSP6Model
from ..nn.vocoder import NSFHiFiGAN
from ._enh_common import mel_ext, pack, reextract, tpad, units_ext, vocoder, vol_batch

__all__ = ["ReflowEnhancer"]


class ReflowEnhancer:
    def __init__(self, reflow_ckpt, reflow_config, vocoder_dir, device="cuda",
                 sr=44100, hop=512):
        self.device = device
        with open(reflow_config, encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}
        self.speaker_mapping = cfg.get("spk")
        self.speaker_capacity = cfg.get("backend_opts", {}).get("num_speaker", cfg.get("model", {}).get("n_spk", 1))
        m = cfg.get("model", cfg)  # DDSP6 usvc 配置的参数在 model 下, 裸配置在顶层
        self.n_units = int(m.get("n_unit", cfg.get("n_unit", 768)))
        win_length = int(m.get("win_length", cfg.get("win_length", 2048)))
        mel_cfg = cfg.get("mel") or {}
        # 增强器固定工作在 44.1k/512 reflow mel 域 (引擎处理 rvc 48k 桥接)
        self.sr, self.hop = 44100, 512
        self._mode = "shallow" if cfg.get("backend") == "reflow_shallow" else "ddsp6"
        if self._mode == "shallow":
            opts = cfg.get("backend_opts", {})
            from ..backends.reflow_shallow.model import ReflowShallowModel
            self.shallow_model = ReflowShallowModel(
                out_dims=int(opts.get("out_dims", 128)),
                n_layers=int(opts.get("n_layers", 6)),
                n_chans=int(opts.get("n_chans", 768)),
                n_units=int(opts.get("n_units", opts.get("n_unit", self.n_units))),
                n_spk=int(opts.get("num_speaker", opts.get("n_spk", 1))),
                n_hidden=int(opts.get("n_hidden", 256)))
            self._load_shallow_weights(reflow_ckpt)
            self.shallow_model.eval().to(device)
            self.ddsp_model = None
        elif cfg.get("backend") == "ddsp6":
            from ..train.config import TrainConfig
            from ..speakers import load_model_speakers
            from ..backends.ddsp6.module import DDSPModule
            train_cfg = TrainConfig.from_yaml(reflow_config)
            self.speaker_mapping, self.speaker_capacity = load_model_speakers(train_cfg, reflow_ckpt)
            self.ddsp_model = DDSPModule(train_cfg).model
            self.n_units = train_cfg.backend_opts.get("n_units", 768)
            self._load_reflow_weights(reflow_ckpt)
            self.ddsp_model.eval().to(device)
            self.shallow_model = None
        else:
            self.ddsp_model = DDSP6Model(
                n_units=self.n_units, n_spk=self.speaker_capacity, win_length=win_length,
                sampling_rate=44100, block_size=512,
                n_aux_layers=int(m.get("n_aux_layers", cfg.get("n_aux_layers", 6))),
                n_aux_chans=int(m.get("n_aux_chans", cfg.get("n_aux_chans", 512))),
                n_layers=int(m.get("n_layers", cfg.get("n_layers", 6))),
                n_chans=int(m.get("n_chans", cfg.get("n_chans", 768))),
                velocity_net=str(m.get("velocity_net", cfg.get("velocity_net", "LYNXNet2"))))
            self._load_reflow_weights(reflow_ckpt)
            self.ddsp_model.eval().to(device)
            self.shallow_model = None
        self.mel_extractor = mel_ext(mel_cfg, 44100, 512, device,
                                     win=win_length, fmax=16000.0)
        self.units_kind = str(cfg.get("units_kind", cfg.get("backend_opts", {}).get("units_kind", "contentvec")))
        self.units_extractor = units_ext(
            self.units_kind,
            cfg.get("units_ckpt", "pretrained/contentvec/pytorch_model.bin"),
            self.sr, self.hop, device) if self.units_kind != "contentvec" else None
        self.vocoder = vocoder(NSFHiFiGAN, vocoder_dir, device)

    def _load_shallow_weights(self, ckpt_path):
        from ..io.ckpt import load_ckpt_state
        sd = load_ckpt_state(ckpt_path)
        if any(k.startswith("model.") for k in sd):  # lightning model. 前缀
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        sd = {k.replace("reflow_velocity_fn.", "reflow_model.velocity_fn.")
              if k.startswith("reflow_velocity_fn.") else k: v for k, v in sd.items()}
        model_sd = self.shallow_model.state_dict()
        dropped = [k for k, v in sd.items()
                   if k in model_sd and v.shape != model_sd[k].shape]
        if dropped:  # DDSP6 base 微调时 dim_cond 不同, shape 不匹配键丢弃
            print(f"[reflow_shallow] dropped {len(dropped)} shape-mismatched "
                  f"key(s) (expected when finetuning from a DDSP6 base with a "
                  f"different dim_cond): {dropped[:5]}")
        sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = self.shallow_model.load_state_dict(sd, strict=False)
        OPT = ("ddsp_model.", "mel_extractor.", "vocoder", "_", "cfg",
               "unit_embed.", "f0_embed.", "volume_embed.", "spk_embed.",
               "aug_shift_embed.")
        real = [k for k in missing
                if not k.startswith(OPT) and k not in set(dropped)]
        if real:
            raise RuntimeError(
                f"[reflow_shallow] ckpt missing {len(real)} required "
                f"weight(s); architecture mismatch? e.g. {real[:5]}")
        if missing:
            print(f"[reflow_shallow] missing optional/new-layer keys: "
                  f"{len(missing)} (e.g. {missing[:3]})")
        bad = [k for k in unexpected
               if not k.startswith(("ddsp_model.", "mel_extractor.", "_", "vocoder", "cfg"))]
        if bad:
            print(f"[reflow_shallow] unexpected keys: {bad[:5]}")

    def _load_reflow_weights(self, ckpt_path):
        from ..io.ckpt import load_ckpt_state
        sd = load_ckpt_state(ckpt_path)
        if any(k.startswith("model.") for k in sd):  # lightning model. 前缀
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, _ = self.ddsp_model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("mel_extractor.")]
        if bad:
            raise RuntimeError(f"reflow ckpt missing required model keys: {bad[:8]}")

    def _shallow_sample(self, units, f0, volume, spk_id, gt_spec, t_start, infer_step, method):
        # 绕过 Unit2Wav.forward: 其 infer 分支在有 mel_extractor 时复用 DDSP mel 当 gt_spec
        ddsp_wav, _ = self.ddsp_model.ddsp_model(
            units, f0, volume, spk_id=spk_id, infer=True)  # [B,L]
        ddsp_mel = self.ddsp_model.mel_extractor.extract(ddsp_wav)  # [B,T,M]
        T_gt = gt_spec.shape[1]
        ddsp_mel = ddsp_mel[:, :T_gt]
        if ddsp_mel.shape[1] < T_gt:
            ddsp_mel = torch.nn.functional.pad(ddsp_mel, (0, 0, 0, T_gt - ddsp_mel.shape[1]))
        return self.ddsp_model.reflow_model(
            condition=ddsp_mel.to(gt_spec.device), gt_spec=gt_spec,
            infer=True, infer_step=infer_step, method=method,
            t_start=t_start, use_tqdm=False)  # [1,T,M] dB

    @torch.no_grad()
    def enhance(self, units, f0, rms, spk_id, gen_wav, t_start=0.6,
                infer_step=20, method="euler"):
        T = units.shape[0]
        u2 = reextract(self.units_extractor, gen_wav, T, self.sr, self.device)
        if u2 is not None:
            units = u2
        mel = tpad(self.mel_extractor.extract(gen_wav), T).unsqueeze(0).to(self.device)  # [1,T,M] dB
        units_b, f0_b, vol_b, spk_b = pack(units, f0, vol_batch(rms, mel, self.device),
                                           spk_id, self.device)
        if self._mode == "shallow":
            denoised = self.shallow_model.sample(
                units_b, f0_b, vol_b, mel, t_start=t_start, infer_step=infer_step,
                method=method, spk_id=spk_b)  # [1,T,M] denormalized dB
        else:
            denoised = self._shallow_sample(units_b, f0_b.unsqueeze(-1),
                                            vol_b.unsqueeze(-1), spk_b, mel,
                                            t_start=t_start, infer_step=infer_step,
                                            method=method)
        with torch.autocast("cuda", dtype=torch.float16):  # conv 跑 tensor core
            wav = self.vocoder.infer(denoised.float(), f0_b)
        return wav[0, 0].float().cpu().numpy()
