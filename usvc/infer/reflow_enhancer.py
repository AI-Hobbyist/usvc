"""Reflow shallow-diffusion enhancer for sovits/rvc.

Two modes:
  - "shallow" (ReflowShallowModel): condition = sovits/rvc mel + units/f0/vol
    embeddings; the velocity network applies a targeted "dirty → clean"
    correction.
  - "ddsp6" (DDSP6Model): DDSP-synthesised mel as condition + reflow.

Data flow: rvc/sovits wav → mel → reflow(t_start) → refined mel →
NSF-HiFiGAN vocoder → final wav.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from ..backends.ddsp6.model import DDSP6Model
from ..data.feature_config import FeatureConfig
from ..features.mel import MelExtractor
from ..nn.vocoder import NSFHiFiGAN

__all__ = ["ReflowEnhancer"]


class ReflowEnhancer:
    """Reflow shallow-diffusion enhancer (DDSP6 backend).

    Loads a DDSP6 checkpoint (``ddsp_model`` + ``reflow_model``) and a shared
    NSF-HiFiGAN vocoder.  Given a wav-domain generator output (sovits/rvc),
    extracts mel, builds the DDSP6 condition via route B, runs t_start reflow
    shallow diffusion, and resynthesises via the PC vocoder.
    """

    def __init__(
        self,
        reflow_ckpt: str | Path,
        reflow_config: str | Path,
        vocoder_dir: str | Path,
        device: str = "cuda",
        sr: int = 44100,
        hop: int = 512,
    ) -> None:
        self.device = device
        # The enhancer always operates in the reflow mel domain (44.1k/512).
        # self.sr/self.hop are seeded from constructor args then overridden by
        # the model config below; they are used by enhance() for resampling.
        self.sr = sr
        self.hop = hop

        # ---- DDSP6 model config ----
        with open(reflow_config) as f:
            cfg = yaml.safe_load(f) or {}
        m = cfg.get("model", cfg)
        self.n_units = int(m.get("n_unit", cfg.get("n_unit", 768)))
        n_layers = int(m.get("n_layers", cfg.get("n_layers", 6)))
        n_chans = int(m.get("n_chans", cfg.get("n_chans", 768)))
        n_aux_layers = int(m.get("n_aux_layers", cfg.get("n_aux_layers", 6)))
        n_aux_chans = int(m.get("n_aux_chans", cfg.get("n_aux_chans", 512)))
        velocity_net = str(m.get("velocity_net", cfg.get("velocity_net", "LYNXNet2")))
        # ---- detect enhancer mode from the config backend field ----
        # "reflow_shallow" = standalone ReflowShallowModel (no DDSP, no timbre
        #   leakage; condition = sovits/rvc output mel).
        # otherwise (ddsp6 configs) = DDSP6Model with ddsp_model + reflow.
        self._mode = "shallow" if cfg.get("backend") == "reflow_shallow" else "ddsp6"

        # The reflow model always operates in the 44.1k/512 mel domain
        # (matching the bundled nsf-hifigan vocoder).  We do NOT inherit the
        # backend sr/hop — the engine handles the rvc (48k) sample-rate bridge.
        sampling_rate = 44100
        block_size = 512
        win_length = int(m.get("win_length", cfg.get("win_length", 2048)))

        mel_cfg = cfg.get("mel") or {}

        # Lock the enhancer to the reflow mel domain (overrides constructor args).
        self.sr = sampling_rate
        self.hop = block_size

        if self._mode == "shallow":
            # ---- ReflowShallowModel (standalone, no DDSP) ----
            opts = cfg.get("backend_opts", {})
            from ..backends.reflow_shallow.model import ReflowShallowModel
            self.shallow_model = ReflowShallowModel(
                out_dims=int(opts.get("out_dims", 128)),
                n_layers=int(opts.get("n_layers", 6)),
                n_chans=int(opts.get("n_chans", 768)),
                n_units=int(opts.get("n_units", opts.get("n_unit", self.n_units))),
                n_spk=int(opts.get("num_speaker", opts.get("n_spk", 1))),
                n_hidden=int(opts.get("n_hidden", 256)),
            )
            self._load_shallow_weights(reflow_ckpt)
            self.shallow_model.eval()
            self.shallow_model.to(device)
            self.ddsp_model = None  # no DDSP in shallow mode
        else:
            # ---- DDSP6 model (ddsp_model + reflow_model) ----
            self.ddsp_model = DDSP6Model(
                n_units=self.n_units,
                n_spk=1,
                win_length=win_length,
                sampling_rate=sampling_rate,
                block_size=block_size,
                n_aux_layers=n_aux_layers,
                n_aux_chans=n_aux_chans,
                n_layers=n_layers,
                n_chans=n_chans,
                velocity_net=velocity_net,
            )
            self._load_reflow_weights(reflow_ckpt)
            self.ddsp_model.eval()
            self.ddsp_model.to(device)
            self.shallow_model = None
        # reflow sampling stays fp32 (ODE iterations are numerically sensitive)

        # mel extractor: must match the reflow model's mel domain
        self.mel_extractor = MelExtractor(
            FeatureConfig(
                "mel",
                {
                    "n_fft": int(mel_cfg.get("n_fft", win_length)),
                    "win_length": int(mel_cfg.get("win_length", win_length)),
                    "n_mels": int(mel_cfg.get("n_mels", 128)),
                    "fmin": float(mel_cfg.get("fmin", 40.0)),
                    "fmax": float(mel_cfg.get("fmax", 16000.0)),
                    "center": bool(mel_cfg.get("center", False)),
                },
            ),
            sr=int(mel_cfg.get("sampling_rate", sampling_rate)),
            hop=int(mel_cfg.get("hop", block_size)),
        ).to(device)

        # Optional re-extraction of contentvec units from the enhancer-domain
        # wav (usually unnecessary since sovits/rvc already use contentvec 768-d).
        self.units_kind = str(cfg.get("units_kind", "contentvec"))
        self.units_extractor = None
        if self.units_kind != "contentvec":
            from ..features.units import UnitsExtractor
            self.units_extractor = UnitsExtractor(
                FeatureConfig(
                    "units",
                    {
                        "kind": self.units_kind,
                        "sr_in": 16000,
                        "hop_native": 320,
                        "align_mode": "gather",
                        "model_ckpt": cfg.get(
                            "units_ckpt",
                            "pretrained/contentvec/pytorch_model.bin",
                        ),
                    },
                ),
                sr, hop, device, dtype=torch.float32,
            )

        # ---- vocoder (fp32; sampling numerically sensitive) ----
        vocoder_dir = Path(vocoder_dir)
        self.vocoder = NSFHiFiGAN(
            vocoder_dir / "config.json",
            vocoder_dir / "model.ckpt",
            device=device,
            is_half=False,
        )
        # 推理加速: channels_last 数据流（conv1d→conv2d(1,k) 等价变换）。
        # 数值等价已验证（SNR ≥ 64dB）；autocast fp16 下消除逐层布局转换。
        if hasattr(self.vocoder, "generator"):  # 测试用 _DummyVocoder 无此属性
            self.vocoder.generator._use_cl = True

    # -- weight loading ----------------------------------------------------

    def _load_shallow_weights(self, ckpt_path: str | Path) -> None:
        """Load a ReflowShallowModel checkpoint.

        Accepts lightning ckpts (keys under ``state_dict`` with a ``model.``
        prefix from ``self.model = ReflowShallowModel``) and bare model ckpts.
        """
        from ..io.ckpt import load_ckpt_state
        sd = load_ckpt_state(ckpt_path)
        # Strip the lightning "model." prefix if present.
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        # Remap stored key name to ReflowShallowModel structure.
        sd = {
            (k.replace("reflow_velocity_fn.", "reflow_model.velocity_fn.")
             if k.startswith("reflow_velocity_fn.") else k): v
            for k, v in sd.items()
        }
        # drop shape-mismatched keys (e.g. DDSP6 conditioner_projection 128→256)
        model_sd = self.shallow_model.state_dict()
        dropped_shape = [k for k, v in sd.items()
                         if k in model_sd and v.shape != model_sd[k].shape]
        if dropped_shape:
            print(
                f"[reflow_shallow] dropped {len(dropped_shape)} shape-mismatched "
                f"key(s) (expected when finetuning from a DDSP6 base with a "
                f"different dim_cond): {dropped_shape[:5]}"
            )
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        dropped_shape_set = set(dropped_shape)
        missing, unexpected = self.shallow_model.load_state_dict(sd, strict=False)
        if missing:
            # Tolerable missing keys: optional submodules + reflow_shallow-
            # only embedding layers (random-initialised when migrating from DDSP6).
            _OPTIONAL = (
                "ddsp_model.", "mel_extractor.", "vocoder", "_", "cfg",
                "unit_embed.", "f0_embed.", "volume_embed.", "spk_embed.",
                "aug_shift_embed.",
            )
            real_missing = [k for k in missing
                            if not k.startswith(_OPTIONAL) and k not in dropped_shape_set]
            if real_missing:
                raise RuntimeError(
                    f"[reflow_shallow] ckpt missing {len(real_missing)} required "
                    f"weight(s); architecture mismatch? e.g. {real_missing[:5]}"
                )
            print(f"[reflow_shallow] missing optional/new-layer keys: {len(missing)} (e.g. {missing[:3]})")
        # unexpected: tolerate lightning metadata / optimizer keys
        bad = [k for k in unexpected
               if not k.startswith(("ddsp_model.", "mel_extractor.", "_", "vocoder", "cfg"))]
        if bad:
            print(f"[reflow_shallow] unexpected keys: {bad[:5]}")

    def _load_reflow_weights(self, ckpt_path: str | Path) -> None:
        from ..io.ckpt import load_ckpt_state
        sd = load_ckpt_state(ckpt_path)
        # Strip the lightning "model." prefix if present (usvc training ckpts save
        # DDSP6Model under 'model.ddsp_model.*' / 'model.reflow_model.*').
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = self.ddsp_model.load_state_dict(sd, strict=False)
        # DDSP6 mel_extractor weights are not in the reflow ckpt (they are
        # deterministic STFT buffers); tolerate them as missing.
        tolerable_missing = [
            k for k in missing
            if k.startswith(("mel_extractor.",))
        ]
        bad_missing = [k for k in missing if k not in tolerable_missing]
        if bad_missing:
            raise RuntimeError(
                f"reflow ckpt missing required model keys: {bad_missing[:8]}"
            )

    # -- shallow-diffusion sample (direct ddsp + reflow with external gt) --

    def _shallow_sample(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        volume: torch.Tensor,
        spk_id: torch.Tensor,
        gt_spec: torch.Tensor,
        t_start: float,
        infer_step: int,
        method: str,
    ) -> torch.Tensor:
        """Run DDSP synthesis (for the condition mel) + reflow toward ``gt_spec``.

        We bypass ``Unit2Wav.forward`` because its infer branch ignores the
        caller's ``gt_spec`` whenever a mel extractor is present (it reuses
        the DDSP-synthesised mel as both condition *and* target).  Instead we
        build the condition mel from DDSP synthesis, then feed the *external*
        (sovits/rvc) mel as the reflow refinement target.

        Steps:
          1. ddsp_wav, _ = ddsp_model(units, f0, volume, spk_id)
          2. ddsp_mel = mel_extractor.extract(ddsp_wav)   # reflow condition
          3. mel = reflow_model(ddsp_mel, gt_spec=gt_spec, t_start, infer)
        """
        # 1) DDSP synthesis -> condition mel
        ddsp_wav, _hidden = self.ddsp_model.ddsp_model(
            units, f0, volume, spk_id=spk_id, infer=True,
        )
        # ddsp_wav is [B, L]; DDSP6 MelExtractor.extract expects [B, L].
        ddsp_mel = self.ddsp_model.mel_extractor.extract(ddsp_wav)
        # ddsp_mel is now [B, T_ddsp, M]; align T_ddsp to gt_spec frames (dim 1).
        T_gt = gt_spec.shape[1]
        ddsp_mel = ddsp_mel[:, :T_gt]
        if ddsp_mel.shape[1] < T_gt:
            ddsp_mel = torch.nn.functional.pad(
                ddsp_mel, (0, 0, 0, T_gt - ddsp_mel.shape[1])
            )
        ddsp_mel = ddsp_mel.to(gt_spec.device)  # [1, T, M]

        # 2) reflow shallow diffusion toward the external (sovits/rvc) mel
        mel = self.ddsp_model.reflow_model(
            condition=ddsp_mel,
            gt_spec=gt_spec,
            infer=True,
            infer_step=infer_step,
            method=method,
            t_start=t_start,
            use_tqdm=False,
        )  # [1, T, M] denormalized (dB domain)
        return mel

    def enhance(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        rms: torch.Tensor | None,
        spk_id: int,
        gen_wav: torch.Tensor | np.ndarray,
        t_start: float = 0.6,
        infer_step: int = 20,
        method: str = "euler",
    ) -> np.ndarray:
        """Shallow-diffuse ``gen_wav`` via DDSP6 reflow.

        The DDSP6 RectifiedFlow takes the DDSP-synthesised mel as its
        condition and the sovits/rvc mel as the refinement target
        (``gt_spec``).  We call ``ddsp_model.ddsp_model`` to synthesise the
        condition mel, then run ``reflow_model`` with the external mel as the
        target, then resynthesise with the PC vocoder.

        Args:
            units:  [T, D] content features (device tensor, any dtype).
            f0:     [T] continuous f0 (Hz).
            rms:    [T] volume or None (fallback: estimated from mel energy).
            spk_id: 0-based speaker id.
            gen_wav: sovits/rvc output wav [L] float (at ``self.sr``).
            t_start: reflow ODE start time in [0, 1).  0 = full generation,
                →1 = keep gt.  Typical shallow diffusion: 0.3–0.7.
            infer_step: ODE euler steps.  ≥20 recommended: with 10 steps the
                euler dt (0.05 at t_start=0.5) overshoots near the training
                lower bound and mangles the spectrum (1-3 kHz bulge =
                "underwater").  20 steps / dt 0.025 measured clean.
            method: 'euler' (fast) or 'rk4' (higher order).

        Returns:
            enhanced_wav [L'] float32 numpy (at ``self.sr``).
        """
        T = units.shape[0]

        # 0) re-extract units if the enhancer uses a different encoder
        if self.units_extractor is not None:
            import librosa
            wav_np = gen_wav if isinstance(gen_wav, np.ndarray) else gen_wav.cpu().numpy()
            wav16k = librosa.resample(
                wav_np.astype(np.float32), orig_sr=self.sr, target_sr=16000
            )
            wav16k_t = torch.from_numpy(wav16k).float().to(self.device)
            units = self.units_extractor.extract(wav16k_t)
            units = units[:T]
            if units.shape[0] < T:
                units = torch.nn.functional.pad(units, (0, 0, 0, T - units.shape[0]))

        # 1) gen_wav → gt_spec (log-dB mel, matching DDSP6 training params)
        mel = self.mel_extractor.extract(gen_wav)  # [T', M] cpu float32
        mel = mel[:T]
        if mel.shape[0] < T:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, T - mel.shape[0]))
        mel = mel.unsqueeze(0).to(self.device)  # [1, T, M]

        # 2) batch tensors (fp32, ODE is numerically sensitive)
        units_b = units.float().unsqueeze(0).to(self.device)  # [1, T, D]
        f0_b = f0.float().unsqueeze(0).to(self.device)        # [1, T]
        if rms is not None:
            vol_b = rms.float().unsqueeze(0).to(self.device)  # [1, T]
        else:
            vol_b = (mel.exp() - 1e-6).clamp(min=1e-6).sqrt().mean(dim=-1)
        # DDSP6 expects [B, T, 1] for f0 and volume
        f0_b = f0_b.unsqueeze(-1)
        vol_b = vol_b.unsqueeze(-1)
        spk_b = torch.tensor([spk_id], dtype=torch.long, device=self.device)

        # 3-4) shallow diffusion
        with torch.no_grad():
            if self._mode == "shallow":
                # shallow mode: condition = sovits/rvc mel + embeddings
                denoised = self.shallow_model.sample(
                    units_b, f0_b.squeeze(-1), vol_b.squeeze(-1), mel,
                    t_start=t_start, infer_step=infer_step, method=method,
                    spk_id=spk_b,
                )  # [1, T, M] denormalized (dB domain)
            else:
                # ddsp6 mode: DDSP-synthesised mel as condition + reflow
                denoised = self._shallow_sample(
                    units_b, f0_b, vol_b, spk_b, mel,
                    t_start=t_start, infer_step=infer_step, method=method,
                )  # [1, T, M] denormalized (dB domain)

            # 5) PC vocoder resynthesis (autocast fp16: conv 跑 tensor core)
            f0_for_voc = f0_b.squeeze(-1)  # [1, T]
            with torch.autocast("cuda", dtype=torch.float16):
                wav = self.vocoder.infer(denoised.float(), f0_for_voc)

        return wav[0, 0].float().cpu().numpy()
