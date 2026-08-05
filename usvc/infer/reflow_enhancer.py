"""Reflow shallow-diffusion enhancer (DDSP6 RectifiedFlow backend).

Reuses the ``RectifiedFlow`` + ``ddsp_model.unit2ctrl`` from the DDSP6 backend
to perform t_start shallow diffusion on the mel produced by a wav-domain
generator (sovits / rvc). The final wav is resynthesised by the shared
PC NSF-HiFiGAN vocoder.

Data flow::

    rvc/sovits wav ──MelExtractor──▶ gt_spec (log-dB mel)
    units/f0/rms/spk ──ddsp_model.unit2ctrl──▶ hidden (condition)
    gt_spec + hidden ──RectifiedFlow(t_start)──▶ 去噪 mel
    去噪 mel + f0 ──NSF-HiFiGAN (PC)──▶ 最终 wav

Key design:
  - The DDSP6 ``RectifiedFlow`` velocity network takes the *mel* as its
    condition (``dim_cond == out_dims == 128``).  The condition mel comes
    from DDSP synthesis: ``units+f0+volume+spk -> ddsp_model -> ddsp_wav ->
    extract mel``.  ``Unit2Wav.forward``'s infer branch is unsuitable here
    because it ignores the caller's ``gt_spec`` whenever a mel extractor is
    present (it reuses the DDSP mel as both condition *and* target).
  - For shallow diffusion we therefore build the condition mel via DDSP
    synthesis ourselves, then feed the *external* (sovits/rvc) mel as the
    reflow ``gt_spec`` (refinement target):
      1. ddsp_wav = ddsp_model(units, f0, volume, spk)
      2. ddsp_mel = mel_extractor.extract(ddsp_wav)   # condition
      3. mel = reflow_model(condition=ddsp_mel, gt_spec=<external mel>,
                            t_start)                   # shallow diffusion
    This makes the sovits/rvc output genuinely drive the refinement.

RectifiedFlow shallow-diffusion math (``model.py:RectifiedFlow.forward``)::

    if gt_spec is None:                 # 纯生成
        x = randn(shape);  t = 0;       dt = 1/infer_step
    else:                               # 浅扩散
        x = t_start * norm(gt_spec) + (1-t_start) * randn(shape)
        t = t_start;        dt = (1 - t_start) / infer_step
    for i in range(infer_step):
        x, t = sample_euler(x, t, dt, cond)   # euler / rk4

This is mathematically equivalent to diffusion k_step shallow diffusion, but
rectified flow converges in ~10 steps (vs 25-100 for DDPM).
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


def _load_ckpt_state(ckpt_path: str | Path) -> dict:
    """Load state dict from a torch ckpt (lightning or bare)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f"cannot interpret checkpoint: {ckpt_path}")
    return sd


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
        half: bool = False,
        sr: int = 44100,
        hop: int = 512,
    ) -> None:
        self.device = device
        # The enhancer always operates in the DDSP6 mel domain (44.1k/512).
        # The engine handles the sample-rate bridge for rvc (48k) separately,
        # so self.sr/self.hop reflect the enhancer's own domain, not the
        # backend's.  These are overridden below from the model config.
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

        # ---- mel extractor (must match DDSP6 training mel params) ----
        # ---- mel extractor (must EXACTLY match DDSP6's internal mel_extractor) ----
        # DDSP6's MelExtractor uses fmin=40 / fmax=16000 defaults (44.1k/512/2048).
        # The gt_spec we extract here is the reflow *target*; it must live in
        # the same mel domain as the DDSP-synthesised condition mel.
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

        # ---- contentvec units extractor (if enhancer units differ from
        # the main generator's units, re-extract from the enhancer-domain wav)
        # ----
        # DDSP6 expects contentvec (768-d).  sovits/rvc already use contentvec
        # 768-d, so in most cases re-extraction is unnecessary.  We keep the
        # hook for safety (matching DiffusionEnhancer's design).
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

    # -- weight loading ----------------------------------------------------

    def _load_shallow_weights(self, ckpt_path: str | Path) -> None:
        """Load a ReflowShallowModel checkpoint.

        Accepts lightning ckpts (keys under ``state_dict`` with a ``model.``
        prefix from ``self.model = ReflowShallowModel``) and bare model ckpts.
        """
        sd = _load_ckpt_state(ckpt_path)
        # Strip the lightning "model." prefix if present.
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        # drop shape-mismatched keys (e.g. DDSP6 conditioner_projection 128→256)
        model_sd = self.shallow_model.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = self.shallow_model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[reflow_shallow] missing keys: {len(missing)} (e.g. {missing[:3]})")
        # unexpected: tolerate lightning metadata / optimizer keys
        bad = [k for k in unexpected
               if not k.startswith(("ddsp_model.", "mel_extractor.", "_", "vocoder", "cfg"))]
        if bad:
            print(f"[reflow_shallow] unexpected keys: {bad[:5]}")

    def _load_reflow_weights(self, ckpt_path: str | Path) -> None:
        sd = _load_ckpt_state(ckpt_path)
        missing, unexpected = self.ddsp_model.load_state_dict(sd, strict=False)
        # DDSP6 mel_extractor weights are not in the reflow ckpt (they are
        # deterministic STFT buffers); tolerate them as missing.
        tolerable_missing = [
            k for k in missing
            if k.startswith("mel_extractor.")
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
        t_start: float = 0.5,
        infer_step: int = 10,
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
            infer_step: ODE solver steps (10 is plenty for euler).
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
                # ReflowShallowModel v2: condition = units/f0/vol embeddings
                # (mel-independent), gt_spec = sovits/rvc mel.  The reflow
                # ODE starts from (t_start*sovits_mel + noise) and integrates
                # toward the clean-mel manifold guided by the mel-free
                # condition — a true shallow diffusion (DDPM-equivalent).
                denoised = self.shallow_model.sample(
                    units_b, f0_b.squeeze(-1), vol_b.squeeze(-1), mel,
                    t_start=t_start, infer_step=infer_step, method=method,
                    spk_id=spk_b,
                )  # [1, T, M] denormalized (dB domain)
            else:
                # DDSP6 mode: DDSP synthesis (condition) + reflow toward the
                # external mel.  _shallow_sample bypasses Unit2Wav.forward
                # (whose infer branch ignores gt_spec when a mel extractor is
                # present) and feeds the sovits/rvc mel directly.
                denoised = self._shallow_sample(
                    units_b, f0_b, vol_b, spk_b, mel,
                    t_start=t_start, infer_step=infer_step, method=method,
                )  # [1, T, M] denormalized (dB domain)

            # 5) PC vocoder resynthesis
            f0_for_voc = f0_b.squeeze(-1)  # [1, T]
            wav = self.vocoder.infer(denoised.float(), f0_for_voc)

        return wav[0, 0].float().cpu().numpy()
