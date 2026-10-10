# usvc

Unified SVC (Singing Voice Conversion) framework: shared preprocessing,
pluggable backends, single training & inference pipeline. Distilled from
seven reference projects into a clean, zero-legacy codebase.

**English** | **[中文](README_zh.md)**

Seven backends in one framework:

| Backend | Source project | Type |
|---------|---------------|------|
| `ddsp6` | DDSP-SVC 6.x | DDSP + Rectified Flow |
| `diffusion` | Diffusion-SVC | DDPM (WaveNet denoiser) |
| `rift_dit` | RIFT-SVC | Flow Matching (DiT) |
| `sovits` | so-vits-svc 4.1 | GAN (NSF HiFi-GAN) |
| `rvc` | RVC | GAN (NSF HiFi-GAN) |
| `reflow_vae` | DDSP-SVC reflow | Rectified Flow (standalone) |
| `reflow_shallow` | — | Shallow diffusion enhancer |

Shallow-diffusion enhancement is available for `rvc` and `sovits` via either
the `reflow_shallow` or `diffusion` enhancer.

## Install

### Linux (full acceleration out of the box)

```bash
# 1. Install CUDA build of PyTorch first (brings triton automatically)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. Install usvc — triton comes as a dependency, all acceleration enabled
pip install usvc
```

That's it. `torch.compile`, hand-written Triton kernels, channels_last, and
CUDA graphs all work automatically. Run `usvc-checkenv` to verify.

### Windows (full acceleration)

```bash
# 1. Install CUDA build of PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. Install usvc with the [gui] extra (pulls triton-windows, version-matched)
pip install usvc[gui]
```

`triton-windows` provides the `triton` module on Windows (official `triton`
has no Windows wheels). Its version **must match** your PyTorch minor version:

| PyTorch | triton-windows |
|---------|----------------|
| 2.10    | 3.6            |
| 2.9     | 3.5            |
| 2.8     | 3.4            |
| 2.7     | 3.3            |
| 2.6     | 3.2            |
| 2.4–2.5 | 3.1            |

If the versions mismatch, `pip install usvc[gui]` may pull the wrong one — in
that case install `triton-windows==<matching>` explicitly.

### CPU-only / no Triton (graceful degradation)

```bash
pip install usvc
```

Without triton, `torch.compile` falls back to eager and hand-written kernels
fall back to native PyTorch — everything still **works**, just slower. You'll
see a one-time hint on `import usvc`. To restore full speed, install triton
(Linux: `pip install triton`; Windows: `pip install usvc[gui]`).

### Verify your setup

```bash
usvc-checkenv
```

Requires Python ≥ 3.10, NVIDIA GPU sm ≥ 8.0 (Ampere+) for channels_last;
older GPUs (Turing/Volta) still get TensorCore fp16 + torch.compile speedups.

### Pretrained weights

Base models (ContentVec, vocoder, F0 extractors) are **downloaded
automatically** on first use — no manual setup needed. To pre-download or
use a custom location, set ``USVC_PRETRAINED_DIR``.

SVC base models (for fine-tuning) are available from
[ModelScope](https://www.modelscope.cn/models/baicai1145/usvc-pretrained)
or [HuggingFace](https://huggingface.co/baicai1145/usvc-pretrained):

| Directory | Contents |
|-----------|----------|
| `pretrained/contentvec/` | ContentVec encoder (HF format) |
| `pretrained/nsf_hifigan/` | PC-NSF-HiFiGAN vocoder |
| `pretrained/hubert/` | HubertSoft (so-vits-svc 4.0) |
| `pretrained/rmvpe/` | RMVPE F0 extractor |
| `pretrained/fcpe/` | FCPE F0 extractor |
| `pretrained/base_models/` | SVC base models (ddsp6/sovits/rvc/rift_dit/...) |

```bash
# Download everything (optional — base models auto-download on first use)
# HuggingFace:
hf download baicai1145/usvc-pretrained --local-dir pretrained
# ModelScope:
modelscope download --model baicai1145/usvc-pretrained --local_dir pretrained
```

## Quick start

### DDSP-SVC 6.5

The `ddsp6` backend supports 6.1, 6.2, 6.3 and 6.5; the default remains 6.3.
Use `configs/ddsp65.yaml` for 6.5:

```bash
usvc-preprocess run data/my_dataset --backend ddsp6 --train-cfg configs/ddsp65.yaml
usvc-train --config configs/ddsp65.yaml
```

6.5 adds dual-timestep training (25% of frames select a second timestep), RMSNorm,
configurable SoftSign/ATan/double-SoftSign GLU and a new DDSP control network.
Original 6.5 state-dict keys are preserved; 6.1–6.3 weights are incompatible.
Local reference tests require identical FP32/CUDA BF16 losses and Euler/RK4 mel
outputs with shared inputs, weights and random seeds, and numerically matching
gradients (allowing CUDA STFT reduction rounding). The shared
mel frontend is checked independently too.
BF16 parity uses FP32 spectral controls on both sides: raw upstream BF16 complex
operations fail on the current PyTorch, so usvc keeps 6.5 spectral synthesis in
FP32 while preserving gradients.

Differences from upstream: usvc uses zero-based speaker IDs and masks padded
frames in the flow loss. Preprocessing retains usvc's frame alignment and RMS
volume, whereas upstream uses centered-window standard deviation. Dataset
features are therefore not guaranteed to match frame for frame. The template
uses AdamW rather than upstream Muon+AdamW. Upstream-specific fused Triton training
kernels are not ported; usvc's `compile_velocity` remains optional. 6.5 currently
requires PyTorch; MLX explicitly rejects it.

### Inference

```bash
# Convert a wav file (any backend)
usvc-infer input.wav \
    --config configs/sovits.yaml \
    --ckpt experiments/sovits/last.ckpt \
    --output output.wav \
    --trans 0

# With shallow diffusion enhancement
usvc-infer input.wav \
    --config configs/sovits.yaml \
    --ckpt experiments/sovits/last.ckpt \
    --shd \
    --reflow-config configs/reflow_shallow.yaml \
    --reflow-ckpt experiments/reflow_shallow/last.ckpt
```

### Training

```bash
# 1. Stage audio + init metadata
usvc-preprocess init data/myds --src /path/to/raw --sr 44100 --hop 512

# 2. Extract features (config-hash dedup: re-running skips done features)
usvc-preprocess run data/myds --backend sovits --device cuda

# 3. Train
usvc-train --config configs/sovits.yaml \
    --data-dir data/myds \
    --exp-dir experiments/sovits
```

## Performance (A10G)

usvc is aggressively optimized for A10G with Triton kernels, `torch.compile`, and
CUDA-graph inference. Timings (pure 1000 steps / real audio):

| Stage | Baseline | Optimized | Speedup |
|-------|----------|-----------|---------|
| **Preprocess** contentvec (units) | 1077 ms | 323 ms | **3.33×** |
| **Train** ddsp6 (conv) | ~3.8 min/1k | 1.25 min/1k | 1.10× |
| **Train** diffusion (conv) | ~4.2 min/1k | 1.15 min/1k | 1.38× |
| **Train** reflow_shallow (conv) | ~3.1 min/1k | 0.42 min/1k | 1.27× |
| **Train** sovits / rvc (GAN) | — | compute-bound | at roofline |
| **Infer** rvc (10s clip) | — | 134 ms | RTF 0.013 (75× rt) |

Key optimizations (opt-in, semantics-preserving, bit-exact equivalence guarded):

- **contentvec flash-attention** (Triton): 3.1× on the dominant preprocessing step.
- **`torch.compile`** on conv velocity/denoise networks (`compile_velocity` /
  `compile_denoise` backend opts): 1.10–1.38× on conv backends.
- **`--compile-graph`** inference flag (CUDA-graph via `reduce-overhead`): 1.16× on rvc.

GAN backends (sovits/rvc) are compute-bound at ~95% SM occupancy (NSF-HiFiGAN conv
dominates) — `torch.compile` is ineffective there due to random-slice conflicts, so
they ship at the measured roofline.

## Backend comparison

| Backend | Domain | Sample rate | Hop | Units encoder | F0 default | Shallow diffusion |
|---------|--------|-------------|-----|---------------|-----------|-------------------|
| `ddsp6` | mel | 44.1k | 512 | ContentVec (TTA 2x) | FCPE | — |
| `diffusion` | mel | 44.1k | 512 | ContentVec | FCPE | — |
| `rift_dit` | mel | 44.1k | 512 | ContentVec (TTA 2x) | FCPE | — |
| `reflow_vae` | mel | 44.1k | 512 | ContentVec | FCPE | — |
| `sovits` | wav | 44.1k | 512 | ContentVec | FCPE | ✅ reflow / diffusion |
| `rvc` | wav | 48k | 480 | ContentVec | FCPE | ✅ reflow / diffusion |
| `reflow_shallow` | mel | 44.1k | 512 | — (conditioned on sovits/rvc mel) | — | is an enhancer |

**Mel-domain** backends sample a mel-spectrogram, then resynthesize via a
shared NSF HiFi-GAN vocoder. **Wav-domain** backends (sovits/rvc) generate
audio directly with an integrated GAN generator.

## CLI reference

```text
usvc-preprocess init   # Stage audio + init meta.json
usvc-preprocess run    # Extract features (units/f0/mel/rms)

usvc-train             # Train any backend

usvc-infer             # Offline inference (all backends)
  --config        -c   Backend config YAML
  --ckpt               Model checkpoint
  --vocoder-dir        Vocoder dir (mel-domain backends)
  --output        -o   Output wav path
  --trans              Pitch shift (semitones)
  --spk                Speaker ID
  --steps              Sampling steps (rift_dit / reflow / diffusion)
  --f0-method          rmvpe | fcpe (runtime override)
  --noice-scale        GAN noise scale (rvc/sovits)
  --formant-shift      Formant shift (semitones)
  --vocal-register-shift   Register shift (mel-domain, semitones)
  --resample-sr        Output resample rate
  --target-loudness    LUFS normalization before conversion
  --rms-mix-rate       Output RMS envelope match (0–1)
  --protect            RVC source-feature protection (0–0.5)
  --index-path         Feature retrieval index
  --cluster-path       Cluster model for retrieval
  --shd                Enable shallow diffusion (rvc/sovits)
  --reflow-config      Reflow enhancer config
  --reflow-ckpt        Reflow enhancer checkpoint
  --diffusion-config   Diffusion enhancer config
  --diffusion-ckpt     Diffusion enhancer checkpoint
  --k-step             Shallow diffusion k_step (20–1000)
  --t-start            Reflow ODE start time (0–1)
  --ds-cfg             Rift-DiT DS classifier-free guidance
  --spk-cfg            Rift-DiT speaker CFG
  --skip-cfg           Rift-DiT skip CFG
```

## Acknowledgements

This framework is built upon the work of these projects:

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) — DDSP6, reflow, LYNXNet
- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) — DDPM, WaveNet denoiser
- [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC) — Flow matching, DiT
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) — GAN, NSF HiFi-GAN
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — GAN, contentvec, retrieval
- [ContentVec](https://github.com/auspicious3000/contentvec) — Speech representation
- [NSF-HiFiGAN](https://github.com/openvpi/SingingVocoders) — Neural source-filter vocoder
- [RMVPE](https://github.com/yxliang20/RMVPE) / [FCPE](https://github.com/CNChTu/FCPE) — F0 extraction

## License

MIT
