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

```bash
pip install usvc
```

Requires Python ≥ 3.10, PyTorch ≥ 2.4 with CUDA.

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
