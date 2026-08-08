# usvc

统一 SVC（歌声转换）框架：共享预处理、可插拔后端、统一的训练与推理流程。从七个参考项目中提炼精华，零历史包袱的干净代码库。

**[English](README.md)** | **中文**

一个框架，七个后端：

| 后端 | 来源项目 | 类型 |
|------|---------|------|
| `ddsp6` | DDSP-SVC 6.x | DDSP + Rectified Flow |
| `diffusion` | Diffusion-SVC | DDPM（WaveNet 去噪器）|
| `rift_dit` | RIFT-SVC | Flow Matching（DiT）|
| `sovits` | so-vits-svc 4.1 | GAN（NSF HiFi-GAN）|
| `rvc` | RVC | GAN（NSF HiFi-GAN）|
| `reflow_vae` | DDSP-SVC reflow | Rectified Flow（独立）|
| `reflow_shallow` | — | 浅扩散增强器 |

`rvc` 和 `sovits` 支持通过 `reflow_shallow` 或 `diffusion` 增强器进行浅扩散增强。

## 安装

```bash
pip install usvc
```

需要 Python ≥ 3.10，PyTorch ≥ 2.4（CUDA 版本）。

### 预训练权重

基础模型（ContentVec、声码器、F0 提取器）在首次使用时**自动下载**，
无需手动配置。如需预下载或使用自定义路径，设置环境变量
``USVC_PRETRAINED_DIR``。

各 SVC 后端底模（用于微调）可从
[ModelScope](https://www.modelscope.cn/models/baicai1145/usvc-pretrained)
或 [HuggingFace](https://huggingface.co/baicai1145/usvc-pretrained) 下载：

| 目录 | 内容 |
|------|------|
| `pretrained/contentvec/` | ContentVec 编码器（HF 格式）|
| `pretrained/nsf_hifigan/` | PC-NSF-HiFiGAN 声码器 |
| `pretrained/hubert/` | HubertSoft（so-vits-svc 4.0）|
| `pretrained/rmvpe/` | RMVPE 基频提取器 |
| `pretrained/fcpe/` | FCPE 基频提取器 |
| `pretrained/base_models/` | 各后端底模（ddsp6/sovits/rvc/rift_dit/...）|

```bash
# 下载全部（可选——基础模型首次使用时自动下载）
# HuggingFace:
hf download baicai1145/usvc-pretrained --local-dir pretrained
# ModelScope:
modelscope download --model baicai1145/usvc-pretrained --local_dir pretrained
```

## 快速开始

### 推理

```bash
# 转换音频文件（任意后端）
usvc-infer input.wav \
    --config configs/sovits.yaml \
    --ckpt experiments/sovits/last.ckpt \
    --output output.wav \
    --trans 0

# 开启浅扩散增强
usvc-infer input.wav \
    --config configs/sovits.yaml \
    --ckpt experiments/sovits/last.ckpt \
    --shd \
    --reflow-config configs/reflow_shallow.yaml \
    --reflow-ckpt experiments/reflow_shallow/last.ckpt
```

### 训练

```bash
# 1. 整理音频 + 初始化元数据
usvc-preprocess init data/myds --src /path/to/raw --sr 44100 --hop 512

# 2. 提取特征（config-hash 去重：重复运行会跳过已完成的特征）
usvc-preprocess run data/myds --backend sovits --device cuda

# 3. 训练
usvc-train --config configs/sovits.yaml \
    --data-dir data/myds \
    --exp-dir experiments/sovits
```

## 性能 (A10G)

usvc 针对 A10G 做了极致优化：Triton kernel + `torch.compile` + CUDA-graph 推理。
纯 1000 步 / 真实音频计时：

| 阶段 | 基线 | 优化后 | 加速 |
|------|------|--------|------|
| **预处理** contentvec (units) | 1077 ms | 323 ms | **3.33×** |
| **训练** ddsp6 (conv) | ~3.8 min/1k | 1.25 min/1k | 1.10× |
| **训练** diffusion (conv) | ~4.2 min/1k | 1.15 min/1k | 1.38× |
| **训练** reflow_shallow (conv) | ~3.1 min/1k | 0.42 min/1k | 1.27× |
| **训练** sovits / rvc (GAN) | — | 计算到顶 | 在 roofline |
| **推理** rvc (10s 片段) | — | 134 ms | RTF 0.013 (75×实时) |

关键优化（可选项，不改训练语义，bit 级等价守卫）：

- **contentvec flash-attention**（Triton）：预处理主项 3.1× 加速。
- **`torch.compile`** 编译 conv velocity/denoise 网络（`compile_velocity` /
  `compile_denoise` backend_opts）：conv 后端 1.10–1.38×。
- **`--compile-graph`** 推理开关（`reduce-overhead` 走 CUDA graph）：rvc 1.16×。

GAN 后端（sovits/rvc）计算到顶（SM ~95%，NSF-HiFiGAN conv 主导）——
`torch.compile` 因随机切片冲突无效，已在实测 roofline。

## 后端对比

| 后端 | 域 | 采样率 | Hop | 内容编码器 | 默认 F0 | 浅扩散 |
|------|-----|--------|-----|-----------|---------|--------|
| `ddsp6` | mel | 44.1k | 512 | ContentVec（TTA 2x）| FCPE | — |
| `diffusion` | mel | 44.1k | 512 | ContentVec | FCPE | — |
| `rift_dit` | mel | 44.1k | 512 | ContentVec（TTA 2x）| FCPE | — |
| `reflow_vae` | mel | 44.1k | 512 | ContentVec | FCPE | — |
| `sovits` | wav | 44.1k | 512 | ContentVec | FCPE | ✅ reflow / diffusion |
| `rvc` | wav | 48k | 480 | ContentVec | FCPE | ✅ reflow / diffusion |
| `reflow_shallow` | mel | 44.1k | 512 | —（以 sovits/rvc mel 为条件）| — | 本身是增强器 |

**mel 域**后端先采样 mel 频谱，再通过共享的 NSF HiFi-GAN 声码器重建音频。**wav 域**后端（sovits/rvc）直接用内置的 GAN 生成器生成音频。

## CLI 参考

```text
usvc-preprocess init   # 整理音频 + 初始化 meta.json
usvc-preprocess run    # 提取特征（units/f0/mel/rms）

usvc-train             # 训练任意后端

usvc-infer             # 离线推理（所有后端）
  --config        -c   后端配置 YAML
  --ckpt               模型权重
  --vocoder-dir        声码器目录（mel 域后端）
  --output        -o   输出路径
  --trans              变调（半音）
  --spk                说话人 ID
  --steps              采样步数（rift_dit / reflow / diffusion）
  --f0-method          rmvpe | fcpe（运行时覆盖）
  --noice-scale        GAN 噪声尺度（rvc/sovits）
  --formant-shift      共振峰偏移（半音）
  --vocal-register-shift   音区偏移（mel 域，半音）
  --resample-sr        输出重采样率
  --target-loudness    转换前 LUFS 响度归一化
  --rms-mix-rate       输出 RMS 包络匹配（0–1）
  --protect            RVC 源特征保护（0–0.5）
  --index-path         特征检索索引
  --cluster-path       聚类模型
  --shd                开启浅扩散（rvc/sovits）
  --reflow-config      Reflow 增强器配置
  --reflow-ckpt        Reflow 增强器权重
  --diffusion-config   Diffusion 增强器配置
  --diffusion-ckpt     Diffusion 增强器权重
  --k-step             浅扩散步数（20–1000）
  --t-start            Reflow ODE 起始时间（0–1）
  --ds-cfg             Rift-DiT DS 分类器自由引导
  --spk-cfg            Rift-DiT 说话人 CFG
  --skip-cfg           Rift-DiT skip CFG
```

## 致谢

本框架基于以下项目的工作构建：

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) — DDSP6、reflow、LYNXNet
- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) — DDPM、WaveNet 去噪器
- [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC) — Flow matching、DiT
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) — GAN、NSF HiFi-GAN
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — GAN、ContentVec、检索
- [ContentVec](https://github.com/auspicious3000/contentvec) — 语音表征
- [NSF-HiFiGAN](https://github.com/openvpi/SingingVocoders) — 神经源滤波器声码器
- [RMVPE](https://github.com/yxliang20/RMVPE) / [FCPE](https://github.com/CNChTu/FCPE) — 基频提取

## 许可证

MIT
