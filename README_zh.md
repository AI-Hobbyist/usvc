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

### Linux（开箱即全速）

```bash
# 1. 先装 CUDA 版 PyTorch（会自动带 triton）
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. 装 usvc —— triton 作为依赖自动装上，全部加速生效
pip install usvc
```

完成。`torch.compile`、手写 Triton kernel、channels_last、CUDA graph 全部
自动生效。运行 `usvc-checkenv` 验证。

### Windows（全速）

```bash
# 1. 先装 CUDA 版 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. 装 usvc 的 [gui] extra（拉取版本匹配的 triton-windows）
pip install usvc[gui]
```

`triton-windows` 在 Windows 上提供 `triton` 模块（官方 triton 无 Windows
wheel）。其版本**必须与 PyTorch 小版本对应**：

| PyTorch | triton-windows |
|---------|----------------|
| 2.10    | 3.6            |
| 2.9     | 3.5            |
| 2.8     | 3.4            |
| 2.7     | 3.3            |
| 2.6     | 3.2            |
| 2.4–2.5 | 3.1            |

若版本不匹配，`pip install usvc[gui]` 可能拉错版本——此时手动
`pip install triton-windows==<对应版本>`。

### 仅 CPU / 无 Triton（优雅降级）

```bash
pip install usvc
```

无 triton 时，`torch.compile` 回退 eager、手写 kernel 回退 PyTorch 原生——
一切仍**可用**，只是变慢。`import usvc` 时会有一次性提示。恢复全速请装
triton（Linux：`pip install triton`；Windows：`pip install usvc[gui]`）。

### 验证环境

```bash
usvc-checkenv
```

需要 Python ≥ 3.10，NVIDIA GPU sm ≥ 8.0（Ampere+）才能用 channels_last；
更老的 GPU（Turing/Volta）仍有 TensorCore fp16 + torch.compile 加速。

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

### DDSP-SVC 6.5

`ddsp6` 后端支持 6.1、6.2、6.3 和 6.5，默认仍为 6.3。6.5 使用
`configs/ddsp65.yaml` 模板：

```bash
usvc-preprocess run data/my_dataset --backend ddsp6 --train-cfg configs/ddsp65.yaml
usvc-train --config configs/ddsp65.yaml
```

6.5 的核心变化是双时间步训练（25% 的帧选择第二时间步）、RMSNorm、
可配置的 `softsign_glu` / `atanglu` / `double_softsign_glu`，以及新的 DDSP
控制网络。新结构保持原版 6.5 权重键，6.1–6.3 权重不能作为 6.5 权重加载。
在同一输入、权重和随机种子下，本地原版对照测试要求 FP32 / CUDA BF16
训练损失、Euler / RK4 推理 mel 完全一致，梯度数值一致（共享 mel 前端也
单独验证；CUDA STFT 梯度归约允许微小舍入误差）。
BF16 对照在两侧都将频谱控制值提升为 FP32；当前 PyTorch 下，原版直接
使用 BF16 复数运算会失败，usvc 6.5 保持频谱合成 FP32，梯度仍正常回传。

与原版 6.5 的区别：usvc 说话人编号从 0 开始，支持变长批次的 flow 损失
掩码；预处理继续使用 usvc 的统一帧对齐和 RMS 音量，原版音量为中心窗口
标准差。因此，原版数据与 usvc 数据不保证逐帧一致。模板使用 AdamW，
原版使用 Muon+AdamW；原版专用 Triton 融合训练核未移植，可选用 usvc 的
`compile_velocity`。6.5 当前支持 PyTorch，MLX 会明确拒绝此版本。

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

## DDSP / SoVITS 多说话人

专用模板为 `configs/ddsp_multispeaker.yaml`（6.5，可改为6.1/6.2/6.3）和
`configs/sovits_multispeaker.yaml`。现有 USVC 配置仍可使用；新增顶层 `spk` 是名称到零基整数的映射，
`backend_opts.num_speaker` 必须等于映射容量。未写 `spk` 的新训练按训练 meta 顺序固定映射。
原始数据按 `<src>/<说话人>/<音频>` 放置；split 使用 `说话人/文件名` 避免同名音频歧义。
预处理与训练使用同一配置。训练目录保存已解析 `config.yaml`；完整 checkpoint 与导出权重均携带映射。

```powershell
usvc-preprocess init data/multi --src data/multi_raw
usvc-preprocess run data/multi --backend ddsp6 --train-cfg configs/ddsp_multispeaker.yaml
usvc-train --config configs/ddsp_multispeaker.yaml
usvc-export --ckpt experiments/ddsp_multi/last.ckpt -o experiments/ddsp_multi/model.pth
usvc-infer --config experiments/ddsp_multi/model.yaml --ckpt experiments/ddsp_multi/model.pth --list-speakers
usvc-infer input.wav --config experiments/ddsp_multi/model.yaml --ckpt experiments/ddsp_multi/model.pth --speaker '歌手乙' -o output.wav
```

多人推理必须提供互斥的 `--speaker <名称>` 或 `--spk <整数>`；单人默认ID0。
旧无映射多人权重仅接受显式数字；名称需要可信的训练映射，不能从目录猜测 embedding 行。
恢复训练拒绝角色新增、删除或行语义变化。SoVITS 聚类与每角色 `.pt` 索引跟随目标名称选择；
DDSP 增强器的多人组合也需可验证的同名对应关系。`usvc-serve` job 参数支持互斥的 `speaker` / `spk`。
本次多说话人验收针对 Windows/PyTorch；MLX 多说话人仍为待验收。

RVC 的 `--index-path` 同时接受现有 `.pt` 与原生 `.index`。原生格式需安装 `.[rvc-index]` extra，
支持 L2 Flat / IVF Flat，保留保存的 IVF `nprobe`，使用有预算限制的 CPU 缓存；比例0跳过索引。
完整操作与各后端说话人支持范围见 [训练推理教程](训练推理教程.md)。

## 许可证

MIT
