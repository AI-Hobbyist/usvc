"""usvc MLX 后端:Apple Silicon Metal 加速实现。

本包镜像 usvc 的 torch 实现,提供 MLX 版的特征提取器、vocoder、各 backend
模型。与 torch 路径**并列共存**,通过 ``usvc._device.resolve_impl`` 选择
(``USVC_BACKEND=mlx`` 环境变量或代码显式指定),torch 路径零改动。

模块结构(按实现优先级):
  - ``_ops``: MLX 缺失算子(weight_norm 合并、STFT、interpolate、reflect pad)
  - ``_convert``: torch ckpt → mlx state_dict(Conv 转置、key remap)
  - ``_common``: 共享 Protocol(FeatureConfigLike)
  - ``_device``: (在 usvc 顶层)impl + device 解析
  - ``contentvec``: MLX ContentVec(推理+预处理最高频)
  - ``mel``: MLX MelExtractor(nvSTFT center=False 帧网格)
  - ``nsf_hifigan``: MLX NSF-HiFiGAN vocoder(mel-domain 必需)

后续(见 MLX 映射手册):
  - backends/{rvc,sovits,ddsp6,diffusion,rift_dit,reflow_vae}.py
  - fcpe.py / rmvpe.py(F0 提取)

数值对齐:所有模型目标与 torch 版 max abs diff < 1e-4(帧网格红线逐位复现)。

用法示例::

    import os; os.environ["USVC_BACKEND"] = "mlx"
    from usvc.nn.mlx.contentvec import contentvec_from_hf_ckpt
    m, _, _ = contentvec_from_hf_ckpt("pretrained/contentvec/pytorch_model.bin")
    hidden = m.encode(wav_mlx)   # [B, T_frames, 768]
"""
from __future__ import annotations

__all__ = ["_HAS_MLX"]

try:
    import mlx.core as mx  # noqa: F401

    _HAS_MLX = True
except Exception:  # pragma: no cover
    _HAS_MLX = False
