"""mlx_engine convert 高级功能:index/cluster 检索 + protect + rms_mix(numpy)。

复用 torch 版 retrieval/predict.py 的数值逻辑,纯 numpy 实现,
避免 mlx_engine 对 torch 的依赖。这些是后处理/预处理,非 model forward。
"""
from __future__ import annotations

import numpy as np

__all__ = ["retrieve", "apply_cluster", "protect_units", "change_rms"]


def retrieve(
    index_vectors: np.ndarray,
    query: np.ndarray,
    k: int = 8,
    ratio: float = 0.75,
    eps: float = 1e-8,
    chunk: int = 8192,
) -> np.ndarray:
    """RVC/so-vits 特征检索:(1/squared-L2)^2 权重 topk 加权。

    镜像 usvc/retrieval/predict.py::retrieve(torch.cdist + topk)。
    """
    k = min(k, index_vectors.shape[0])
    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i : i + chunk].astype(np.float32)
        # squared L2: [chunk, N]
        # cdist^2 = ||q||^2 + ||v||^2 - 2 q.v
        d2 = (
            np.sum(q * q, axis=1, keepdims=True)
            + np.sum(index_vectors * index_vectors, axis=1)
            - 2.0 * q @ index_vectors.T
        )
        d2 = np.maximum(d2, 0.0)
        # topk smallest
        ix = np.argpartition(d2, k - 1, axis=1)[:, :k]
        score = np.take_along_axis(d2, ix, axis=1)
        weight = (1.0 / np.maximum(score, eps)) ** 2
        weight = weight / weight.sum(axis=1, keepdims=True)
        out = (index_vectors[ix] * weight[..., None]).sum(axis=1)  # [chunk, dim]
        results.append(ratio * out + (1.0 - ratio) * q)
    return np.concatenate(results, axis=0)


def apply_cluster(
    centers: np.ndarray,
    query: np.ndarray,
    ratio: float = 0.75,
    chunk: int = 8192,
) -> np.ndarray:
    """so-vits 聚类映射:argmin-L2 -> cluster center -> blend。"""
    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i : i + chunk].astype(np.float32)
        d2 = (
            np.sum(q * q, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)
            - 2.0 * q @ centers.T
        )
        idx = np.argmin(d2, axis=1)
        mapped = centers[idx]
        results.append(ratio * mapped + (1.0 - ratio) * q)
    return np.concatenate(results, axis=0)


def protect_units(
    units: np.ndarray,
    units_src: np.ndarray,
    rms: np.ndarray,
    protect: float,
) -> np.ndarray:
    """RVC protect:浊音帧保留增强 units,清音/静音帧混回 source。

    镜像 engine.py::_protect_units(pitchff = where(rms>1e-4, protect, 1.0))。
    """
    pitchff = np.where(rms > 1e-4, protect, 1.0)[:, None]  # [T, 1]
    return units * pitchff + units_src * (1.0 - pitchff)


def change_rms(
    src_wav: np.ndarray,
    src_sr: int,
    dst_wav: np.ndarray,
    dst_sr: int,
    rate: float,
) -> np.ndarray:
    """RVC rms_mix:把 dst 的 RMS 包络换成 src 的。

    镜像 engine.py::_change_rms(numpy,基于 librosa feature.rms)。
    rate=1.0 完全用 src 包络,rate=0.0 保持 dst 不变。
    """
    import librosa
    # dst 的 RMS 包络(逐帧)
    hop = 160  # 与 RVC 默认一致
    rms_dst = librosa.feature.rms(y=dst_wav, frame_length=hop * 4, hop_length=hop)[0]
    rms_src = librosa.feature.rms(
        y=librosa.resample(src_wav, orig_sr=src_sr, target_sr=dst_sr, res_type="soxr_hq"),
        frame_length=hop * 4, hop_length=hop,
    )[0]
    # 对齐长度
    n = min(len(rms_dst), len(rms_src))
    rms_dst, rms_src = rms_dst[:n], rms_src[:n]
    # 插值到 dst 长度
    xp = np.linspace(0, n - 1, n)
    x = np.linspace(0, n - 1, len(dst_wav))
    rms_dst_full = np.interp(x, xp, rms_dst)
    rms_src_full = np.interp(x, xp, rms_src)
    # 混合包络
    mixed = rms_src_full * rate + rms_dst_full * (1 - rate)
    # 应用:dst * (mixed / rms_dst),避免除零
    coeff = np.where(rms_dst_full > 1e-8, mixed / rms_dst_full, 1.0)
    return dst_wav * coeff
