"""mlx_engine convert 高级后处理:index/cluster 检索 + protect + rms_mix
(纯 numpy,镜像 torch retrieval/predict.py 与 engine.py,无 torch 依赖)。"""
from __future__ import annotations

import numpy as np

__all__ = ["retrieve", "apply_cluster", "protect_units", "change_rms"]


def _d2(q, refs):
    """squared L2 [chunk, N]: ||q||^2 + ||v||^2 - 2 q.v,clamp ≥0。"""
    d2 = (np.sum(q * q, axis=1, keepdims=True)
          + np.sum(refs * refs, axis=1) - 2.0 * q @ refs.T)
    return np.maximum(d2, 0.0)


def retrieve(index_vectors: np.ndarray, query: np.ndarray, k: int = 8,
             ratio: float = 0.75, eps: float = 1e-8, chunk: int = 8192) -> np.ndarray:
    """特征检索:(1/squared-L2)^2 权重 topk 加权 blend(镜像 predict.py)。"""
    k = min(k, index_vectors.shape[0])
    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk].astype(np.float32)
        d2 = _d2(q, index_vectors)
        ix = np.argpartition(d2, k - 1, axis=1)[:, :k]
        score = np.take_along_axis(d2, ix, axis=1)
        weight = (1.0 / np.maximum(score, eps)) ** 2
        weight = weight / weight.sum(axis=1, keepdims=True)
        out = (index_vectors[ix] * weight[..., None]).sum(axis=1)  # [chunk, dim]
        results.append(ratio * out + (1.0 - ratio) * q)
    return np.concatenate(results, axis=0)


def apply_cluster(centers: np.ndarray, query: np.ndarray, ratio: float = 0.75,
                  chunk: int = 8192) -> np.ndarray:
    """so-vits 聚类映射:argmin-L2 → cluster center → blend。"""
    results = []
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk].astype(np.float32)
        idx = np.argmin(_d2(q, centers), axis=1)
        results.append(ratio * centers[idx] + (1.0 - ratio) * q)
    return np.concatenate(results, axis=0)


def protect_units(units: np.ndarray, units_src: np.ndarray, rms: np.ndarray,
                  protect: float) -> np.ndarray:
    """RVC protect:浊音帧(rms>1e-4)保留增强 units,清音/静音混回 source。"""
    pitchff = np.where(rms > 1e-4, protect, 1.0)[:, None]  # [T, 1]
    return units * pitchff + units_src * (1.0 - pitchff)


def change_rms(src_wav: np.ndarray, src_sr: int, dst_wav: np.ndarray,
               dst_sr: int, rate: float) -> np.ndarray:
    """RVC rms_mix:dst 的 RMS 包络换成 src 的(rate=1 全用 src,0 保持 dst)。"""
    import librosa
    hop = 160  # RVC 默认
    rms_dst = librosa.feature.rms(y=dst_wav, frame_length=hop * 4, hop_length=hop)[0]
    rms_src = librosa.feature.rms(
        y=librosa.resample(src_wav, orig_sr=src_sr, target_sr=dst_sr, res_type="soxr_hq"),
        frame_length=hop * 4, hop_length=hop,
    )[0]
    n = min(len(rms_dst), len(rms_src))
    rms_dst, rms_src = rms_dst[:n], rms_src[:n]
    xp = np.linspace(0, n - 1, n)
    x = np.linspace(0, n - 1, len(dst_wav))
    mixed = (np.interp(x, xp, rms_src) * rate
             + np.interp(x, xp, rms_dst) * (1 - rate))
    rms_dst_full = np.interp(x, xp, rms_dst)
    coeff = np.where(rms_dst_full > 1e-8, mixed / rms_dst_full, 1.0)
    return dst_wav * coeff
