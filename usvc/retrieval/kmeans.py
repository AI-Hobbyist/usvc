"""Pure-torch GPU KMeans, ported verbatim from so-vits-svc `cluster/kmeans.py`.

Replaces sklearn/faiss KMeans with a GPU implementation that is numerically
identical to the original project's (random-minibatch stochastic-gradient
updates, kmeans++ seeding, int16 labels). Used for:
  - sovits cluster model (kmeans_10000.pt equivalent)
  - large-index downsampling (>200k vectors -> 10k cluster centers)

Only change vs original: pynvml -> torch.cuda.mem_get_info() for free-memory
probe (zero extra dependency).
"""
from __future__ import annotations

from time import time

import numpy as np
import torch
from torch.nn.functional import normalize

__all__ = ["KMeansGPU", "_kpp"]


def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : Tensor
        A 'k' by 'N' containing the initial centroids.
    """
    batch_size = data.shape[0]
    if sample_size > 0 and batch_size > sample_size:
        data = data[torch.randint(0, batch_size, [sample_size], device=data.device)]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)
    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]
        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(cumprobs, r.sample([1]).to(data.device))]
    return init


class KMeansGPU:
    """Kmeans clustering algorithm implemented with PyTorch.

    Attributes:
        centroids: torch.Tensor, shape: [n_clusters, n_features]
    """

    def __init__(
        self,
        n_clusters,
        max_iter=200,
        tol=1e-4,
        verbose=0,
        mode="euclidean",
        device=torch.device("cuda:0"),
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.mode = mode
        self.device = torch.device(device)
        gpu_index = self.device.index if self.device.index is not None else 0
        free_bytes, _total = torch.cuda.mem_get_info(gpu_index)
        free_gb = free_bytes / 1024 / 1024 / 1024
        self.minibatch = int(33e6 / self.n_clusters * free_gb)
        if self.verbose:
            print("free_mem/GB:", round(free_gb, 2), "minibatch:", self.minibatch)

    @staticmethod
    def cos_sim(a, b):
        return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """Euclidean similarity (negative squared distance, up to a constant)."""
        return 2 * a @ b.transpose(-2, -1) - (a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

    def max_sim(self, a, b):
        if self.mode == "cosine":
            sim_func = self.cos_sim
        elif self.mode == "euclidean":
            sim_func = self.euc_sim
        sim = sim_func(a, b)
        max_sim_v, max_sim_i = sim.max(dim=-1)
        return max_sim_v, max_sim_i

    def fit_predict(self, X):
        """Combination of fit() and predict() methods (original-project math)."""
        assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
        assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        offset = np.power(1.5, np.log(self.n_clusters / 1000)) / np.log(2)
        with torch.no_grad():
            batch_size = X.shape[0]
            start_time = time()
            if self.minibatch * 10 // offset < batch_size:
                x = X[torch.randint(0, batch_size, [int(self.minibatch * 10 / offset)])].to(self.device)
            else:
                x = X.to(self.device)
            self.centroids = _kpp(x, self.n_clusters, min(int(self.minibatch / 12 / offset), batch_size))
            del x
            torch.cuda.empty_cache()
            num_points_in_clusters = torch.ones(self.n_clusters, device=self.device, dtype=X.dtype)
            closest = None
            if self.minibatch >= batch_size // 2 and self.minibatch < batch_size:
                X = X[torch.randint(0, batch_size, [self.minibatch])].to(self.device)
            elif self.minibatch >= batch_size:
                X = X.to(self.device)
            for i in range(self.max_iter):
                iter_time = time()
                if self.minibatch < batch_size // 2:  # minibatch too small, resample each iter
                    x = X[torch.randint(0, batch_size, [self.minibatch])].to(self.device)
                else:  # whole batch cached on GPU
                    x = X

                closest = self.max_sim(a=x, b=self.centroids)[1].to(torch.int16)
                matched_clusters, counts = closest.unique(return_counts=True)
                expanded_closest = closest[None].expand(self.n_clusters, -1)
                mask = (expanded_closest == torch.arange(self.n_clusters, device=self.device)[:, None]).to(X.dtype)
                c_grad = mask @ x / mask.sum(-1)[..., :, None]
                c_grad[c_grad != c_grad] = 0  # remove NaNs (empty clusters)
                error = (c_grad - self.centroids).pow(2).sum()
                if self.minibatch is not None:
                    lr = 1 / num_points_in_clusters[:, None] * 0.9 + 0.1
                else:
                    lr = 1
                matched_clusters = matched_clusters.long()
                num_points_in_clusters[matched_clusters] += counts
                self.centroids = self.centroids * (1 - lr) + c_grad * lr
                if self.verbose >= 2:
                    print("iter:", i, "error:", error.item(), "time spent:", round(time() - iter_time, 4))
                if error <= self.tol:
                    break

            if self.verbose >= 1:
                print(
                    f"used {i + 1} iterations ({round(time() - start_time, 4)}s) "
                    f"to cluster {batch_size} items into {self.n_clusters} clusters"
                )
        return closest
