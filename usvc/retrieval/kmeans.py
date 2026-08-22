"""Pure-torch GPU KMeans, ported verbatim from so-vits-svc cluster/kmeans.py (numerically
identical: kmeans++ seeding, random-minibatch stochastic updates, int16 labels). Used for the
sovits cluster model and large-index downsampling (>200k vectors -> 10k centers).
Only delta vs original: free-memory probe via torch.cuda.mem_get_info (no pynvml dep)."""
from __future__ import annotations

from math import log, pow
from time import time

import torch
from torch.nn.functional import normalize


def _kpp(data, k, sample_size=-1):  # kmeans++ seeding
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
    def __init__(self, n_clusters, max_iter=200, tol=1e-4, verbose=0, mode="euclidean",
                 device=torch.device("cuda:0")):
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
        if self.verbose: print("free_mem/GB:", round(free_gb, 2), "minibatch:", self.minibatch)

    @staticmethod
    def cos_sim(a, b): return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):  # negative squared distance, up to a constant
        return 2 * a @ b.transpose(-2, -1) - (a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

    def max_sim(self, a, b):
        sim = {"cosine": self.cos_sim, "euclidean": self.euc_sim}[self.mode](a, b)
        return sim.max(dim=-1)

    def fit_predict(self, X):  # original-project math; fit+predict combined
        assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
        assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        offset = pow(1.5, log(self.n_clusters / 1000)) / log(2)  # scalar libm == np.power/np.log
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
                if self.minibatch < batch_size // 2:  # minibatch too small -> resample each iter
                    x = X[torch.randint(0, batch_size, [self.minibatch])].to(self.device)
                else:  # whole batch cached on GPU
                    x = X

                closest = self.max_sim(a=x, b=self.centroids)[1].to(torch.int16)
                matched_clusters, counts = closest.unique(return_counts=True)
                expanded_closest = closest[None].expand(self.n_clusters, -1)
                mask = (expanded_closest == torch.arange(self.n_clusters, device=self.device)[:, None]).to(X.dtype)
                c_grad = mask @ x / mask.sum(-1)[..., :, None]
                c_grad[c_grad != c_grad] = 0  # empty clusters -> NaN
                error = (c_grad - self.centroids).pow(2).sum()
                lr = 1 / num_points_in_clusters[:, None] * 0.9 + 0.1 if self.minibatch is not None else 1
                num_points_in_clusters[matched_clusters.long()] += counts
                self.centroids = self.centroids * (1 - lr) + c_grad * lr
                if self.verbose >= 2:
                    print("iter:", i, "error:", error.item(), "time spent:", round(time() - iter_time, 4))
                if error <= self.tol: break

            if self.verbose >= 1:
                print(f"used {i + 1} iterations ({round(time() - start_time, 4)}s) "
                      f"to cluster {batch_size} items into {self.n_clusters} clusters")
        return closest
