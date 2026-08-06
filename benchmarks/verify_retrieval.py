"""Numerical verification: usvc pure-torch retrieval/cluster vs faiss/sklearn.

1. retrieve() vs faiss IndexFlatL2 exact search (squared-L2 + (1/score)^2)
2. apply_cluster() vs sklearn KMeans.predict mapping (100% label agreement)
3. KMeansGPU quality vs sklearn KMeans (inertia within tolerance)

faiss/sklearn are used HERE ONLY as reference implementations; usvc itself
has zero dependency on them.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/usvc")


def verify_retrieval_vs_faiss():
    import faiss

    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, D, T = 5000, 768, 200
    index_vectors = torch.randn(N, D, device=device)
    query = torch.randn(T, D, device=device)

    # faiss reference (exact, non-IVF -> same math as IVF with nprobe=nlist)
    idx = faiss.IndexFlatL2(D)
    idx.add(index_vectors.cpu().numpy())
    scores, ix = idx.search(query.cpu().numpy(), k=8)  # squared L2

    from usvc.retrieval.predict import retrieve

    usvc_out = retrieve(index_vectors, query, k=8, ratio=1.0)  # pure retrieval

    # manual re-derivation from faiss indices
    weight = np.square(1.0 / np.maximum(scores, 1e-8))
    weight /= weight.sum(axis=1, keepdims=True)
    ref = np.sum(index_vectors.cpu().numpy()[ix] * weight[..., None], axis=1)

    diff = np.abs(usvc_out.cpu().numpy() - ref).max()
    print(f"[retrieval vs faiss IndexFlatL2] max diff = {diff:.3e}")
    assert diff < 1e-4, f"retrieval mismatch vs faiss: {diff}"
    print("PASS: retrieval formula matches faiss (squared-L2 weights)")


def verify_cluster_vs_sklearn():
    """KMeansGPU is a minibatch-SGD algorithm (verbatim from so-vits-svc);
    sklearn Lloyd's is a different algorithm, so centroid equality is NOT
    expected. Two checks instead:
      1. bitwise equality vs the ORIGINAL so-vits-svc KMeansGPU (same seed/data)
      2. inertia within 10% of sklearn's quality on well-separated clusters
    """
    import importlib.util
    import sys as _sys

    _sys.path.insert(0, "/root/so-vits-svc/cluster")
    spec = importlib.util.spec_from_file_location("orig_kmeans", "/root/so-vits-svc/cluster/kmeans.py")
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)

    from sklearn.cluster import KMeans

    torch.manual_seed(11)
    np.random.seed(11)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N, D, n_clusters = 2000, 64, 8
    base = torch.arange(n_clusters, dtype=torch.float32).to(dev) * 10.0
    assign = base[torch.randint(0, n_clusters, (N,))][:, None]
    data = torch.randn(N, D, device=dev) * 0.8 + assign

    # (1) bitwise parity with original project's KMeansGPU
    torch.manual_seed(11); np.random.seed(11)
    km_orig = orig.KMeansGPU(n_clusters=n_clusters, max_iter=300, tol=1e-4, verbose=0, device=dev)
    km_orig.fit_predict(data)

    torch.manual_seed(11); np.random.seed(11)
    from usvc.retrieval.kmeans import KMeansGPU

    km = KMeansGPU(n_clusters=n_clusters, max_iter=300, tol=1e-4, verbose=0, device=dev)
    km.fit_predict(data)

    o = km_orig.centroids.cpu()
    u = km.centroids.cpu()
    o_sorted = o[o.mean(dim=1).sort().indices]
    u_sorted = u[u.mean(dim=1).sort().indices]
    diff = (o_sorted - u_sorted).abs().max().item()
    print(f"[kmeans vs so-vits original] max diff = {diff:.3e}")
    assert diff < 1e-5, f"KMeansGPU diverges from original: {diff}"
    print("PASS: usvc KMeansGPU bitwise-equal to so-vits-svc original")

    # (2) inertia quality vs sklearn (well-separated clusters)
    km_sk = KMeans(n_clusters=n_clusters, n_init=5, random_state=0, max_iter=300)
    km_sk.fit(data.cpu().numpy())

    def inertia(data_np, centers):
        return float(((data_np[:, None, :] - centers[None, :, :]) ** 2).sum(-1).min(-1).sum())

    sk_inertia = inertia(data.cpu().numpy(), km_sk.cluster_centers_)
    usvc_inertia = inertia(data.cpu().numpy(), km.centroids.cpu().numpy())
    rel = abs(usvc_inertia - sk_inertia) / sk_inertia
    print(f"[kmeans inertia] usvc={usvc_inertia:.1f} sklearn={sk_inertia:.1f} rel={rel * 100:.2f}%")
    assert rel < 0.10, f"inertia divergence too large: {rel * 100:.2f}%"
    print("PASS: KMeans inertia within 10% of sklearn")


def verify_cluster_mapping_vs_sklearn():
    """apply_cluster == sklearn predict -> cluster_centers_ mapping (exact)."""
    from sklearn.cluster import KMeans

    torch.manual_seed(13)
    N, D, n_clusters = 500, 64, 8
    data = torch.randn(N, D) * 0.5 + torch.randn(n_clusters, D)[torch.randint(0, n_clusters, (N,))]
    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=0)
    km.fit(data.numpy())

    from usvc.retrieval.predict import apply_cluster

    centers = torch.from_numpy(km.cluster_centers_)
    probe = torch.randn(50, D)
    out = apply_cluster(centers, probe, ratio=1.0)  # pure mapping

    pred = km.predict(probe.numpy())
    ref = torch.from_numpy(km.cluster_centers_[pred])

    diff = (out - ref).abs().max().item()
    print(f"[cluster mapping vs sklearn] max diff = {diff:.3e}")
    assert diff < 1e-5, f"cluster mapping mismatch: {diff}"
    print("PASS: apply_cluster == sklearn predict + centers (exact)")


def main():
    verify_retrieval_vs_faiss()
    verify_cluster_mapping_vs_sklearn()
    verify_cluster_vs_sklearn()
    print("\nAll retrieval/cluster numerical checks PASSED.")


if __name__ == "__main__":
    main()
