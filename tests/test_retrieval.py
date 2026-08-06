"""Tests for usvc.retrieval: pure-torch index/cluster (no faiss/sklearn import)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from usvc.retrieval.build_cluster import ClusterTrainer
from usvc.retrieval.build_index import IndexBuilder
from usvc.retrieval.kmeans import KMeansGPU
from usvc.retrieval.predict import apply_cluster, retrieve


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
def test_retrieve_matches_squared_l2_formula():
    """retrieve() == manual (squared-L2 topk + (1/score)^2 weights) formula."""
    torch.manual_seed(42)
    idx = torch.randn(1000, 768, device=_device())
    q = torch.randn(50, 768, device=_device())

    result = retrieve(idx, q, k=8, ratio=0.75)

    d2 = torch.cdist(q, idx).pow(2)
    score, ix = d2.topk(8, dim=1, largest=False)
    weight = (1.0 / score.clamp_min(1e-8)).pow(2)
    weight = weight / weight.sum(dim=1, keepdim=True)
    ref = (idx[ix] * weight.unsqueeze(-1)).sum(dim=1)
    ref = 0.75 * ref + 0.25 * q

    diff = (result - ref).abs().max().item()
    assert diff < 1e-5, f"retrieval formula mismatch: diff={diff:.2e}"


def test_retrieve_chunked_consistent():
    """Chunked retrieval == unchunked (bitwise-close)."""
    torch.manual_seed(42)
    idx = torch.randn(1000, 256, device=_device())
    q = torch.randn(200, 256, device=_device())

    r_full = retrieve(idx, q, k=8, ratio=0.5, chunk=10000)
    r_chunk = retrieve(idx, q, k=8, ratio=0.5, chunk=50)
    diff = (r_full - r_chunk).abs().max().item()
    assert diff < 1e-5, f"chunked mismatch: diff={diff:.2e}"


def test_retrieve_edge_small_k():
    """k > corpus size clamps to corpus size (no crash)."""
    idx = torch.randn(4, 16, device=_device())
    q = torch.randn(3, 16, device=_device())
    out = retrieve(idx, q, k=8, ratio=1.0)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_apply_cluster_matches_argmin():
    """apply_cluster == argmin-L2 -> center -> blend."""
    torch.manual_seed(42)
    centers = torch.randn(100, 768, device=_device())
    q = torch.randn(50, 768, device=_device())

    result = apply_cluster(centers, q, ratio=0.5)

    dist = torch.cdist(q, centers)
    idx = dist.argmin(dim=1)
    ref = 0.5 * centers[idx] + 0.5 * q

    diff = (result - ref).abs().max().item()
    assert diff < 1e-5, f"cluster mapping mismatch: diff={diff:.2e}"


def test_kmeans_converges():
    """KMeansGPU finds 3 well-separated clusters."""
    torch.manual_seed(42)
    base = torch.tensor([5.0, -5.0, 0.0])
    data = torch.cat([torch.randn(300, 64) + base[i] for i in range(3)]).to(_device())

    km = KMeansGPU(n_clusters=3, max_iter=200, tol=1e-4, verbose=0, device=_device())
    km.fit_predict(data)

    means = km.centroids.mean(dim=1).sort().values
    assert abs(means[0].item() - (-5.0)) < 1.0, f"cluster0 misplaced: {means.tolist()}"
    assert abs(means[1].item() - 0.0) < 1.0, f"cluster1 misplaced: {means.tolist()}"
    assert abs(means[2].item() - 5.0) < 1.0, f"cluster2 misplaced: {means.tolist()}"


def _make_fake_dataset(root: Path, n_speakers: int = 2, n_files: int = 4):
    """Minimal units-only dataset (no meta.json needed for builders)."""
    for si in range(n_speakers):
        spk = f"spk{si + 1}"
        d = root / "units" / "contentvec" / "abc12345" / spk
        d.mkdir(parents=True, exist_ok=True)
        for fi in range(n_files):
            torch.save(torch.randn(100, 768), d / f"file{fi}.pt")
    return root


def test_index_builder_integration():
    """Build per-speaker + global indexes from a fake units corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_fake_dataset(Path(tmp))
        # per-speaker
        b = IndexBuilder(ds, "contentvec", "abc12345")
        outs = b.build(device=_device(), per_speaker=True)
        assert len(outs) == 2
        for p in outs:
            payload = torch.load(p, weights_only=False)
            assert payload["kind"] == "units_index"
            assert payload["vectors"].shape[1] == 768
            assert payload["vectors"].shape[0] == 4 * 100
            assert payload["units"]["config_hash"] == "abc12345"
        # global
        outs = b.build(device=_device(), per_speaker=False)
        assert len(outs) == 1
        payload = torch.load(outs[0], weights_only=False)
        assert payload["spk_id"] is None
        assert payload["vectors"].shape[0] == 800


def test_index_builder_downsample():
    """Corpus above threshold gets compressed to n_clusters centroids."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_fake_dataset(Path(tmp), n_speakers=1, n_files=4)
        b = IndexBuilder(ds, "contentvec", "abc12345",
                         n_clusters_downsample=8, downsample_threshold=50)
        outs = b.build(device=_device(), per_speaker=True)
        payload = torch.load(outs[0], weights_only=False)
        assert payload["vectors"].shape[0] == 8
        assert payload["config"]["downsample"]["applied"] is True
        assert payload["stats"]["n_total"] == 400


def test_cluster_builder_integration():
    """Train per-speaker cluster model."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_fake_dataset(Path(tmp))
        t = ClusterTrainer(ds, "contentvec", "abc12345", n_clusters=8, max_iter=20)
        out = t.build(device=_device())
        payload = torch.load(out, weights_only=False)
        assert payload["kind"] == "units_cluster"
        assert payload["n_clusters"] == 8
        for spk in ("spk1", "spk2"):
            assert spk in payload
            assert payload[spk]["cluster_centers_"].shape == (8, 768)
            assert payload[spk]["n_features_in_"] == 768


def test_pyproject_no_faiss_dependency():
    """pyproject.toml must not require faiss."""
    content = Path("pyproject.toml").read_text()
    assert "faiss" not in content, "pyproject.toml still references faiss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
