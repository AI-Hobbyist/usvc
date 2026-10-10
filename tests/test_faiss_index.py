"""Native FAISS acceptance: fixtures plus unchanged local pretrained RVC resources."""
from pathlib import Path
import gc
import hashlib
import sys

import faiss
import numpy as np
import pytest
import torch
import soundfile as sf

from usvc.retrieval.faiss_index import NativeIndexCache
from usvc.infer.engine import InferenceEngine
from test_multispeaker_infer import archive_audio

ROOT = Path(__file__).resolve().parents[1]


def write_index(tmp_path, index):
    path = tmp_path / "中文 索引.index"
    path.write_bytes(faiss.serialize_index(index).tobytes())
    return path


def reference(index, vectors, q, ratio):
    distances, ids = index.search(np.ascontiguousarray(q, dtype=np.float32), min(8, index.ntotal))
    valid = (ids >= 0) & (ids < index.ntotal) & np.isfinite(distances) & (distances >= 0)
    weights = np.where(valid, np.square(1 / np.maximum(distances, 1e-8)), 0)
    sums = weights.sum(1, keepdims=True)
    weights /= np.where(sums > 0, sums, 1)
    retrieved = (vectors[np.where(valid, ids, 0)] * weights[..., None]).sum(1)
    return np.where(sums > 0, ratio * retrieved + (1 - ratio) * q, q)


@pytest.mark.parametrize("ivf", [False, True])
def test_native_search_and_cache(tmp_path, ivf):
    rng = np.random.default_rng(321)
    vectors = rng.normal(size=(200, 16)).astype("float32")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(16), 16, 4) if ivf else faiss.IndexFlatL2(16)
    if ivf:
        index.train(vectors)
        index.nprobe = 2
    index.add(vectors)
    path = write_index(tmp_path, index)
    q = torch.from_numpy(np.concatenate((vectors[:1], rng.normal(size=(18, 16)).astype("float32"))))
    cache = NativeIndexCache()
    for ratio in (0, .5, 1):
        actual = cache.retrieve(path, q, ratio=ratio, chunk=3)
        np.testing.assert_allclose(actual.numpy(), reference(index, vectors, q.numpy(), ratio), atol=1e-6)
        assert torch.isfinite(actual).all()
    loaded, _ = cache.load(path, 16)
    assert cache.load(path, 16)[0] is loaded
    if ivf:
        assert loaded.nprobe == 2
    replacement = faiss.IndexFlatL2(16)
    replacement.add(vectors[:3])
    write_index(tmp_path, replacement)
    assert cache.load(path, 16)[0] is not loaded
    assert len(cache.entries) == 1
    with pytest.raises(ValueError, match="dimension"):
        cache.load(path, 256)
    with pytest.raises(ValueError, match="budget"):
        NativeIndexCache(max_bytes=path.stat().st_size + 1).load(path, 16)
    with pytest.raises(ValueError, match="budget"):
        NativeIndexCache(max_bytes=1).load(path, 16)
    cache.clear()
    assert cache.bytes == 0 and not cache.entries


def test_missing_neighbors_and_small_corpus(tmp_path):
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(2), 2, 2)
    training = np.concatenate((np.zeros((40, 2)), np.full((40, 2), 100))).astype("float32")
    index.train(training)
    index.add(np.zeros((2, 2), dtype="float32"))
    path = write_index(tmp_path, index)
    query = torch.tensor([[0., 0.], [100., 100.], [1., 1.]], dtype=torch.float16)
    cache = NativeIndexCache()
    out = cache.retrieve(path, query, ratio=1)
    assert torch.equal(out[1], query[1])  # no IDs in the queried IVF list
    assert torch.equal(out[0], query[0])
    assert torch.isfinite(out).all() and out.dtype == query.dtype
    flat = faiss.IndexFlatL2(2)
    flat.add(np.ones((1, 2), dtype="float32"))
    write_index(tmp_path, flat)
    assert torch.equal(cache.retrieve(path, query, ratio=1), torch.ones_like(query))


@pytest.mark.parametrize("kind", ["emptyfile", "corrupt", "empty", "untrained", "ip", "pq", "ids"])
def test_invalid_native_indexes(tmp_path, kind):
    vectors = np.random.default_rng(0).normal(size=(100, 8)).astype("float32")
    path = tmp_path / "bad.index"
    if kind in ("emptyfile", "corrupt"):
        path.write_bytes(b"" if kind == "emptyfile" else b"broken index")
    else:
        if kind == "ip":
            index = faiss.IndexFlatIP(8)
        elif kind == "pq":
            index = faiss.IndexPQ(8, 2, 2)
            index.train(vectors)
        elif kind in ("untrained", "ids"):
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 2)
            if kind == "ids":
                index.train(vectors)
                index.add_with_ids(vectors, np.arange(100, 200, dtype="int64"))
        else:
            index = faiss.IndexFlatL2(8)
        if kind in ("ip", "pq"):
            index.add(vectors)
        path = write_index(tmp_path, index)
    with pytest.raises(ValueError):
        NativeIndexCache().load(path, 8)


def test_cache_eviction(tmp_path):
    cache = NativeIndexCache(max_entries=1)
    index = faiss.IndexFlatL2(2)
    index.add(np.ones((5, 2), dtype="float32"))
    p1 = write_index(tmp_path, index)
    cache.load(p1, 2)
    p2 = tmp_path / "second.index"
    p2.write_bytes(p1.read_bytes())
    cache.load(p2, 2)
    assert len(cache.entries) == 1 and next(iter(cache.entries))[0] == str(p2.resolve())


def test_no_faiss_import_and_pt_engine(tmp_path):
    script = tmp_path / "without_faiss.py"
    script.write_text('''import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'faiss': raise ModuleNotFoundError('isolated faiss absence')
sys.meta_path.insert(0, Block())
import torch
from usvc.infer.engine import InferenceEngine
from usvc.retrieval.faiss_index import NativeIndexCache
e = InferenceEngine.__new__(InferenceEngine)
e.backend_name, e.device = 'rvc', 'cpu'
q = torch.randn(3, 8)
torch.save({'vectors': q.clone()}, sys.argv[1])
assert torch.isfinite(e._enhance_units(q, sys.argv[1], .5)).all()
assert NativeIndexCache().retrieve('missing.index', q, ratio=0) is q
try: NativeIndexCache().retrieve('missing.index', q, ratio=1)
except ImportError as ex: assert 'rvc-index' in str(ex)
else: raise AssertionError('missing dependency not rejected')
assert 'faiss' not in sys.modules
''', encoding="utf-8")
    # Existing foreground CLI runner accepts run_path via a tiny entry module.
    import subprocess
    def ps_quote(value):
        return "'" + str(value).replace("'", "''") + "'"
    command = f"& {ps_quote(sys.executable)} {ps_quote(script)} {ps_quote(tmp_path / 'old.pt')} 2>&1 | Tee-Object -FilePath {ps_quote(tmp_path / 'build.log')}; $code=$LASTEXITCODE; if ($code -ne 0) {{ Get-Content {ps_quote(tmp_path / 'build.log')}; exit $code }}"
    completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], cwd=ROOT)
    assert completed.returncode == 0, (tmp_path / "build.log").read_text(errors="replace")


@pytest.mark.parametrize("directory", ["guanguanV1", "keruanV1", "kikiV1", "youzhanv2-xi", "芙宁娜"])
def test_real_rvc_models(tmp_path, directory):
    assert torch.cuda.is_available(), "M4 local acceptance requires CUDA"
    folder = ROOT / "models" / "rvc" / directory
    checkpoint = next(folder.glob("*.pth"))
    index_path = next(folder.glob("*.index"), None)
    resources = [checkpoint, folder / "config.yaml"] + ([index_path] if index_path else [])
    original_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in resources]
    class ObservedEngine(InferenceEngine):
        def _enhance_units(self, units, index_path=None, index_ratio=0., *args, **kw):
            out = super()._enhance_units(units, index_path, index_ratio, *args, **kw)
            self.query, self.enhanced = units.detach().cpu(), out.detach().cpu()
            return out
    engine = ObservedEngine(folder / "config.yaml", checkpoint, device="cuda", half=False)
    fixed_input = ROOT / "tests" / "input" / "1.wav"
    baseline = None
    for ratio in ((0, .5, 1) if index_path else (0,)):
        torch.manual_seed(123)
        np.random.seed(123)
        output = engine.convert(fixed_input, spk=0, index_path=index_path, index_ratio=ratio)
        assert np.isfinite(output).all() and len(output) > engine.sr
        if baseline is None:
            baseline = engine.query
        else:
            torch.testing.assert_close(engine.query, baseline, rtol=0, atol=0)
        if index_path and ratio > 0:
            native = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype="uint8"))
            vectors = native.reconstruct_n(0, native.ntotal)
            expected = reference(native, vectors, engine.query.numpy(), ratio)
            np.testing.assert_allclose(engine.enhanced.numpy(), expected, atol=2e-6, rtol=2e-6)
            assert not torch.equal(engine.query, engine.enhanced)
            cached, _ = engine._native_index_cache.load(index_path, engine.query.shape[1])
            if isinstance(native, faiss.IndexIVFFlat):
                assert cached.nprobe == native.nprobe
        else:
            assert torch.equal(engine.query, engine.enhanced)
        path = tmp_path / f"ratio{ratio}.wav"
        sf.write(path, output, engine.sr)
        archive_audio(path, dict(backend="rvc", model=str(checkpoint), target_id=0, device="cuda",
                                index=str(index_path), index_ratio=ratio, seed=123,
                                input_sha256=hashlib.sha256(fixed_input.read_bytes()).hexdigest()))
    assert original_hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in resources]
    del engine
    gc.collect()
    torch.cuda.empty_cache()
