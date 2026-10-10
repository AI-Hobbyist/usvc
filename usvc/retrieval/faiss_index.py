"""Optional CPU RVC native FlatL2/IVFFlat search; never substitute exact search for IVF."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


class NativeIndexCache:
    def __init__(self, max_bytes=1024 ** 3, max_entries=4):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def _drop(self, key):
        _, _, size = self.entries.pop(key)
        self.bytes -= size

    def load(self, path, dimension):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError('native RVC .index requires pip install "usvc[rvc-index]"') from exc
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        for old in list(self.entries):
            if old[0] == str(path) and old != key:
                self._drop(old)
        if key in self.entries:
            index, vectors, _ = self.entries[key]
            if index.d != dimension:
                raise ValueError("native index feature dimension mismatch")
            self.entries.move_to_end(key)
            return index, vectors
        if stat.st_size == 0:
            raise ValueError("native index file is empty")
        if stat.st_size > self.max_bytes or self.max_entries < 1:
            raise ValueError("native index exceeds CPU cache budget")
        while self.entries and (self.bytes + stat.st_size > self.max_bytes or len(self.entries) >= self.max_entries):
            self._drop(next(iter(self.entries)))
        # Read bytes ourselves: faiss.read_index has Windows Unicode path limitations.
        try:
            index = faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))
        except RuntimeError as exc:
            raise ValueError("corrupt or unsupported native index") from exc
        if type(index) not in (faiss.IndexFlatL2, faiss.IndexIVFFlat) or index.metric_type != faiss.METRIC_L2:
            raise ValueError("native index supports only L2 Flat/IVF Flat with sequential IDs")
        if not index.is_trained or index.ntotal < 1:
            raise ValueError("native index must be trained and nonempty")
        if index.d != dimension:
            raise ValueError("native index feature dimension mismatch")
        size = stat.st_size + index.ntotal * index.d * 4 + index.ntotal * 8
        if size > self.max_bytes:
            raise ValueError("native index reconstruction exceeds CPU cache budget")
        while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
            self._drop(next(iter(self.entries)))
        if isinstance(index, faiss.IndexIVFFlat):
            ids = []
            lists = index.invlists
            for i in range(index.nlist):
                count = lists.list_size(i)
                if count:
                    pointer = lists.get_ids(i)
                    try:
                        ids.append(faiss.rev_swig_ptr(pointer, count).copy())
                    finally:
                        lists.release_ids(i, pointer)
            if not ids or not np.array_equal(np.sort(np.concatenate(ids)), np.arange(index.ntotal)):
                raise ValueError("native IVF index must have sequential IDs 0..N-1")
        vectors = np.ascontiguousarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("native index contains nonfinite vectors")
        self.entries[key] = (index, vectors, size)
        self.bytes += size
        return index, vectors

    def retrieve(self, path, query, ratio=0.75, k=8, chunk=512):
        if not 0 <= ratio <= 1:
            raise ValueError("native index ratio must be in 0..1")
        if ratio == 0:
            return query
        if query.ndim != 2 or k < 1 or chunk < 1:
            raise ValueError("native retrieval requires [T,D] queries and positive k/chunk")
        index, vectors = self.load(path, query.shape[1])
        source = np.ascontiguousarray(query.detach().float().cpu().numpy())
        if not np.isfinite(source).all():
            raise ValueError("native retrieval query must be finite")
        result = source.copy()
        for start in range(0, len(source), chunk):
            q = source[start:start + chunk]
            distances, ids = index.search(q, min(k, index.ntotal))
            valid = (ids >= 0) & (ids < len(vectors)) & np.isfinite(distances) & (distances >= 0)
            safe_ids = np.where(valid, ids, 0)
            weights = np.where(valid, (1 / np.maximum(distances, 1e-8)) ** 2, 0)
            total = weights.sum(1, keepdims=True)
            weights /= np.where(total > 0, total, 1)
            retrieved = (vectors[safe_ids] * weights[..., None]).sum(1)
            usable = total[:, 0] > 0
            result[start:start + len(q)][usable] = ratio * retrieved[usable] + (1 - ratio) * q[usable]
        output = torch.from_numpy(result).to(device=query.device, dtype=query.dtype)
        if not torch.isfinite(output).all():
            raise ValueError("native retrieval result is not finite in the query dtype")
        return output
