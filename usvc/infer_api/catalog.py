"""Filesystem contracts and CPU-only availability records; validators own execution."""
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import threading

from ..train.config import TrainConfig
from ..speakers import load_model_speakers, validate_mapping, resolve_speaker
from .errors import APIError

BACKENDS = ("ddsp6", "diffusion", "reflow_vae", "rift_dit", "rvc", "sovits")
LOG = logging.getLogger(__name__)


def contained(path, roots):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f"resource escapes allowed roots: {path}")
    if not path.is_file():
        raise ValueError(f"missing resource: {path}")
    return path


def choose(directory, explicit, candidates, label, roots):
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError(f"invalid {label} path")
        return contained(directory / explicit, roots)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f"{label} requires exactly one candidate or model.json selection")
    return contained(candidates[0], roots)


@dataclass
class Model:
    id: str
    backend: str
    directory: Path
    config: Path
    checkpoint: Path
    cfg: TrainConfig
    metadata: dict
    speakers: list
    indexes: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)
    default_index: str | None = None
    default_speaker: int | None = None
    fingerprint: tuple = ()

    @property
    def fixture(self):
        return self.metadata.get("test_fixture") is True

    def public(self):
        return dict(id=self.id, backend=self.backend, role="model", name=self.directory.name,
                    status="ready", test_fixture=self.fixture,
                    fixture_output=self.metadata.get("fixture_output") if self.fixture else None,
                    multi_speaker=len(self.speakers) > 1, speakers=self.speakers,
                    default_speaker=self.default_speaker, default_index=self.default_index,
                    indexes=list(self.indexes))


class Catalog:
    def __init__(self, root, validator, shared_roots=()):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("model root must be a directory")
        self.roots = (self.root, *(Path(p).resolve() for p in shared_roots))
        self.validator = validator
        self.models, self.diagnostics, self.cache = {}, {}, {}
        self.lock = threading.RLock()

    def _read(self, directory, backend):
        contained(directory / "model.json", self.roots) if (directory / "model.json").exists() else None
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8")) if (directory / "model.json").exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError("model.json must be an object")
        config = choose(directory, metadata.get("config"), [p for p in directory.iterdir() if p.suffix.lower() in (".yaml", ".yml")], "config", self.roots)
        checkpoint = choose(directory, metadata.get("checkpoint"), [p for p in directory.iterdir() if p.suffix.lower() in (".ckpt", ".pth", ".pt") and not any(x in p.stem.lower() for x in ("index", "cluster", "vocoder"))], "checkpoint", self.roots)
        cfg = TrainConfig.from_yaml(config)
        if cfg.backend != backend:
            raise ValueError("config backend does not match directory")
        if cfg.spk == {}:
            cfg.spk = None
        declared = cfg.spk
        if backend in ("ddsp6", "sovits"):
            mapping, capacity = load_model_speakers(cfg, checkpoint)
            if declared is None:
                if capacity != 1:
                    raise ValueError("undeclared multi-speaker checkpoint requires spk configuration")
                mapping = {"default": 0}
            else:
                mapping = validate_mapping(declared)
        else:
            mapping = validate_mapping(declared) if declared is not None else {"default": 0}
            if len(mapping) != 1:
                raise ValueError("this backend exposes one logical speaker")
        default = metadata.get("default_speaker")
        if default is not None:
            default = resolve_speaker(default, mapping, len(mapping))
        elif len(mapping) == 1:
            default = 0
        if metadata.get("test_fixture") is True and metadata.get("fixture_output") != "source_f0_sine":
            raise ValueError("unsupported fixture output")
        model = Model(f"{backend}/{directory.name}", backend, directory, config, checkpoint, cfg, metadata,
                      [{"id": i, "name": name} for name, i in sorted(mapping.items(), key=lambda x: x[1])], default_speaker=default)
        # Resource validators are shared with inference; malformed optional resources are omitted.
        from ..retrieval.faiss_index import NativeIndexCache
        import torch
        dimension = cfg.backend_opts.get("n_units", cfg.backend_opts.get("cvec_dim", 768))
        for path in sorted(directory.iterdir()):
            if path == checkpoint or not path.is_file():
                continue
            try:
                if path.suffix == ".index" and backend == "rvc":
                    NativeIndexCache().load(contained(path, self.roots), dimension)
                    model.indexes[path.name] = path.resolve()
                elif path.suffix == ".pt" and "index" in path.stem.lower():
                    obj = torch.load(contained(path, self.roots), map_location="cpu", weights_only=False)
                    vectors = obj.get("vectors") if isinstance(obj, dict) else obj
                    if not torch.is_tensor(vectors) or vectors.ndim != 2 or vectors.shape[1] != dimension or not len(vectors) or not torch.isfinite(vectors).all():
                        raise ValueError("invalid index vectors")
                    model.indexes[path.name] = path.resolve()
            except (ValueError, RuntimeError, ImportError, OSError) as exc:
                LOG.warning("Ignoring resource %s: %s", path, exc)
        selected = metadata.get("default_index")
        if selected is not None:
            if selected not in model.indexes:
                raise ValueError("default_index is not a valid discovered index")
            model.default_index = selected
        elif len(model.indexes) == 1:
            model.default_index = next(iter(model.indexes))
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        model.fingerprint = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
        return model

    def refresh(self):
        with self.lock:
            models, diagnostics = {}, {}
            for backend_dir in sorted(self.root.iterdir()):
                if not backend_dir.is_dir():
                    continue
                if backend_dir.name not in BACKENDS:
                    diagnostics[backend_dir.name] = "not an independently inferable backend"
                    continue
                for directory in sorted(backend_dir.iterdir()):
                    if not directory.is_dir():
                        continue
                    identity = f"{backend_dir.name}/{directory.name}"
                    try:
                        model = self._read(directory, backend_dir.name)
                        cached = self.cache.get(identity)
                        if cached != model.fingerprint:
                            self.validator(model)
                            self.cache[identity] = model.fingerprint
                        models[identity] = model
                    except Exception as exc:
                        diagnostics[identity] = str(exc)
                        self.cache.pop(identity, None)
                        LOG.warning("Excluded %s: %s", identity, exc)
            self.models, self.diagnostics = models, diagnostics
            return [model.public() for _, model in sorted(models.items())]

    def get(self, identity):
        if identity not in self.models:
            raise APIError(404, "unknown_model", "Unknown or unavailable model", "model")
        return self.models[identity]

    def invalidate(self, identity):
        with self.lock:
            self.cache.pop(identity, None)
            self.models.pop(identity, None)
