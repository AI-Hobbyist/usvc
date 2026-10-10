"""Training configuration. TrainConfig is the single entry point (loaded from YAML, backend-agnostic).

Field names + yaml keys are a frozen contract: configs/ yaml files pin training semantics.
"""
import yaml


class OptimizerConfig:
    def __init__(self, type="adamw", lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 muon_momentum=0.95, muon_weight_decay=0.1):
        # type: 'schedulefree' | 'muon_adamw' | 'adamw'
        self.type, self.lr, self.betas, self.eps, self.weight_decay = type, lr, betas, eps, weight_decay
        self.muon_momentum, self.muon_weight_decay = muon_momentum, muon_weight_decay

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(d.get("type", "adamw"), d.get("lr", 1e-4), tuple(d.get("betas", (0.9, 0.999))),
                   d.get("eps", 1e-8), d.get("weight_decay", 0.0),
                   d.get("muon_momentum", 0.95), d.get("muon_weight_decay", 0.1))


class ScheduleConfig:
    def __init__(self, type=None, warmup_steps=0, decay_step=100000, gamma=0.5, min_lr=0.0):
        # type: None | 'step' | 'exponential' | 'linear_warmup_decay'
        self.type, self.warmup_steps, self.decay_step = type, warmup_steps, decay_step
        self.gamma, self.min_lr = gamma, min_lr

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls()
        return cls(d.get("type"), d.get("warmup_steps", 0), d.get("decay_step", 100000),
                   d.get("gamma", 0.5), d.get("min_lr", 0.0))


class TrainConfig:
    def __init__(self, backend="rift_dit", dataset_dir="", exp_dir="experiments/default", val_dir=None,
                 batch_size=32, max_frames=256, num_workers=8, augment=True,
                 optimizer=None, schedule=None, grad_clip=1.0, precision="bf16-mixed",
                 max_epochs=None, max_steps=500_000, val_interval=2000, save_interval=2000,
                 limit_val_batches=None, resume=None, pretrained=None, keep_last_n=5,
                 backend_opts=None, feature_hashes=None, spk=None):
        self.backend, self.dataset_dir, self.exp_dir, self.val_dir = backend, dataset_dir, exp_dir, val_dir
        self.batch_size, self.max_frames, self.num_workers, self.augment = batch_size, max_frames, num_workers, augment
        self.optimizer = optimizer if optimizer is not None else OptimizerConfig()
        self.schedule = schedule if schedule is not None else ScheduleConfig()
        self.grad_clip, self.precision = grad_clip, precision
        self.max_epochs, self.max_steps, self.val_interval, self.save_interval = max_epochs, max_steps, val_interval, save_interval
        self.limit_val_batches, self.resume, self.pretrained = limit_val_batches, resume, pretrained
        self.keep_last_n = keep_last_n
        self.backend_opts = backend_opts if backend_opts is not None else {}
        self.feature_hashes = feature_hashes if feature_hashes is not None else {}
        self.spk = spk

    # Throughput-optimal defaults per backend (empirically tuned on A10G 24GB).
    # flow/diffusion: AMP bf16 safe (<0.03% divergence). GAN: TF32-only + large batch.
    # Set apply_optimal_defaults=True in apply_optimal_throughput() to populate.
    _OPTIMAL_THROUGHPUT = {
        "rift_dit":  {"batch_size": 64, "precision": "bf16-mixed"},
        "ddsp6":    {"batch_size": 48, "precision": "bf16-mixed"},
        "diffusion": {"batch_size": 48, "precision": "bf16-mixed"},
        "sovits":   {"batch_size": 48, "precision": "bf16-mixed"},  # 完整GAN B=48, VRAM 10.5GB
        "rvc":      {"batch_size": 32, "precision": "bf16-mixed"},  # 完整GAN B=32, VRAM 10.4GB
    }

    def apply_optimal_throughput(self, override=False):
        # fill unset/inferior values only unless override=True (math unchanged, throughput up)
        opt = self._OPTIMAL_THROUGHPUT.get(self.backend, {})
        if opt:
            if override or self.batch_size == 32:  # 32 = the constructor default
                self.batch_size = opt["batch_size"]
            if override or self.precision == "bf16-mixed":
                self.precision = opt["precision"]

    # Top-level yaml keys that override feature generation. feature_spec() reads them from the dict it
    # receives; preprocessing and training must see the SAME values, so from_yaml merges any top-level
    # occurrence into backend_opts (backend_opts wins on conflict to keep existing configs working).
    _FEATURE_OVERRIDE_KEYS = (
        "units_kind", "units_ckpt", "f0_extractor", "rmvpe_ckpt", "uv_policy",
        "n_mels", "fmin", "fmax", "rvc_version", "version",
    )

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            source = f.read()
        d = yaml.safe_load(source)
        if d.get("backend") in ("ddsp6", "sovits") and d.get("spk") is not None:
            node = yaml.compose(source)
            for key, value in node.value:
                if key.value == "spk" and isinstance(value, yaml.MappingNode):
                    names = [name.value for name, _ in value.value if isinstance(name, yaml.ScalarNode)]
                    if len(set(names)) != len(names):
                        raise ValueError("duplicate speaker names in YAML spk mapping")
        bo = dict(d.get("backend_opts", {}) or {})
        for k in cls._FEATURE_OVERRIDE_KEYS:
            if k in d and k not in bo:
                bo[k] = d[k]
        return cls(d["backend"], d.get("dataset_dir", "data/default"), d.get("exp_dir", "experiments/default"),
                   d.get("val_dir"), d.get("batch_size", 32), d.get("max_frames", 256), d.get("num_workers", 8),
                   d.get("augment", True), OptimizerConfig.from_dict(d.get("optimizer", {})),
                   ScheduleConfig.from_dict(d.get("schedule")), d.get("grad_clip", 1.0),
                   d.get("precision", "bf16-mixed"), d.get("max_epochs"), d.get("max_steps", 500_000),
                   d.get("val_interval", 2000), d.get("save_interval", 2000), d.get("limit_val_batches"),
                   d.get("resume"), d.get("pretrained"), d.get("keep_last_n", 5), bo, spk=d.get("spk"))
