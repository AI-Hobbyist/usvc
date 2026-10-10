"""Zero-based speaker identities shared by DDSP and SoVITS."""
from collections import Counter

TARGET_BACKENDS = {"ddsp6", "sovits"}


def validate_mapping(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("spk must be a nonempty name-to-ID mapping")
    if any(not isinstance(n, str) or not n.strip() for n in mapping):
        raise ValueError("speaker names must be nonempty strings")
    ids = list(mapping.values())
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(ids))):
        raise ValueError("speaker IDs must be unique contiguous integers 0..N-1")
    return dict(mapping)


def resolve_training_speakers(cfg, meta):
    names = meta.speakers
    if len(set(names)) != len(names):
        raise ValueError("duplicate speaker names in training meta")
    mapping = validate_mapping(cfg.spk if cfg.spk is not None else {n: i for i, n in enumerate(names)})
    if set(names) != set(mapping):
        raise ValueError("training speakers do not match spk mapping")
    capacity = cfg.backend_opts.get("num_speaker", len(mapping))
    if type(capacity) is not int or capacity != len(mapping):
        raise ValueError("num_speaker must equal the spk mapping capacity")
    cfg.spk = mapping
    cfg.backend_opts["num_speaker"] = capacity
    return mapping


def dataset_keys(meta, split):
    keys = [(a.speaker, a.file_name) for a in meta.audios]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate speaker/file identity in meta")
    if not meta.splits:
        return keys
    counts = Counter(fn for _, fn in keys)
    selected = {}
    identities = {f"{spk}/{fn}": (spk, fn) for spk, fn in keys}
    for name, entries in meta.splits.items():
        resolved = set()
        for entry in entries:
            if entry in identities:
                resolved.add(identities[entry])
            elif counts[entry] == 1:
                resolved.add(next(k for k in keys if k[1] == entry))
            elif counts[entry] > 1:
                raise ValueError(f"ambiguous legacy split {entry!r}; migrate to speaker/file_name")
            else:
                raise ValueError(f"unknown split identity: {entry!r}")
        selected[name] = resolved
    if selected.get("train", set()) & selected.get("val", set()):
        raise ValueError("training and validation splits overlap")
    return [k for k in keys if k in selected.get(split, set())]


def bind_dataset_speakers(ds, mapping, *, training=False):
    mapping = validate_mapping(mapping)
    if set(ds.meta.speakers) - set(mapping) or any(s not in mapping for s, _ in ds.keys):
        raise ValueError("unknown dataset speaker")
    if training:
        if not ds.keys:
            raise ValueError("empty training dataset")
        if {s for s, _ in ds.keys} != set(mapping):
            raise ValueError("each speaker must have training samples")
    ds.spk2idx = mapping


def resolve_speaker(selection, mapping, capacity):
    if mapping is not None:
        mapping = validate_mapping(mapping)
        if len(mapping) != capacity:
            raise ValueError("speaker mapping and model capacity disagree")
    if selection is None:
        if capacity != 1:
            raise ValueError("multi-speaker model requires an explicit speaker")
        return 0
    if isinstance(selection, str):
        if mapping is None or selection not in mapping:
            raise ValueError(f"unknown speaker name: {selection!r}")
        return mapping[selection]
    if type(selection) is not int or not 0 <= selection < capacity:
        raise ValueError(f"speaker ID must be an integer in 0..{capacity - 1}")
    return selection
