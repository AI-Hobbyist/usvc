"""Speaker metadata hooks, limited to the two multi-speaker backends."""
from ..speakers import TARGET_BACKENDS, validate_mapping


def speaker_metadata(cfg):
    if cfg.backend not in TARGET_BACKENDS or cfg.spk is None:
        return None
    mapping = validate_mapping(cfg.spk)
    capacity = cfg.backend_opts.get("num_speaker", len(mapping))
    if type(capacity) is not int or capacity != len(mapping):
        raise ValueError("speaker mapping and capacity disagree")
    return {"version": 1, "spk": mapping, "capacity": capacity}


def read_speaker_metadata(payload):
    metadata = payload.get("usvc_speakers")
    if metadata is None:
        return None
    if metadata.get("version") != 1:
        raise ValueError("unsupported usvc_speakers version")
    mapping = validate_mapping(metadata.get("spk"))
    capacity = metadata.get("capacity")
    if type(capacity) is not int or capacity != len(mapping):
        raise ValueError("invalid checkpoint speaker capacity")
    return {"version": 1, "spk": mapping, "capacity": capacity}


def config_payload(cfg):
    payload = vars(cfg).copy()
    payload["optimizer"] = vars(cfg.optimizer).copy()
    payload["optimizer"]["betas"] = list(cfg.optimizer.betas)
    payload["schedule"] = vars(cfg.schedule).copy()
    return payload


class SpeakerCheckpointMixin:
    def on_save_checkpoint(self, checkpoint):
        metadata = speaker_metadata(self.cfg)
        if metadata is not None:
            checkpoint["usvc_speakers"] = metadata
            checkpoint["usvc_config"] = config_payload(self.cfg)

    def on_load_checkpoint(self, checkpoint):
        if self.cfg.backend not in TARGET_BACKENDS:
            return
        metadata = read_speaker_metadata(checkpoint)
        expected = speaker_metadata(self.cfg)
        if expected is not None and metadata != expected:
            raise ValueError("resume speaker mapping/capacity mismatch (or missing metadata)")
        saved = checkpoint.get("usvc_config")
        if saved is not None:
            if saved["backend"] != self.cfg.backend or saved["backend_opts"] != self.cfg.backend_opts:
                raise ValueError("resume architecture/configuration mismatch")
