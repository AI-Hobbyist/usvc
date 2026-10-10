"""Settings and request validation use the same parameter definitions."""
import math
from .errors import APIError
from ..speakers import resolve_speaker


def parameter(key, kind, default, description, *, nullable=False, minimum=None, maximum=None, enum=None, stage="convert", condition="always"):
    return dict(key=key, type=kind, default=default, description=description, nullable=nullable,
                minimum=minimum, maximum=maximum, enum=enum, stage=stage, condition=condition)


def definitions(model):
    params = [parameter("speaker", "speaker", model.default_speaker, "Name or zero-based speaker ID", nullable=True),
              parameter("trans", "number", 0., "Pitch transposition in semitones", minimum=-48., maximum=48.),
              parameter("f0_method", "string", None, "F0 extractor; null selects model configuration", nullable=True, enum=["fcpe", "rmvpe"]),
              parameter("resample_sr", "integer", None, "Output sample rate; null keeps model rate", nullable=True, minimum=8000, maximum=192000)]
    if model.fixture:
        return params
    params.extend([parameter("rms_mix_rate", "number", 1., "Output envelope mixing", minimum=0., maximum=1.),
                   parameter("target_loudness", "number", None, "Input target LUFS", nullable=True, minimum=-70., maximum=0.),
                   parameter("restore_loudness", "boolean", True, "Restore original LUFS", condition="target_loudness is set"),
                   parameter("half", "boolean", False, "FP16 autocast", stage="construct", condition="CUDA and non-diffusion model"),
                   parameter("compile_graph", "boolean", False, "Compile engine graphs", stage="construct"),
                   parameter("dec_cl", "boolean", True, "Decoder channels-last", stage="construct", condition="CUDA sm80+ and RVC/SoVITS"),
                   parameter("vocoder_cl", "boolean", True, "Vocoder channels-last", stage="construct", condition="CUDA sm80+ and external vocoder"),
                   parameter("index", "resource", None, "Registered index ID; false disables", nullable=True),
                   parameter("index_ratio", "number", 0., "Retrieval blend", minimum=0., maximum=1.),
                   parameter("protect", "number", .33, "Unvoiced feature protection", minimum=0., maximum=.5)])
    if model.backend in ("ddsp6", "diffusion", "reflow_vae", "rift_dit"):
        params.append(parameter("steps", "integer", None, "Model sampling steps", nullable=True, minimum=1, maximum=1000))
    if model.backend == "sovits":
        params.extend([parameter("noice_scale", "number", .4, "SoVITS latent noise scale", minimum=0., maximum=2.),
                       parameter("predict_f0", "boolean", False, "Use trained F0 predictor", condition="automatic F0 prediction enabled")])
    return params


def settings(model, device):
    return dict(backend=model.backend, model=model.id, implementation="torch", device=device,
                speakers=model.speakers, indexes=list(model.indexes), enhancers=[],
                test_fixture=model.fixture, parameters=definitions(model))


def validate(model, values, device):
    if not isinstance(values, dict):
        raise APIError(422, "invalid_parameter", "params must be a JSON object", "params")
    specs = {p["key"]: p for p in definitions(model)}
    unknown = values.keys() - specs.keys()
    if unknown:
        raise APIError(422, "invalid_parameter", "Unknown or unsupported parameter", sorted(unknown)[0])
    result = {key: values.get(key, spec["default"]) for key, spec in specs.items()}
    for key, value in result.items():
        spec = specs[key]
        if value is None and spec["nullable"]:
            continue
        kind = spec["type"]
        valid = ((kind == "number" and type(value) in (int, float) and math.isfinite(value)) or
                 (kind == "integer" and type(value) is int) or
                 (kind == "boolean" and type(value) is bool) or
                 (kind == "string" and isinstance(value, str)) or
                 (kind == "speaker" and type(value) in (str, int)) or
                 (kind == "resource" and (isinstance(value, str) or value is False)))
        if not valid:
            raise APIError(422, "invalid_parameter", f"Invalid type for {key}", key)
        if spec["enum"] and value not in spec["enum"]:
            raise APIError(422, "invalid_parameter", "Unsupported value", key)
        if spec["minimum"] is not None and value < spec["minimum"] or spec["maximum"] is not None and value > spec["maximum"]:
            raise APIError(422, "invalid_parameter", "Value outside allowed range", key)
    mapping = {p["name"]: p["id"] for p in model.speakers}
    try:
        result["speaker"] = resolve_speaker(result["speaker"], mapping, len(mapping))
    except ValueError as exc:
        raise APIError(422, "invalid_parameter", str(exc), "speaker") from exc
    if result.get("half") and (not device.startswith("cuda") or model.backend == "diffusion"):
        raise APIError(422, "invalid_parameter", "half requires CUDA and a non-diffusion backend", "half")
    if result.get("predict_f0") and not model.cfg.backend_opts.get("use_automatic_f0_prediction", True):
        raise APIError(422, "invalid_parameter", "Model has no F0 predictor", "predict_f0")
    if "index" in result:
        selected = result["index"]
        if selected is not None and selected is not False and selected not in model.indexes:
            raise APIError(422, "invalid_parameter", "Unknown index ID", "index")
        if result["index_ratio"] > 0 and selected is not False:
            selected = selected or model.default_index
            if selected is None:
                raise APIError(422, "invalid_parameter", "Positive ratio requires an available index", "index")
        result["index"] = selected
    return result
