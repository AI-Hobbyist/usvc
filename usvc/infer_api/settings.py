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
                   parameter("protect", "number", .5, "Unvoiced feature protection", minimum=0., maximum=.5),
                   parameter("cluster", "resource", None, "Registered cluster ID", nullable=True),
                   parameter("cluster_ratio", "number", 0., "Cluster blend", minimum=0., maximum=1.)])
    if model.backend in ("ddsp6", "diffusion", "reflow_vae", "rift_dit"):
        params.append(parameter("steps", "integer", None, "Model sampling steps", nullable=True, minimum=1, maximum=1000))
    if model.backend == "sovits":
        params.extend([parameter("noice_scale", "number", .4, "SoVITS latent noise scale", minimum=0., maximum=2.),
                       parameter("predict_f0", "boolean", False, "Use trained F0 predictor", condition="automatic F0 prediction enabled")])
    if model.backend in ("ddsp6", "diffusion", "reflow_vae", "rift_dit"):
        params.append(parameter("vocal_register_shift", "number", 0., "Model F0 register condition", minimum=-24., maximum=24., condition="mel model and pitch-capable vocoder"))
        if model.backend != "rift_dit":
            params.append(parameter("formant_shift", "number", 0., "Pitch augmentation condition in semitones", minimum=-24., maximum=24., condition="trained pitch augmentation"))
        params.append(parameter("vocoder", "resource", None, "Registered external vocoder ID (default)", nullable=True, stage="construct"))
    if model.backend in ("rvc", "sovits"):
        params.extend([parameter("shallow_diffusion", "boolean", False, "Enable registered shallow enhancer", condition="available enhancer resource"),
                       parameter("diffusion_model", "resource", None, "Registered diffusion enhancer ID", nullable=True, stage="construct"),
                       parameter("reflow_model", "resource", None, "Registered reflow enhancer ID", nullable=True, stage="construct"),
                       parameter("k_step", "integer", 100, "Shallow diffusion noise steps", minimum=1, maximum=1000, condition="diffusion enhancer active"),
                       parameter("diffusion_method", "string", "ddim", "Shallow diffusion sampler", enum=["ddim", "ddpm", "plms"], condition="diffusion enhancer active"),
                       parameter("diffusion_speedup", "integer", 10, "Shallow diffusion interval", minimum=1, maximum=1000, condition="diffusion enhancer active"),
                       parameter("t_start", "number", None, "Shallow reflow start time; null uses engine 0.6", nullable=True, minimum=0., maximum=1., condition="reflow enhancer active"),
                       parameter("reflow_steps", "integer", 20, "Shallow reflow ODE steps", minimum=1, maximum=1000, condition="reflow enhancer active")])
    if model.backend == "rift_dit":
        for key in ("ds_cfg_strength", "spk_cfg_strength", "skip_cfg_strength"):
            params.append(parameter(key, "number", model.cfg.backend_opts.get(key, 0.), "RIFT guidance strength", minimum=0., maximum=20., condition="corresponding trained guidance capability"))
    return params


def settings(model, device):
    return dict(backend=model.backend, model=model.id, implementation="torch", device=device,
                speakers=model.speakers, indexes=list(model.indexes), clusters=list(model.clusters), enhancers=list(model.enhancers),
                vocoders=["default"] if model.backend in ("ddsp6", "diffusion", "reflow_vae", "rift_dit") else [],
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
                import importlib.util
                if model.backend == "rvc" and importlib.util.find_spec("faiss") is None and str(model.metadata.get("default_index") or "").endswith(".index"):
                    raise APIError(422, "missing_dependency", 'Native retrieval requires pip install "usvc[rvc-index]"', "index")
                raise APIError(422, "invalid_parameter", "Positive ratio requires an available index", "index")
        result["index"] = selected
        if model.backend == "sovits" and selected and result["index_ratio"] > 0:
            import torch
            data = torch.load(model.indexes[selected], map_location="cpu", weights_only=False)
            target = next(p["name"] for p in model.speakers if p["id"] == result["speaker"])
            if isinstance(data, dict) and data.get("spk_id") is not None and data["spk_id"] != target:
                raise APIError(422, "invalid_parameter", "Index belongs to a different target speaker", "index")
    if result.get("cluster") is not None and result["cluster"] not in model.clusters:
        raise APIError(422, "invalid_parameter", "Unknown cluster ID", "cluster")
    if result.get("cluster_ratio", 0) > 0 and result.get("cluster") is None:
        raise APIError(422, "invalid_parameter", "Positive ratio requires a cluster ID", "cluster")
    if model.backend == "sovits" and result.get("cluster") and result["cluster_ratio"] > 0:
        import torch
        data = torch.load(model.clusters[result["cluster"]], map_location="cpu", weights_only=False)
        target = next(p["name"] for p in model.speakers if p["id"] == result["speaker"])
        if isinstance(data, dict) and "cluster_centers_" not in data and target not in data:
            raise APIError(422, "invalid_parameter", "Cluster has no resource for target speaker", "cluster")
    for key in ("k_step", "diffusion_method", "diffusion_speedup"):
        if key in values and not result.get("diffusion_model"):
            raise APIError(422, "invalid_parameter", "Parameter requires active diffusion enhancer", key)
    for key in ("t_start", "reflow_steps"):
        if key in values and not result.get("reflow_model"):
            raise APIError(422, "invalid_parameter", "Parameter requires active reflow enhancer", key)
    if "restore_loudness" in values and result.get("target_loudness") is None:
        raise APIError(422, "invalid_parameter", "restore_loudness requires target_loudness", "restore_loudness")
    for key in ("dec_cl", "vocoder_cl"):
        if key in values and values[key]:
            import torch
            supported_backend = model.backend in ("rvc", "sovits") if key == "dec_cl" else model.backend in ("ddsp6", "diffusion", "reflow_vae", "rift_dit") or result.get("shallow_diffusion")
            if not device.startswith("cuda") or not supported_backend or torch.cuda.get_device_capability(device)[0] < 8:
                raise APIError(422, "invalid_parameter", "channels-last option is not effective for this backend/device", key)
    if result.get("formant_shift", 0) != 0 and not model.cfg.backend_opts.get("use_pitch_aug", model.backend in ("ddsp6", "reflow_vae")):
        raise APIError(422, "invalid_parameter", "Model was not configured with pitch augmentation", "formant_shift")
    if result.get("vocal_register_shift", 0) != 0:
        from pathlib import Path
        import json
        from ..hub import pretrained_dir
        directory = (model.directory / model.metadata["vocoder_dir"]).resolve() if model.metadata.get("vocoder_dir") else pretrained_dir() / "nsf_hifigan"
        if not directory.is_relative_to(model.directory.parent.parent) and not directory.is_relative_to(pretrained_dir()):
            raise APIError(422, "invalid_parameter", "Vocoder path escapes allowed roots", "vocal_register_shift")
        try:
            supported = json.loads((directory / "config.json").read_text()).get("pc_aug") is True
        except (OSError, ValueError):
            supported = False
        if not supported:
            raise APIError(422, "invalid_parameter", "Vocoder must declare pc_aug capability", "vocal_register_shift")
    if result.get("spk_cfg_strength", 0) > 0 and model.cfg.backend_opts.get("drop_spk_prob", 0) <= 0:
        raise APIError(422, "invalid_parameter", "Speaker CFG requires trained speaker dropout", "spk_cfg_strength")
    if result.get("target_loudness") is not None:
        import importlib.util
        if importlib.util.find_spec("pyloudnorm") is None:
            raise APIError(503, "missing_dependency", "target_loudness requires pyloudnorm", "target_loudness")
    if result.get("vocoder") not in (None, "default"):
        raise APIError(422, "invalid_parameter", "Unknown vocoder ID", "vocoder")
    for key in ("diffusion_model", "reflow_model"):
        selected = result.get(key)
        if selected is not None:
            enhancer = model.enhancers.get(selected)
            allowed = ("diffusion",) if key == "diffusion_model" else ("ddsp6", "reflow_shallow")
            if enhancer is None or enhancer.backend not in allowed:
                raise APIError(422, "invalid_parameter", "Unknown or incompatible enhancer ID", key)
    if result.get("diffusion_model") and result.get("reflow_model"):
        raise APIError(422, "invalid_parameter", "Choose one enhancer", "reflow_model")
    if result.get("shallow_diffusion") and not (result.get("diffusion_model") or result.get("reflow_model")):
        raise APIError(422, "invalid_parameter", "shallow_diffusion requires a registered enhancer", "shallow_diffusion")
    if (result.get("diffusion_model") or result.get("reflow_model")) and not result.get("shallow_diffusion"):
        raise APIError(422, "invalid_parameter", "Enhancer selection requires shallow_diffusion=true", "shallow_diffusion")
    if result.get("diffusion_model"):
        enhancer = model.enhancers[result["diffusion_model"]]
        if result["k_step"] > enhancer.cfg.backend_opts.get("timesteps", 1000) or result["diffusion_speedup"] > result["k_step"]:
            raise APIError(422, "invalid_parameter", "Invalid shallow diffusion step/interval", "k_step")
        if result["diffusion_method"] == "ddpm" and result["diffusion_speedup"] != 1:
            raise APIError(422, "invalid_parameter", "ddpm requires diffusion_speedup=1", "diffusion_speedup")
    return result
