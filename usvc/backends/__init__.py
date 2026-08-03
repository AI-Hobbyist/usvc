"""Backend registry."""
from __future__ import annotations

__all__ = ["get_backend", "BACKENDS"]


def get_backend(name: str):
    """Return the Backend class for the given name.

    Lazily imported to avoid loading all backends' dependencies.
    """
    registry = {
        "rift_dit": "usvc.backends.rift_dit:RIFTDiTBackend",
        "ddsp6": "usvc.backends.ddsp6:DDSP6Backend",
        "diffusion": "usvc.backends.diffusion:DiffusionBackend",
        "sovits": "usvc.backends.sovits:SoVitsBackend",
        "rvc": "usvc.backends.rvc:RVCBackend",
        "reflow_vae": "usvc.backends.reflow_vae:ReflowVAEBackend",
    }
    if name not in registry:
        raise ValueError(f"Unknown backend: {name}. Available: {list(registry)}")
    mod_path, cls_name = registry[name].split(":")
    import importlib

    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


BACKENDS = ["rift_dit", "ddsp6", "diffusion", "sovits", "rvc", "reflow_vae"]
