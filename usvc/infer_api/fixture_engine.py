"""Fixture weights are validated independently of the audible source-F0 adapter."""
import torch
from ..backends import get_backend
from ..infer.engine import InferenceEngine
from ..train.dataset import SVCBatch


def load_network(model, device="cpu"):
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.backend_name, engine.ckpt_path = model.backend, str(model.checkpoint)
    module = get_backend(model.backend)().build_module(model.cfg)
    engine._load_weights(module)
    return module.eval().to(device)


@torch.inference_mode()
def validate_network(model, device="cpu"):
    module = load_network(model, device)
    opts = model.cfg.backend_opts
    frames = 16
    units = torch.randn(1, frames, opts.get("n_units", opts.get("cvec_dim", 768)), device=device)
    f0 = torch.full((1, frames), 180., device=device)
    rms = torch.full((1, frames), .1, device=device)
    mel = torch.zeros(1, frames, opts.get("out_dims", opts.get("n_mel_channels", 128)), device=device)
    outputs = []
    for speaker in model.speakers:
        batch = SVCBatch(units=units, f0=f0, mel=mel, rms=rms,
                         uv=torch.ones_like(f0), spk_id=torch.tensor([speaker["id"]], device=device),
                         frame_len=torch.tensor([frames], device=device))
        if model.backend == "sovits":
            output = module.sample(batch)
        else:
            output = module.sample(batch, steps=2)
        if not output.numel() or not torch.isfinite(output).all():
            raise ValueError("network forward is empty or nonfinite")
        outputs.append(output.detach().float().cpu())
    return outputs
