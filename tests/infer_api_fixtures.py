"""Persistent fixtures: create missing directories only, never overwrite or delete."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from usvc.backends import get_backend
from usvc.train.config import TrainConfig
from usvc.train.speaker_checkpoint import speaker_metadata, config_payload

MATRIX = [("ddsp6", "__test_ddsp65_single", 1), ("ddsp6", "__test_ddsp65_multi", 2),
          ("sovits", "__test_sovits_single", 1), ("sovits", "__test_sovits_multi", 2),
          ("diffusion", "__test_diffusion_single", 1),
          ("reflow_vae", "__test_reflow_vae_single", 1), ("rift_dit", "__test_rift_dit_single", 1)]


def generate(root):
    root = Path(root).resolve()
    generated = []
    for backend, name, capacity in MATRIX:
        directory = root / backend / name
        if directory.exists():
            print(f"PRESERVED {directory}", flush=True)
            continue
        opts = dict(n_units=768, num_speaker=capacity, sampling_rate=44100, hop_length=512,
                    block_size=512, units_kind="contentvec", f0_extractor="fcpe")
        if backend == "ddsp6":
            opts.update(version="6.5", n_aux_layers=1, n_aux_chans=32, n_layers=1, n_chans=32, use_pitch_aug=True)
        elif backend == "sovits":
            opts.update(inter_channels=32, hidden_channels=32, filter_channels=64, gin_channels=32,
                        n_layers_enc=1, n_layers_dec=1, n_sdp_layers0=1, n_sdp_layers=1,
                        use_automatic_f0_prediction=False)
        elif backend == "diffusion":
            opts.update(n_hidden=32, cn_layers=1, cn_chans=32, timesteps=20, k_step=20)
        elif backend == "reflow_vae":
            opts.update(n_layers=1, n_chans=32, n_hidden=32, use_attention=False)
        else:
            opts.update(dim=64, depth=4, head_dim=16, cvec_dim=768, eval_sample_steps=2)
        cfg = TrainConfig(backend=backend, backend_opts=opts,
                          spk={"测试甲": 0, "测试乙": 1} if capacity == 2 else {"测试歌手": 0})
        torch.manual_seed(20261010)
        module = get_backend(backend)().build_module(cfg)
        state = {k: v.cpu() for k, v in module.state_dict().items() if not k.startswith("discriminator.")}
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "config.yaml").write_text(yaml.safe_dump(dict(backend=backend, backend_opts=opts, spk=cfg.spk), allow_unicode=True), encoding="utf-8")
        payload = {"state_dict": state, "usvc_config": config_payload(cfg)}
        metadata = speaker_metadata(cfg)
        if metadata:
            payload["usvc_speakers"] = metadata
        torch.save(payload, directory / "model.ckpt")
        (directory / "model.json").write_text(json.dumps(dict(config="config.yaml", checkpoint="model.ckpt",
            test_fixture=True, fixture_output="source_f0_sine", untrained=True, seed=20261010,
            generator_version=2), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"CREATED {directory}", flush=True)
        generated.append(directory)
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    generate(parser.parse_args().model_root)
