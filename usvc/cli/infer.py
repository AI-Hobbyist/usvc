"""usvc-infer CLI: offline voice conversion for any backend.

Usage:
    usvc-infer input.wav --config configs/sovits.yaml --ckpt ck.ckpt -o out.wav
    usvc-infer input.wav --config configs/diffusion.yaml --ckpt ck.ckpt \
        --vocoder-dir pretrained/nsf_hifigan --spk 0 --trans 0 --steps 50
    usvc-infer input.wav --config configs/rvc.yaml --ckpt ck.ckpt \
        --index-path index/rvc.pt --index-ratio 0.75
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="usvc-infer",
        description="Offline SVC inference (all backends: rift_dit/ddsp6/"
                    "diffusion/sovits/rvc/reflow_vae).",
    )
    p.add_argument("input", help="input wav path")
    p.add_argument("--config", "-c", required=True, help="training config yaml")
    p.add_argument("--ckpt", required=True, help="model checkpoint (.ckpt)")
    p.add_argument("--output", "-o", default=None, help="output wav path (default: out_<input>_<backend>.wav)")
    p.add_argument("--vocoder-dir", default=None,
                   help="NSF-HiFiGAN dir with config.json + model.ckpt "
                        "(default: usvc/pretrained/nsf_hifigan; only mel-domain backends)")
    p.add_argument("--spk", type=int, default=0, help="speaker id")
    p.add_argument("--trans", type=int, default=0, help="pitch shift in semitones")
    p.add_argument("--steps", type=int, default=None, help="sampling steps (flow/diff backends)")
    p.add_argument("--resample-sr", type=int, default=None,
                   help="output sample rate (default: backend sr; soxr resample)")
    p.add_argument("--noice-scale", type=float, default=0.4,
                   help="so-vits prior noise scale (sovits only; default 0.4)")
    p.add_argument("--auto-predict-f0", action="store_true",
                   help="so-vits: use the f0 decoder head instead of ground f0 (sovits only)")
    p.add_argument("--formant-shift", type=float, default=0.0,
                   help="formant shift in semitones (ddsp6/diffusion/reflow_vae with "
                        "use_pitch_aug; 0=off)")
    p.add_argument("--vocal-register-shift", type=float, default=0.0,
                   help="pc vocal register shift in semitones (mel-domain backends only: "
                        "ddsp6/diffusion/reflow_vae/rift_dit + pc-aug vocoder). Model sees "
                        "shifted f0, vocoder sees original: pitch stays, register shifts. "
                        "Positive = higher register. 0=off")
    p.add_argument("--f0-method", default=None, choices=["rmvpe", "fcpe"],
                   help="f0 extractor for inference: rmvpe | fcpe "
                        "(default: config's f0_extractor)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true",
                   help="use fp16 inference (rvc 默认开；合成器/声码器后端保持 fp32 可选)")
    p.add_argument("--no-half", action="store_true",
                   help="强制 fp32 推理（rvc 也关闭 fp16）")
    p.add_argument("--compile-graph", action="store_true",
                   help="torch.compile the generator/model (reduce-overhead CUDA graph; launch-bound inference)")
    p.add_argument("--index-path", default=None, help="retrieval index .pt (RVC style)")
    p.add_argument("--index-ratio", type=float, default=0.0, help="retrieval blend ratio (0-1)")
    p.add_argument("--cluster-path", default=None, help="cluster model .pt (so-vits style)")
    p.add_argument("--cluster-ratio", type=float, default=0.0, help="cluster blend ratio (0-1)")
    p.add_argument("--rms-mix-rate", type=float, default=1.0,
                   help="output RMS envelope match to input (1.0=off, 0.0=fully input envelope)")
    p.add_argument("--protect", type=float, default=0.5,
                   help="blend source units back on speech frames (<0.5 enables, 0.33 typical; "
                        "only meaningful with --index-path/--cluster-path)")
    # 浅扩散 (so-vits -shd; 也支持 rvc)
    p.add_argument("--shd", "--shallow-diffusion", dest="shallow_diffusion",
                   action="store_true", default=False,
                   help="shallow diffusion: enhance sovits/rvc output via an "
                        "independent diffusion model + PC vocoder "
                        "(需要 --diffusion-ckpt/--diffusion-config)")
    p.add_argument("--diffusion-ckpt", default=None, help="diffusion model checkpoint")
    p.add_argument("--diffusion-config", default=None, help="diffusion model config yaml")
    p.add_argument("--k-step", type=int, default=100, help="shallow diffusion steps (default 100)")
    p.add_argument("--diffusion-method", default="ddim",
                   help="sampling method: ddim|ddpm|plms (default ddim)")
    p.add_argument("--diffusion-speedup", type=int, default=10,
                   help="sampling speedup (default 10)")
    # reflow shallow diffusion (DDSP6 RectifiedFlow; sovits/rvc)
    p.add_argument("--reflow-ckpt", default=None,
                   help="DDSP6 reflow model checkpoint (enables reflow enhancer)")
    p.add_argument("--reflow-config", default=None,
                   help="DDSP6 reflow model config yaml")
    p.add_argument("--t-start", type=float, default=0.5,
                   help="reflow ODE start time [0,1): 0=full gen, ->1=keep gt "
                        "(default 0.5; only used with --reflow-ckpt)")
    # rift_dit CFG overrides (default None = use config values)
    p.add_argument("--ds-cfg", dest="ds_cfg_strength", type=float, default=None,
                   help="rift content(units) CFG strength (overrides config; default=config)")
    p.add_argument("--spk-cfg", dest="spk_cfg_strength", type=float, default=None,
                   help="rift speaker CFG strength (overrides config; default=config)")
    p.add_argument("--skip-cfg", dest="skip_cfg_strength", type=float, default=None,
                   help="rift skip-layer CFG strength (overrides config; default=config)")
    # LUFS loudness normalization (RIFT-SVC style)
    p.add_argument("--target-loudness", type=float, default=None,
                   help="normalize input to this integrated loudness in LUFS before "
                        "conversion (RIFT-SVC default -18.0; default: disabled)")
    p.add_argument("--no-restore-loudness", action="store_true", default=False,
                   help="do not restore output loudness to the input's original "
                        "loudness after conversion (only with --target-loudness)")
    args = p.parse_args(argv)

    # rvc 无独立 vocoder，fp16 已验证近无损（固定 seed SNR 58dB）；
    # 其余后端默认 fp32（--half 显式开启）。
    backend_name = ""
    try:
        import yaml
        _cfg = yaml.safe_load(open(args.config))
        backend_name = str(_cfg.get("backend", ""))
    except Exception:
        pass
    use_half = args.half or (backend_name == "rvc" and not args.no_half)

    from ..infer.engine import InferenceEngine

    engine = InferenceEngine(
        args.config, args.ckpt, vocoder_dir=args.vocoder_dir,
        device=args.device, half=use_half,
        diffusion_ckpt=args.diffusion_ckpt,
        diffusion_config=args.diffusion_config,
        reflow_ckpt=args.reflow_ckpt,
        reflow_config=args.reflow_config,
        t_start=args.t_start,
        compile_graph=args.compile_graph,
    )

    output = args.output
    if output is None:
        in_path = Path(args.input)
        output = f"out_{in_path.stem}_{engine.backend_name}.wav"

    engine.convert_file(
        args.input,
        output,
        spk=args.spk,
        trans=args.trans,
        index_path=args.index_path,
        index_ratio=args.index_ratio,
        cluster_path=args.cluster_path,
        cluster_ratio=args.cluster_ratio,
        steps=args.steps,
        shallow_diffusion=args.shallow_diffusion,
        k_step=args.k_step,
        diffusion_method=args.diffusion_method,
        diffusion_speedup=args.diffusion_speedup,
        t_start=args.t_start,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        f0_method=args.f0_method,
        resample_sr=args.resample_sr,
        noice_scale=args.noice_scale,
        predict_f0=args.auto_predict_f0,
        formant_shift=args.formant_shift,
        ds_cfg_strength=args.ds_cfg_strength,
        spk_cfg_strength=args.spk_cfg_strength,
        skip_cfg_strength=args.skip_cfg_strength,
        target_loudness=args.target_loudness,
        restore_loudness=not args.no_restore_loudness,
        vocal_register_shift=args.vocal_register_shift,
    )
    print(f"[usvc-infer] {args.input} -> {output} "
          f"(backend={engine.backend_name}, sr={engine.sr}, half={args.half})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
