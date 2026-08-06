"""Per-backend training throughput benchmark.

Measures forward+backward time per step for each backend at the original
project's typical batch_size and segment length, then extrapolates epoch time
for a representative dataset (e.g. 1540 clips, ~6s each).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

torch.backends.cudnn.benchmark = True

# Representative dataset: 1540 clips, avg 6.07s each (from preprocessing run)
N_CLIPS = 1540
AVG_CLIP_SEC = 6.07


def warmup(fn, n=3):
    for _ in range(n):
        fn()


def bench(fn, n=10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def bench_rift_dit():
    from usvc.backends.rift_dit.model import RIFTDiTModel

    # RIFT: batch=64, 256 mel frames
    for cfg_name, dim, depth, B in [("prod(1024-16)", 1024, 16, 64), ("a10g(512-8)", 512, 8, 64)]:
        try:
            torch.manual_seed(42)
            m = RIFTDiTModel(dim=dim, depth=depth, head_dim=64, n_mel_channels=128,
                             num_speaker=1, cvec_dim=768).cuda()
            opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
            T = 256
            mel = torch.randn(B, T, 128).cuda()
            f0 = torch.rand(B, T).cuda() * 400 + 100
            cvec = torch.randn(B, T, 768).cuda()
            rms = torch.rand(B, T).cuda()
            sid = torch.zeros(B, dtype=torch.long).cuda()

            def step():
                opt.zero_grad()
                loss, _ = m(mel, sid, f0, rms, cvec)
                loss.backward()
                opt.step()

            warmup(step, 2)
            t = bench(step, 5)
            yield ("rift_dit", cfg_name, B, t)
            del m; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            yield ("rift_dit", cfg_name, B, None)
            torch.cuda.empty_cache()


def bench_ddsp6():
    from usvc.backends.ddsp6.model import DDSP6Model

    # DDSP6.3: batch=48, T~256 frames (block_size=512, sr=44100 → ~3s)
    for cfg_name, vn, nlayers, nchans, B in [
        ("6.3-LYNXNet2", "LYNXNet2", 6, 1024, 48),
        ("6.3-LYNXNet2-a10g", "LYNXNet2", 6, 512, 48),
    ]:
        try:
            m = DDSP6Model(n_units=768, n_spk=1, n_aux_layers=6, n_aux_chans=512,
                           n_layers=nlayers, n_chans=nchans, velocity_net=vn).cuda()
            opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
            B_eff = B
            T = 256
            u = torch.randn(B_eff, T, 768).cuda()
            f0 = torch.rand(B_eff, T).cuda() * 400 + 100
            v = torch.rand(B_eff, T).cuda()
            spec = torch.randn(B_eff, T, 128).cuda()
            sid = torch.zeros(B_eff, dtype=torch.long).cuda()

            def step():
                opt.zero_grad()
                loss, _ = m(u, f0, v, sid, spec)
                loss.backward()
                opt.step()

            warmup(step, 2)
            t = bench(step, 5)
            yield ("ddsp6", cfg_name, B_eff, t)
            del m; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            yield ("ddsp6", cfg_name, B, None)
            torch.cuda.empty_cache()


def bench_diffusion():
    from usvc.backends.diffusion.model import DiffusionModel

    # Diffusion: batch=48, T~256, ConvNext
    try:
        m = DiffusionModel(n_units=768, n_spk=1, out_dims=128, n_hidden=256,
                           cn_layers=20, cn_chans=384).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
        B = 48; T = 256
        u = torch.randn(B, T, 768).cuda()
        mel = torch.randn(B, T, 128).cuda()  # [B,T,M]
        f0 = torch.rand(B, T).cuda() * 400 + 100
        vol = torch.rand(B, T).cuda()
        sid = torch.zeros(B, dtype=torch.long).cuda()

        def step():
            opt.zero_grad()
            loss, _ = m(u, f0, vol, sid, mel)
            loss.backward()
            opt.step()

        warmup(step, 2)
        t = bench(step, 5)
        yield ("diffusion", "ConvNext-6L", B, t)
        del m; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        yield ("diffusion", "ConvNext-6L", 48, None)
        torch.cuda.empty_cache()


def bench_sovits():
    from usvc.backends.sovits.model import SoVITSModel
    from usvc.train.losses import kl_loss

    # sovits: batch=6, segment_size=20 frames (10240 wave), sr=44100
    try:
        m = SoVITSModel(n_units=768, spec_channels=513, inter_channels=192,
                        hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
                        n_speakers=200, segment_size=20, gin_channels=768).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
        B = 6; T = 20
        u = torch.randn(B, T, 768).cuda()
        f0 = torch.rand(B, T).cuda() * 400 + 100
        spec = torch.randn(B, 513, T).cuda()
        wave = torch.randn(B, 1, 10240).cuda()
        sid = torch.zeros(B, dtype=torch.long).cuda()
        fl = torch.full((B,), T).cuda()

        def step():
            opt.zero_grad()
            out = m(u, f0, spec, wave, sid, frame_len=fl)
            loss = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])
            loss.backward()
            opt.step()

        warmup(step, 2)
        t = bench(step, 5)
        yield ("sovits", "VITS-NSF", B, t)
        del m; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        yield ("sovits", "VITS-NSF", 6, None)
        torch.cuda.empty_cache()


def bench_rvc():
    from usvc.backends.rvc.model import RVCModel
    from usvc.train.losses import kl_loss

    # rvc: batch=4, segment_size=36 frames (17280 wave at 48k/480hop), sr=48000
    try:
        m = RVCModel(inter_channels=192, hidden_channels=192, filter_channels=768,
                     n_units=768, spk_embed_dim=200, segment_size=36, sr=48000).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
        B = 4; T = 36
        phone = torch.randn(B, T, 768).cuda()
        pitch = torch.randint(1, 256, (B, T)).cuda()
        pitchf = torch.rand(B, T).cuda() * 400 + 100
        spec = torch.randn(B, 1025, T).cuda()
        wave = torch.randn(B, 1, 17280).cuda()
        sid = torch.zeros(B, dtype=torch.long).cuda()
        plen = torch.full((B,), T).cuda()

        def step():
            opt.zero_grad()
            out = m(phone, pitch, pitchf, spec, wave, sid, frame_len=plen)
            loss = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])
            loss.backward()
            opt.step()

        warmup(step, 2)
        t = bench(step, 5)
        yield ("rvc", "VITS-NSF-768", B, t)
        del m; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        yield ("rvc", "VITS-NSF-768", 4, None)
        torch.cuda.empty_cache()


def main():
    results = []
    print(f"{'Backend':<12} {'Config':<22} {'B':<4} {'ms/step':>9} {'steps/epoch':>12} {'min/epoch':>10}")
    print("-" * 75)

    for gen in [bench_rift_dit, bench_ddsp6, bench_diffusion, bench_sovits, bench_rvc]:
        for backend, cfg, B, t in gen():
            if t is None:
                print(f"{backend:<12} {cfg:<22} {B:<4} {'OOM':>9}")
                results.append({"backend": backend, "config": cfg, "batch_size": B, "ms_per_step": None, "oom": True})
                continue
            # 每个 clip 可切出多个 segment, 近似: 每 epoch = N_CLIPS 个 clip
            # 每个 clip ~6s, segment 长度因 backend 不同
            # steps_per_epoch = N_CLIPS / B (每 clip 取 1 segment)
            steps = N_CLIPS / B
            epoch_sec = steps * t
            epoch_min = epoch_sec / 60
            print(f"{backend:<12} {cfg:<22} {B:<4} {t*1000:>9.1f} {steps:>12.0f} {epoch_min:>10.1f}")
            results.append({
                "backend": backend, "config": cfg, "batch_size": B,
                "ms_per_step": round(t * 1000, 1),
                "steps_per_epoch": round(steps),
                "min_per_epoch": round(epoch_min, 1),
            })

    out = Path("/tmp/usvc_validate/throughput.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {mem:.1f} GB")


if __name__ == "__main__":
    main()
