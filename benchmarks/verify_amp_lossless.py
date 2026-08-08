"""Verify AMP bf16 is near-lossless: forward output relative error vs fp32(TF32)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for bench_epochs import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for usvc imports
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    import bench_epochs as be

    data = be.load_real_dataset(50)
    batches = be.make_batches(data, 8, 256)
    b = batches[0]

    def rel_err_max(a, bb):
        a32, b32 = a.float(), bb.float()
        return ((a32 - b32).abs().max() / (a32.abs().max() + 1e-8)).item()

    def rel_err_loss(la, lb):
        return abs(la.item() - lb.item()) / (abs(la.item()) + 1e-8)

    print("=== AMP bf16 近似无损验证 (vs fp32+TF32) ===\n")

    # rift_dit
    from usvc.backends.rift_dit.model import RIFTDiTModel
    torch.manual_seed(42)
    m = RIFTDiTModel(dim=512, depth=8, head_dim=64, n_mel_channels=128, num_speaker=1, cvec_dim=768).cuda()
    with torch.no_grad():
        l_fp, p_fp = m(b["mel"], b["sid"], b["f0"], b["rms"], b["units"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_amp, p_amp = m(b["mel"], b["sid"], b["f0"], b["rms"], b["units"])
    print(f"rift_dit: loss rel_err={rel_err_loss(l_fp, l_amp):.2e} ({rel_err_loss(l_fp,l_amp)*100:.3f}%)")
    print(f"          pred output max rel_err={rel_err_max(p_fp, p_amp):.2e}")
    print(f"          loss fp32={l_fp.item():.6f} amp={l_amp.item():.6f}")

    # ddsp6
    from usvc.backends.ddsp6.model import DDSP6Model
    torch.manual_seed(42)
    m = DDSP6Model(n_units=768, n_spk=1, n_aux_layers=6, n_aux_chans=512, n_layers=6, n_chans=1024, velocity_net="LYNXNet2").cuda()
    with torch.no_grad():
        l_fp, _ = m(b["units"], b["f0"], b["rms"], b["sid"], b["mel"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_amp, _ = m(b["units"], b["f0"], b["rms"], b["sid"], b["mel"])
    print(f"\nddsp6:    loss rel_err={rel_err_loss(l_fp, l_amp):.2e} ({rel_err_loss(l_fp,l_amp)*100:.3f}%)")
    print(f"          loss fp32={l_fp.item():.6f} amp={l_amp.item():.6f}")

    # diffusion (fix RNG for identical noise)
    from usvc.backends.diffusion.model import DiffusionModel
    torch.manual_seed(42)
    m = DiffusionModel(n_units=768, n_spk=1, out_dims=128, n_hidden=256, cn_layers=20, cn_chans=384).cuda()
    torch.manual_seed(0)
    with torch.no_grad():
        l_fp, _ = m(b["units"], b["f0"], b["rms"], b["sid"], b["mel"])
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l_amp, _ = m(b["units"], b["f0"], b["rms"], b["sid"], b["mel"])
    print(f"\ndiffusion: loss rel_err={rel_err_loss(l_fp, l_amp):.2e} ({rel_err_loss(l_fp,l_amp)*100:.3f}%)")
    print(f"           loss fp32={l_fp.item():.6f} amp={l_amp.item():.6f}")

    # sovits (synthetic, GAN) - kl loss
    from usvc.backends.sovits.model import SoVITSModel
    from usvc.train.losses import kl_loss
    torch.manual_seed(42)
    m = SoVITSModel(n_units=768, spec_channels=513, inter_channels=192, hidden_channels=192,
                    filter_channels=768, n_heads=2, n_layers=6, n_speakers=200, segment_size=20, gin_channels=768).cuda()
    B, T = 6, 20
    g = torch.Generator().manual_seed(1)
    u = torch.randn(B, T, 768, generator=g).cuda(); f0 = torch.rand(B, T, generator=g).cuda()*400+100
    spec = torch.randn(B, 513, T, generator=g).cuda(); wave = torch.randn(B, 1, T*512, generator=g).cuda()
    sid = torch.zeros(B, dtype=torch.long).cuda(); fl = torch.full((B,), T).cuda()
    with torch.no_grad():
        out_fp = m(u, f0, spec, wave, sid, frame_len=fl)
        kl_fp = kl_loss(out_fp["z_p"], out_fp["logs_q"], out_fp["m_p"], out_fp["logs_p"], out_fp["z_mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_amp = m(u, f0, spec, wave, sid, frame_len=fl)
            kl_amp = kl_loss(out_amp["z_p"], out_amp["logs_q"], out_amp["m_p"], out_amp["logs_p"], out_amp["z_mask"])
    print(f"\nsovits:   kl loss rel_err={rel_err_loss(kl_fp, kl_amp):.2e} ({rel_err_loss(kl_fp,kl_amp)*100:.3f}%)")
    print(f"          m_p max rel_err={rel_err_max(out_fp['m_p'], out_amp['m_p']):.2e}")
    print(f"          loss fp32={kl_fp.item():.6f} amp={kl_amp.item():.6f}")


if __name__ == "__main__":
    main()
