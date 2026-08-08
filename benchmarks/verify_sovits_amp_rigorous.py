"""Rigorous AMP loss check for sovits/rvc: isolate bf16 error from posterior noise.

The earlier 19% divergence was dominated by posterior sampling noise (z is random).
This script fixes the noise via torch.manual_seed to isolate the true bf16 error,
then compares multi-step training loss trajectories with matched noise.
"""
import sys
from pathlib import Path

import torch
import warnings
warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from usvc.backends.sovits.model import SoVITSModel
from usvc.backends.rvc.model import RVCModel
from usvc.train.losses import kl_loss


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (a.abs().max() + 1e-8)).item()


def test_sovits_isolated():
    """Fixed posterior noise: compare bf16 forward outputs to fp32."""
    print("=" * 60)
    print("sovits: 固定 posterior noise, 隔离 bf16 误差")
    print("=" * 60)
    torch.manual_seed(42)
    m = SoVITSModel(n_units=768, spec_channels=513, inter_channels=192, hidden_channels=192,
                    filter_channels=768, n_heads=2, n_layers=6, n_speakers=200,
                    segment_size=20, gin_channels=768).cuda()
    m.eval()
    B, T = 6, 20
    g = torch.Generator().manual_seed(1)
    u = torch.randn(B, T, 768, generator=g).cuda()
    f0 = torch.rand(B, T, generator=g).cuda() * 400 + 100
    spec = torch.randn(B, 513, T, generator=g).cuda()
    wave = torch.randn(B, 1, T * 512, generator=g).cuda()
    sid = torch.zeros(B, dtype=torch.long).cuda()
    fl = torch.full((B,), T).cuda()

    # Fixed noise for posterior z sampling
    torch.manual_seed(999)
    with torch.no_grad():
        out_fp = m(u, f0, spec, wave, sid, frame_len=fl)
    torch.manual_seed(999)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_amp = m(u, f0, spec, wave, sid, frame_len=fl)

    print("\n确定性统计量 (不受 z 采样影响):")
    for name in ["m_p", "logs_p", "m_q", "logs_q"]:
        rel_e = rel(out_fp[name], out_amp[name])
        print(f"  {name}: {rel_e:.2e}")

    # KL loss with MATCHED noise (both use same posterior z via seed)
    torch.manual_seed(999)
    with torch.no_grad():
        out_fp = m(u, f0, spec, wave, sid, frame_len=fl)
        kl_fp = kl_loss(out_fp["z_p"], out_fp["logs_q"], out_fp["m_p"], out_fp["logs_p"], out_fp["z_mask"]).item()
    torch.manual_seed(999)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_amp = m(u, f0, spec, wave, sid, frame_len=fl)
        kl_amp = kl_loss(out_amp["z_p"], out_amp["logs_q"], out_amp["m_p"], out_amp["logs_p"], out_amp["z_mask"]).item()
    print(f"\nKL loss (matched noise): fp32={kl_fp:.4f}, bf16={kl_amp:.4f}, diff={abs(kl_fp-kl_amp):.4f} ({abs(kl_fp-kl_amp)/kl_fp*100:.2f}%)")

    # Training trajectory: matched noise per step
    print("\n训练 10 步 (每步固定相同 noise):")
    m.train()
    def run(amp):
        torch.manual_seed(42)
        m2 = SoVITSModel(n_units=768, spec_channels=513, inter_channels=192, hidden_channels=192,
                         filter_channels=768, n_heads=2, n_layers=6, n_speakers=200,
                         segment_size=20, gin_channels=768).cuda()
        opt = torch.optim.AdamW(m2.parameters(), lr=5e-4)
        g2 = torch.Generator().manual_seed(1)
        losses = []
        for step in range(10):
            u = torch.randn(B, T, 768, generator=g2).cuda()
            f0 = torch.rand(B, T, generator=g2).cuda() * 400 + 100
            spec = torch.randn(B, 513, T, generator=g2).cuda()
            wave = torch.randn(B, 1, T * 512, generator=g2).cuda()
            opt.zero_grad()
            torch.manual_seed(1000 + step)  # matched posterior noise
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = m2(u, f0, spec, wave, sid, frame_len=fl)
                loss = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])
            loss.backward(); opt.step()
            losses.append(round(loss.item(), 4))
        return losses
    fp = run(False); amp = run(True)
    diffs = [abs(a-b)/abs(a)*100 for a, b in zip(fp, amp)]
    print(f"  fp32: {fp}")
    print(f"  amp:  {amp}")
    print(f"  逐步相对差%: {[round(d, 2) for d in diffs]}")
    print(f"  均值差: {sum(diffs)/len(diffs):.2f}%, 最大: {max(diffs):.2f}%")


if __name__ == "__main__":
    test_sovits_isolated()
