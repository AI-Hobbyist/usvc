"""Rigorous DDSP 6.1/6.2 training comparison: usvc vs original.

Stubs fairseq to let DDSP-SVC-6.1 import. Uses usvc MelExtractor as the
vocoder (worker proved diff=0 vs NSF nvSTFT). Compares multi-step training
loss with identical seed/data/lr.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def stub_fairseq():
    """Bypass fairseq dataclass incompatibility (Python 3.12)."""
    class Stub:
        def __getattr__(self, n): return Stub()
        def __call__(self, *a, **kw): return Stub()
    for mod in ["fairseq", "fairseq.dataclass", "fairseq.dataclass.configs",
                "fairseq.checkpoint_utils", "fairseq.utils"]:
        sys.modules[mod] = Stub()


def make_data(n=80, T=64, seed=123):
    g = torch.Generator().manual_seed(seed)
    return [{
        "units": torch.randn(T, 768, generator=g),
        "f0": torch.rand(T, generator=g) * 400 + 100,
        "vol": torch.rand(T, generator=g),
        "spec": torch.randn(T, 128, generator=g),
    } for _ in range(n)]


def train_usvc(samples, version, seed=42):
    from usvc.backends.ddsp6.model import DDSP6Model
    vn = "LYNXNet" if version in ("6.1", "6.2") else "LYNXNet2"
    torch.manual_seed(seed)
    m = DDSP6Model(n_units=768, n_spk=1, n_aux_layers=3, n_aux_chans=256,
                   n_layers=2, n_chans=512, velocity_net=vn, t_start=0.0).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        u = torch.stack([samples[i]["units"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda().unsqueeze(-1)
        v = torch.stack([samples[i]["vol"] for i in idx]).cuda().unsqueeze(-1)
        s = torch.stack([samples[i]["spec"] for i in idx]).cuda()
        sid = torch.zeros(len(idx), dtype=torch.long, device="cuda").unsqueeze(-1)
        loss, _ = m(u, f0, v, sid, s)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_orig_61(samples, seed=42):
    sys.path.insert(0, "/root/DDSP-SVC-6.1")
    from reflow.vocoder import Unit2Wav
    from usvc.backends.ddsp6.model import MelExtractor
    torch.manual_seed(seed)
    m = Unit2Wav(
        sampling_rate=44100, block_size=512, win_length=2048, n_unit=768, n_spk=1,
        use_norm=False, use_attention=False, use_pitch_aug=True,
        out_dims=128, n_aux_layers=3, n_aux_chans=256, n_layers=2, n_chans=512,
    ).cuda()
    mel_ext = MelExtractor(sr=44100, n_mels=128, n_fft=2048, win_size=2048,
                           hop_length=512, fmin=40.0, fmax=16000.0)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        u = torch.stack([samples[i]["units"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda().unsqueeze(-1)
        v = torch.stack([samples[i]["vol"] for i in idx]).cuda().unsqueeze(-1)
        s = torch.stack([samples[i]["spec"] for i in idx]).cuda()
        sid = torch.zeros(len(idx), dtype=torch.long, device="cuda").unsqueeze(-1)
        ddsp_loss, reflow_loss = m(u, f0, v, spk_id=sid, vocoder=mel_ext,
                                    gt_spec=s, infer=False, t_start=0.0, silence_front=0)
        loss = ddsp_loss + reflow_loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def compare(name, u, o):
    mu, mo = np.mean(u), np.mean(o)
    md = abs(mu - mo) / abs(mo) * 100
    print(f"\n{'='*60}\n{name}: 真实训练 loss 对比\n{'='*60}")
    print(f"  usvc: {len(u)} steps, mean={mu:.4f}, 首3={np.mean(u[:3]):.4f}, 末3={np.mean(u[-3:]):.4f}")
    print(f"  orig: {len(o)} steps, mean={mo:.4f}, 首3={np.mean(o[:3]):.4f}, 末3={np.mean(o[-3:]):.4f}")
    # 逐步 diff
    diffs = [abs(a-b) for a,b in zip(u,o)]
    max_step_diff = max(diffs)
    print(f"  逐步 max diff: {max_step_diff:.2e}")
    print(f"  均值相对差: {md:.2f}%")
    status = "PASS ✓" if max_step_diff < 1e-3 else "CLOSE △" if max_step_diff < 1e-1 else "FAIL ✗"
    print(f"  → {status}")
    return {"usvc": u, "orig": o, "max_step_diff": max_step_diff, "mean_diff_pct": md, "status": status}


def main():
    stub_fairseq()
    samples = make_data(80, 64)
    results = {}

    print(">>> DDSP 6.1 真实训练对比 (LYNXNet)...")
    u = train_usvc(samples, "6.1")
    o = train_orig_61(samples)
    results["ddsp6.1"] = compare("DDSP-6.1 (LYNXNet)", u, o)

    out = Path("/tmp/usvc_validate/ddsp61_compare.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
