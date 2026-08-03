"""Real 1-epoch training loss comparison for all runnable backends.

For each backend, train BOTH usvc and the original project for 1 epoch on
identical data + identical seed, then compare loss trends.

Backends:
  - rift_dit: usvc vs /root/RIFT-SVC (lightning)
  - ddsp6: usvc vs /root/DDSP-SVC-6.3 (with usvc MelExtractor substituted for NSF vocoder)
  - diffusion: usvc vs /root/Diffusion-SVC (if vector_quantize_pytorch available)
  - sovits/rvc: original projects have heavy deps; use usvc-vs-usvc baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# NOTE: module name conflicts between projects (both have diffusion/).
# Each train_*_orig function adds the right path locally.


def make_data(n=200, T=64, seed=777):
    g = torch.Generator().manual_seed(seed)
    return [{
        "mel": torch.randn(T, 128, generator=g) * 2,
        "f0": torch.rand(T, generator=g) * 400 + 100,
        "rms": torch.rand(T, generator=g),
        "units": torch.randn(T, 768, generator=g),
    } for _ in range(n)]


# ---- RIFT-DiT ----
def train_rift_usvc(samples, seed=42):
    import lightning.pytorch as pl
    from usvc.train.config import TrainConfig, OptimizerConfig
    from usvc.train.dataset import svc_collate
    from usvc.backends.rift_dit.module import RIFTModule

    class DS(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            d = samples[i]; return {**d, "frame_len": d["mel"].shape[0], "spk_id": 0}

    torch.manual_seed(seed); pl.seed_everything(seed, workers=True)
    cfg = TrainConfig(backend="rift_dit")
    cfg.backend_opts = {"dim": 256, "depth": 4, "head_dim": 64, "cvec_dim": 768,
                        "n_mel_channels": 128, "num_speaker": 1}
    cfg.optimizer = OptimizerConfig(type="adamw", lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    cfg.grad_clip = 1.0
    module = RIFTModule(cfg)
    dl = DataLoader(DS(), batch_size=16, shuffle=True, collate_fn=svc_collate, num_workers=0)

    losses = []
    class Tracker(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            m = trainer.callback_metrics
            l = m.get("train/loss") or m.get("train/loss_g")
            if l is not None:
                losses.append(float(l))
    trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=1, precision="32-true",
                        logger=False, enable_progress_bar=False, callbacks=[Tracker()],
                        num_sanity_val_steps=0)
    trainer.fit(module, dl)
    return losses


def train_rift_orig(samples, seed=42):
    if "/root/RIFT-SVC" not in sys.path[:1]:
        sys.path.insert(0, "/root/RIFT-SVC")
    from rift_svc.rf import RF
    from rift_svc.dit import DiT
    torch.manual_seed(seed)
    dit = DiT(dim=256, depth=4, head_dim=64, dropout=0.0, ff_mult=4,
              n_mel_channels=128, num_speaker=1, cvec_dim=768, kernel_size=3, zero_null_spk=True)
    rf = RF(dit, time_schedule="lognorm"); rf.mel_min, rf.mel_max = -12.0, 2.0
    rf = rf.cuda()
    opt = torch.optim.AdamW(rf.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; rf.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        mel = torch.stack([samples[i]["mel"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        rms = torch.stack([samples[i]["rms"] for i in idx]).cuda()
        cvec = torch.stack([samples[i]["units"] for i in idx]).cuda()
        fl = torch.full((len(idx),), mel.shape[1], device="cuda")
        sid = torch.zeros(len(idx), dtype=torch.long, device="cuda")
        loss, _ = rf(mel, spk_id=sid, f0=f0, rms=rms, cvec=cvec, frame_len=fl, drop_speaker=False)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(rf.parameters(), 1.0); opt.step()
        losses.append(float(loss.detach()))
    return losses


# ---- DDSP6 ----
def train_ddsp_usvc(samples, seed=42):
    from usvc.backends.ddsp6.model import DDSP6Model
    torch.manual_seed(seed)
    m = DDSP6Model(n_units=768, n_spk=1, n_aux_layers=3, n_aux_chans=256,
                   n_layers=2, n_chans=512).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        u = torch.stack([samples[i]["units"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        vol = torch.stack([samples[i]["rms"] for i in idx]).cuda()
        spec = torch.stack([samples[i]["mel"] for i in idx]).cuda()
        sid = torch.zeros(len(idx), dtype=torch.long).cuda()
        loss, _ = m(u, f0, vol, sid, spec)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_ddsp_orig(samples, seed=42):
    """Original DDSP6.3 Unit2Wav, using usvc MelExtractor as the 'vocoder' for mel
    extraction (NSF vocoder weight unavailable; worker proved MelExtractor gives
    diff=0 vs NSF nvSTFT)."""
    # Stub fairseq (Python 3.12 dataclass incompatibility)
    class _Stub:
        def __getattr__(self, n): return _Stub()
        def __call__(self, *a, **kw): return _Stub()
    for mod in ["fairseq", "fairseq.dataclass", "fairseq.dataclass.configs",
                "fairseq.checkpoint_utils", "fairseq.utils"]:
        sys.modules.setdefault(mod, _Stub())
    if "/root/DDSP-SVC-6.3" not in sys.path[:1]:
        sys.path.insert(0, "/root/DDSP-SVC-6.3")
    from usvc.backends.ddsp6.model import MelExtractor
    from reflow.vocoder import Unit2Wav

    torch.manual_seed(seed)
    m = Unit2Wav(
        sampling_rate=44100, block_size=512, win_length=2048, n_unit=768, n_spk=1,
        use_norm=True, use_attention=False, use_pitch_aug=True,
        out_dims=128, n_aux_layers=3, n_aux_chans=256, n_layers=2, n_chans=512,
    ).cuda()
    # Create MelExtractor instance to use as 'vocoder' (has .extract method)
    mel_ext = MelExtractor(sr=44100, n_mels=128, n_fft=2048, win_size=2048,
                            hop_length=512, fmin=40.0, fmax=16000.0)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        u = torch.stack([samples[i]["units"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda().unsqueeze(-1)  # [B,T,1]
        vol = torch.stack([samples[i]["rms"] for i in idx]).cuda().unsqueeze(-1)  # [B,T,1]
        spec = torch.stack([samples[i]["mel"] for i in idx]).cuda()
        sid = torch.zeros(len(idx), dtype=torch.long, device="cuda").unsqueeze(-1)  # [B,1]
        loss = m(u, f0, vol, spk_id=sid, vocoder=mel_ext, gt_spec=spec, infer=False,
                 t_start=0.0, silence_front=0)
        # Unit2Wav returns (ddsp_loss, reflow_loss); total = ddsp + reflow
        if isinstance(loss, tuple):
            ddsp_loss, reflow_loss = loss
            loss = ddsp_loss + reflow_loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def compare(name, u, o):
    mu, mo = np.mean(u), np.mean(o)
    md = abs(mu - mo) / abs(mo) * 100
    print(f"\n{'='*60}\n{name}: 真实训练 1 epoch 对比\n{'='*60}")
    print(f"  usvc: {len(u)} steps, mean={mu:.4f}, 前5={np.mean(u[:5]):.4f}, 后5={np.mean(u[-5:]):.4f}")
    print(f"  orig: {len(o)} steps, mean={mo:.4f}, 前5={np.mean(o[:5]):.4f}, 后5={np.mean(o[-5:]):.4f}")
    print(f"  均值相对差: {md:.1f}%")
    print(f"  → {'✓ 接近' if md < 5 else '△ 需检查' if md < 15 else '✗ 偏移过大'}")
    return {"mean_diff_pct": md, "usvc_mean": mu, "orig_mean": mo}


def train_diff_usvc(samples, seed=42):
    from usvc.backends.diffusion.model import DiffusionModel
    torch.manual_seed(seed)
    m = DiffusionModel(n_units=768, n_spk=1, n_hidden=256).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        u = torch.stack([samples[i]["units"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        vol = torch.stack([samples[i]["rms"] for i in idx]).cuda()
        spec = torch.stack([samples[i]["mel"] for i in idx]).cuda()
        sid = torch.zeros(len(idx), dtype=torch.long).cuda()
        loss, _ = m(u, f0, vol, sid, spec)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_diff_orig(samples, seed=42):
    # Diffusion-SVC has its own diffusion/ module; insert path and reload
    if "/root/Diffusion-SVC" not in sys.path[:1]:
        sys.path.insert(0, "/root/Diffusion-SVC")
        import importlib
        if "diffusion.unit2mel" in sys.modules:
            importlib.reload(sys.modules["diffusion"])
            importlib.reload(sys.modules["diffusion.unit2mel"])
    from diffusion.unit2mel import Unit2MelV2, DotDict
    torch.manual_seed(seed)
    m = Unit2MelV2(
        input_channel=768, n_spk=1, use_pitch_aug=False, out_dims=128,
        n_hidden=256, denoise_fn=DotDict({"type": "ConvNext", "cn_layers": 6, "cn_chans": 384}),
        naive_fn=None,
    ).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        u = torch.stack([samples[i]["units"] for i in idx]).cuda()
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda().unsqueeze(-1)  # [B,T,1]
        vol = torch.stack([samples[i]["rms"] for i in idx]).cuda().unsqueeze(-1)  # [B,T,1]
        spec = torch.stack([samples[i]["mel"] for i in idx]).cuda()
        sid = torch.zeros(len(idx), dtype=torch.long).cuda()
        out = m(u, f0, vol, sid, gt_spec=spec, infer=False)
        if isinstance(out, dict):
            loss = out.get("diff_loss", list(out.values())[0])
        else:
            loss = out
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_sovits_usvc(samples, seed=42):
    from usvc.backends.sovits.model import SoVITSModel
    torch.manual_seed(seed)
    # sovits needs spec [B, spec_ch, T] and wave
    m = SoVITSModel(n_units=256, spec_channels=513, inter_channels=192,
                    hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
                    n_speakers=1, segment_size=32, gin_channels=256).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        B = len(idx); T = 64
        c = torch.randn(B, T, 256).cuda()  # usvc: units [B, T, ssl_dim]
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        spec = torch.randn(B, 513, T).cuda()
        wave = torch.randn(B, 1, T * 256).cuda()
        g = torch.zeros(B, dtype=torch.long).cuda()
        fl = torch.full((B,), T).cuda()
        out = m(c, f0, spec, wave, g, frame_len=fl)
        # KL loss (deterministic part of GAN G loss)
        from usvc.train.losses import kl_loss
        loss = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_sovits_orig(samples, seed=42):
    if "/root/so-vits-svc" not in sys.path[:1]:
        sys.path.insert(0, "/root/so-vits-svc")
    from models import SynthesizerTrn
    torch.manual_seed(seed)
    # gin_channels=256 matches usvc n_units=256 path
    m = SynthesizerTrn(
        spec_channels=513, segment_size=32, inter_channels=192,
        hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
        kernel_size=3, p_dropout=0.1, resblock="1",
        resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
        upsample_rates=[8,8,2,2,2], upsample_initial_channel=512,
        upsample_kernel_sizes=[16,16,4,4,4],
        gin_channels=256, ssl_dim=256,
        n_speakers=1, sampling_rate=44100, vol_embedding=False,
        use_automatic_f0_prediction=True,
    ).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        B = len(idx); T = 64
        c = torch.randn(B, 256, T).cuda()  # orig: c [B, ssl_dim, T]
        f0 = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        spec = torch.randn(B, 513, T).cuda()
        g = torch.zeros(B, 1, dtype=torch.long).cuda()  # orig: g [B, 1]
        c_len = torch.full((B,), T).cuda()
        spec_len = torch.full((B,), T).cuda()
        uv = (f0 > 0).float()
        # orig forward: (o, ids, spec_mask, (z,z_p,m_p,logs_p,m_q,logs_q), pred_lf0, norm_lf0, lf0)
        o, ids, smask, (z, z_p, m_p, logs_p, m_q, logs_q), _, _, _ = m(
            c, f0, uv, spec, g=g, c_lengths=c_len, spec_lengths=spec_len, vol=None)
        from usvc.train.losses import kl_loss
        loss = kl_loss(z_p, logs_q, m_p, logs_p, smask)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


# ---- RVC ----
def train_rvc_usvc(samples, seed=42):
    from usvc.backends.rvc.model import RVCModel
    torch.manual_seed(seed)
    m = RVCModel(inter_channels=192, hidden_channels=192, filter_channels=768,
                 n_units=768, spk_embed_dim=1, segment_size=20, sr=48000).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        B = len(idx); T = 64
        phone = torch.randn(B, T, 768).cuda()
        pitch = torch.randint(1, 256, (B, T)).cuda()
        pitchf = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        spec = torch.randn(B, 1025, T).cuda()
        wave = torch.randn(B, 1, T * 256).cuda()
        sid = torch.zeros(B, dtype=torch.long).cuda()
        plen = torch.full((B,), T).cuda()
        out = m(phone, pitch, pitchf, spec, wave, sid, frame_len=plen)
        from usvc.train.losses import kl_loss
        loss = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_rvc_orig(samples, seed=42):
    # Isolate rvc_src import: clear infer cache, ensure rvc_src has priority
    for k in list(sys.modules):
        if k == "infer" or k.startswith("infer."):
            del sys.modules[k]
    sys.path = [p for p in sys.path if "RIFT-SVC" not in p]
    if "/tmp/rvc_src" not in sys.path:
        sys.path.insert(0, "/tmp/rvc_src")
    from infer.module.models import SynthesizerTrnMs768NSFsid
    torch.manual_seed(seed)
    m = SynthesizerTrnMs768NSFsid(
        spec_channels=1025, segment_size=20, inter_channels=192, hidden_channels=192,
        filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, resblock="1",
        resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
        upsample_rates=[10,10,2,2], upsample_initial_channel=512, upsample_kernel_sizes=[16,16,4,4],
        spk_embed_dim=1, gin_channels=256, sr=48000, is_half=False,
    ).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).tolist()
    losses = []; m.train()
    for start in range(0, len(order), 16):
        idx = order[start:start+16]
        B = len(idx); T = 64
        phone = torch.randn(B, T, 768).cuda()
        pitch = torch.randint(1, 256, (B, T)).cuda()
        pitchf = torch.stack([samples[i]["f0"] for i in idx]).cuda()
        spec = torch.randn(B, 1025, T).cuda()
        sid = torch.zeros(B, dtype=torch.long).cuda()
        plen = torch.full((B,), T).cuda()
        # orig: (o, ids, x_mask, y_mask, (z,z_p,m_p,logs_p,m_q,logs_q))
        o, ids, xmask, ymask, (z, z_p, m_p, logs_p, m_q, logs_q) = m(phone, plen, pitch, pitchf, spec, plen, sid)
        from usvc.train.losses import kl_loss
        loss = kl_loss(z_p, logs_q, m_p, logs_p, ymask)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return losses


def main():
    samples = make_data(200, 64)
    results = {}

    print(">>> RIFT-DiT 真实训练对比...")
    results["rift_dit"] = compare("RIFT-DiT", train_rift_usvc(samples), train_rift_orig(samples))

    print("\n>>> DDSP6 真实训练对比...")
    try:
        results["ddsp6"] = compare("DDSP6", train_ddsp_usvc(samples), train_ddsp_orig(samples))
    except Exception as e:
        print(f"  DDSP6 原项目对比失败: {e}")
        import traceback; traceback.print_exc()
        results["ddsp6"] = {"error": str(e)}

    print("\n>>> Diffusion 真实训练对比...")
    try:
        results["diffusion"] = compare("Diffusion", train_diff_usvc(samples), train_diff_orig(samples))
    except Exception as e:
        print(f"  Diffusion 原项目对比失败: {e}")
        import traceback; traceback.print_exc()
        results["diffusion"] = {"error": str(e)}

    print("\n>>> SoVITS KL loss 对比 (生成器前向)...")
    try:
        results["sovits"] = compare("SoVITS", train_sovits_usvc(samples), train_sovits_orig(samples))
    except Exception as e:
        print(f"  SoVITS 原项目对比失败: {e}")
        import traceback; traceback.print_exc()
        results["sovits"] = {"error": str(e)}

    print("\n>>> RVC KL loss 对比 (生成器前向)...")
    try:
        results["rvc"] = compare("RVC", train_rvc_usvc(samples), train_rvc_orig(samples))
    except Exception as e:
        print(f"  RVC 原项目对比失败: {e}")
        import traceback; traceback.print_exc()
        results["rvc"] = {"error": str(e)}

    out = Path("/tmp/usvc_validate/real_training_compare.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
