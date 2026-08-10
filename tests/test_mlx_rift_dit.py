"""rift_dit mlx 结构测试(随机权重 + 合成 ckpt 加载)。"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    _HAS_MLX = True
except Exception:  # pragma: no cover
    _HAS_MLX = False

needs_mlx = pytest.mark.skipif(not _HAS_MLX, reason="mlx 未安装")

DIM, DEPTH, HEAD_DIM, NMEL, NSPK, CVEC, KS = 256, 4, 64, 128, 2, 768, 31


def _make_model(num_speaker=NSPK, **over):
    from usvc.nn.mlx.backends.rift_dit import RIFTDiTModel
    kw = dict(dim=DIM, depth=DEPTH, head_dim=HEAD_DIM, n_mel_channels=NMEL,
              num_speaker=num_speaker, cvec_dim=CVEC, kernel_size=KS)
    kw.update(over)
    m = RIFTDiTModel(**kw)
    m.eval()
    return m


def _inputs(B=1, T=40):
    cvec = mx.random.normal((B, T, CVEC))
    f0 = mx.full((B, T), 440.0)
    rms = mx.full((B, T), 0.5)
    spk = mx.zeros((B,), dtype=mx.int32)
    return cvec, f0, rms, spk


@needs_mlx
def test_dit_forward_shape():
    """DiT forward → [B, T, n_mel]。"""
    from usvc.nn.mlx.backends.rift_dit import DiT
    dit = DiT(dim=DIM, depth=DEPTH, head_dim=HEAD_DIM,
              n_mel_channels=NMEL, cvec_dim=CVEC, kernel_size=KS)
    dit.eval()
    cvec, f0, rms, spk = _inputs()
    spec = mx.random.normal((1, 40, NMEL))
    out = dit(spec, spk, f0, rms, cvec, mx.array([500.0]))
    assert out.shape == (1, 40, NMEL)
    assert bool(mx.isfinite(out).all())


@needs_mlx
def test_dit_skip_layers():
    """skip_layers 与不 skip 的输出应不同且有限。"""
    from usvc.nn.mlx.backends.rift_dit import DiT
    dit = DiT(dim=DIM, depth=DEPTH, head_dim=HEAD_DIM,
              n_mel_channels=NMEL, cvec_dim=CVEC, kernel_size=KS)
    dit.eval()
    cvec, f0, rms, spk = _inputs()
    spec = mx.random.normal((1, 40, NMEL))
    out = dit(spec, spk, f0, rms, cvec, mx.array([500.0]))
    out_skip = dit(spec, spk, f0, rms, cvec, mx.array([500.0]),
                   skip_layers=[0, 1])
    assert bool(mx.isfinite(out_skip).all())
    assert bool((mx.abs(out - out_skip) > 1e-6).any())


@needs_mlx
def test_sample_paths():
    """sample 基本 + 各 CFG 路径 → [B, T, M] 有限。"""
    m = _make_model()
    cvec, f0, rms, spk = _inputs()
    out = m.sample(cvec, f0, rms, spk_id=spk, steps=6)
    assert out.shape == (1, 40, NMEL)
    assert bool(mx.isfinite(out).all())
    bad = mx.random.normal((1, 40, CVEC))
    for kw in [
        dict(bad_units=bad, ds_cfg_strength=2.0),
        dict(spk_cfg_strength=1.0),
        dict(skip_cfg_strength=1.0, cfg_skip_layers=[0]),
        dict(bad_units=bad, ds_cfg_strength=1.5, spk_cfg_strength=1.0),
    ]:
        o = m.sample(cvec, f0, rms, spk_id=spk, steps=4, **kw)
        assert o.shape == (1, 40, NMEL)
        assert bool(mx.isfinite(o).all())


@needs_mlx
def test_sample_mask_gt_spec():
    """mask 外帧回填 gt_spec(0)。"""
    m = _make_model()
    cvec, f0, rms, spk = _inputs()
    mask = mx.full((1, 40), True, dtype=mx.bool_)
    mask[0, 35:] = False
    gt = mx.zeros((1, 40, NMEL))
    out = m.sample(cvec, f0, rms, spk_id=spk, steps=4, mask=mask, gt_spec=gt)
    assert bool((mx.abs(out[0, 38]) < 1e-6).all())
    assert bool(mx.isfinite(out).all())


@needs_mlx
def test_rotary_math():
    """rotary 与手写 interleaved RoPE 参考一致(scale=1)。"""
    from usvc.nn.mlx.backends.rift_dit import RotaryEmbedding, _apply_rotary_pos_emb
    hd, n = 64, 8
    r = RotaryEmbedding(hd)
    freqs, scale = r.forward_from_seq_len(n)
    assert freqs.shape == (1, n, hd) and scale == 1.0
    # 手写参考(interleaved:x[2i], x[2i+1] 旋转 θ_i)
    pos = np.arange(n, dtype=np.float32)
    inv = 1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    theta = pos[:, None] * inv[None, :]                      # [n, hd/2]
    freqs_ref = np.repeat(theta, 2, axis=-1)                  # interleaved
    t = mx.random.normal((1, 4, n, hd))
    out = _apply_rotary_pos_emb(t, freqs, 1.0)
    tnp = np.asarray(t)
    x_even, x_odd = tnp[..., 0::2], tnp[..., 1::2]
    ref = np.empty_like(tnp)
    ref[..., 0::2] = x_even * np.cos(theta) - x_odd * np.sin(theta)
    ref[..., 1::2] = x_odd * np.cos(theta) + x_even * np.sin(theta)
    assert np.max(np.abs(np.asarray(out) - ref)) < 1e-5


@needs_mlx
def test_attention_sdpa_vs_manual():
    """Attention(SDPA)与手动 softmax 参考一致。"""
    from usvc.nn.mlx.backends.rift_dit import Attention
    attn = Attention(dim=128, head_dim=64)
    attn.eval()
    B, n = 1, 6
    x = mx.random.normal((B, n, 128))
    out = attn(x)
    # 手动参考(无 mask,无 rope)
    Wq, Wk, Wv = (np.asarray(p) for p in
                  (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight))
    bq, bk, bv = (np.asarray(p) for p in
                  (attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias))
    xn = np.asarray(x)
    q = xn @ Wq.T + bq; k = xn @ Wk.T + bk; v = xn @ Wv.T + bv
    h, hd = 2, 64
    q = q.reshape(B, n, h, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, n, h, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, n, h, hd).transpose(0, 2, 1, 3)
    # norm_q/norm_k(0 参数 LayerNorm over hd)
    def _ln(a, eps=1e-6):
        mu = a.mean(-1, keepdims=True); var = a.var(-1, keepdims=True)
        return (a - mu) / np.sqrt(var + eps)
    q, k = _ln(q), _ln(k)
    logits = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / 128.0)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    ref = p @ v
    ref = ref.transpose(0, 2, 1, 3).reshape(B, n, 128)
    Wo = np.asarray(attn.attn_out.weight); bo = np.asarray(attn.attn_out.bias)
    ref = ref @ Wo.T + bo
    assert np.max(np.abs(np.asarray(out) - ref)) < 1e-4


def _synthetic_torch_sd(tmp_path, lora=False, with_prefix=True):
    """构造与 torch RIFTDiTModel state_dict 同构的合成 sd(随机权重)。

    Returns (sd dict, ckpt path)。键形如 transformer.* 或 model.transformer.*。
    """
    import torch
    from torch import nn

    dim, depth, hd, nmel, nspk, cvec, ks = DIM, DEPTH, HEAD_DIM, NMEL, NSPK, CVEC, KS
    sd = {}
    sd["transformer.spk_embed.weight"] = torch.randn(nspk, dim) * 0.1
    sd["transformer.null_spk_embed.weight"] = torch.randn(1, dim) * 0.1
    sd["transformer.tembed.time_emb.weight"] = torch.randn(dim, 256) * 0.1
    sd["transformer.tembed.time_emb.bias"] = torch.zeros(dim)
    sd["transformer.tembed.proj.weight"] = torch.randn(dim, dim) * 0.1
    sd["transformer.tembed.proj.bias"] = torch.zeros(dim)
    sd["transformer.cond_embed.f0_embed.weight"] = torch.randn(dim, 1) * 0.1
    sd["transformer.cond_embed.f0_embed.bias"] = torch.zeros(dim)
    sd["transformer.cond_embed.rms_embed.weight"] = torch.randn(dim, 1) * 0.1
    sd["transformer.cond_embed.rms_embed.bias"] = torch.zeros(dim)
    sd["transformer.cond_embed.cvec_embed.weight"] = torch.randn(dim, cvec) * 0.1
    sd["transformer.cond_embed.cvec_embed.bias"] = torch.zeros(dim)
    sd["transformer.cond_embed.out.weight"] = torch.randn(dim, dim) * 0.1
    sd["transformer.cond_embed.out.bias"] = torch.zeros(dim)
    sd["transformer.cond_embed.ln.weight"] = torch.ones(dim)
    sd["transformer.cond_embed.ln.bias"] = torch.zeros(dim)
    sd["transformer.input_embed.mel_embed.weight"] = torch.randn(dim, nmel) * 0.1
    sd["transformer.input_embed.mel_embed.bias"] = torch.zeros(dim)
    sd["transformer.input_embed.proj.weight"] = torch.randn(dim, 2 * dim) * 0.1
    sd["transformer.input_embed.proj.bias"] = torch.zeros(dim)
    sd["transformer.rotary_embed.inv_freq"] = (
        1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
    )
    for i in range(depth):
        sd[f"transformer.transformer_blocks.{i}.attn_norm.proj.weight"] = torch.randn(6 * dim, dim) * 0.1
        sd[f"transformer.transformer_blocks.{i}.attn_norm.proj.bias"] = torch.zeros(6 * dim)
        for p in ("q_proj", "k_proj", "v_proj", "attn_out"):
            sd[f"transformer.transformer_blocks.{i}.attn.{p}.weight"] = torch.randn(dim, dim) * 0.1
            sd[f"transformer.transformer_blocks.{i}.attn.{p}.bias"] = torch.zeros(dim)
        # torch Conv1d weight (O, C/g, K)
        sd[f"transformer.transformer_blocks.{i}.mlp.dwconv.weight"] = torch.randn(dim, 1, ks) * 0.1
        sd[f"transformer.transformer_blocks.{i}.mlp.mlp_proj.weight"] = torch.randn(4 * dim, dim) * 0.1
        sd[f"transformer.transformer_blocks.{i}.mlp.mlp_proj.bias"] = torch.zeros(4 * dim)
        sd[f"transformer.transformer_blocks.{i}.mlp.mlp_out.weight"] = torch.randn(dim, 4 * dim) * 0.1
        sd[f"transformer.transformer_blocks.{i}.mlp.mlp_out.bias"] = torch.zeros(dim)
    sd["transformer.norm_out.proj.weight"] = torch.randn(2 * dim, dim) * 0.1
    sd["transformer.norm_out.proj.bias"] = torch.zeros(2 * dim)
    sd["transformer.output.weight"] = torch.randn(nmel, dim) * 0.1
    sd["transformer.output.bias"] = torch.zeros(nmel)

    if lora:
        # k_proj/v_proj → LoRALinear(linear.weight/bias + A/B),rank=8
        rank = 8
        for i in range(depth):
            for p in ("k_proj", "v_proj"):
                base = f"transformer.transformer_blocks.{i}.attn.{p}"
                sd[base + ".linear.weight"] = sd.pop(base + ".weight")
                sd[base + ".linear.bias"] = sd.pop(base + ".bias")
                sd[base + ".A"] = torch.randn(dim, rank) * 0.01
                sd[base + ".B"] = torch.randn(dim, rank) * 0.01

    if with_prefix:
        sd = {"model." + k: v for k, v in sd.items()}
    p = tmp_path / "rift_dit_synth.pt"
    torch.save(sd, p)
    return sd, str(p)


@needs_mlx
def test_from_ckpt_loads_and_runs(tmp_path):
    """from_ckpt:权重真实加载 + forward 有限 + 配置反推正确。"""
    import torch
    sd, path = _synthetic_torch_sd(tmp_path, with_prefix=True)
    from usvc.nn.mlx.backends.rift_dit import RIFTDiTModel
    m = RIFTDiTModel.from_ckpt(path)
    assert m.transformer.dim == DIM
    assert m.transformer.depth == DEPTH
    assert m.n_mel_channels == NMEL
    assert m.transformer.num_speaker == NSPK
    assert m.transformer.cond_embed.cvec_dim == CVEC
    # 权重真实加载(比对 mlx 与合成 sd 的值)
    from mlx.utils import tree_flatten
    loaded = dict(tree_flatten(m.parameters()))
    ref = sd["model.transformer.output.weight"]
    mlx_w = loaded["transformer.output.weight"]
    assert np.max(np.abs(np.asarray(mlx_w) - np.asarray(ref))) < 1e-6, "output.weight 未加载"
    cvec, f0, rms, spk = _inputs()
    out = m.sample(cvec, f0, rms, spk_id=spk, steps=4)
    assert out.shape == (1, 40, NMEL)
    assert bool(mx.isfinite(out).all())


@needs_mlx
def test_from_ckpt_bare_keys(tmp_path):
    """无前缀(裸 transformer.* sd,无 model. wrapper)也能加载。"""
    _, path = _synthetic_torch_sd(tmp_path, with_prefix=False)
    from usvc.nn.mlx.backends.rift_dit import RIFTDiTModel
    m = RIFTDiTModel.from_ckpt(path)
    cvec, f0, rms, spk = _inputs()
    out = m.sample(cvec, f0, rms, spk_id=spk, steps=4)
    assert out.shape == (1, 40, NMEL)


@needs_mlx
def test_from_ckpt_lora(tmp_path):
    """LoRA ckpt:自动套 LoRALinear,forward 有限。"""
    _, path = _synthetic_torch_sd(tmp_path, lora=True, with_prefix=True)
    from usvc.nn.mlx.backends.rift_dit import RIFTDiTModel, LoRALinear
    m = RIFTDiTModel.from_ckpt(path)
    block0 = m.transformer.transformer_blocks[0].attn
    assert isinstance(block0.k_proj, LoRALinear)
    assert isinstance(block0.v_proj, LoRALinear)
    cvec, f0, rms, spk = _inputs()
    out = m.sample(cvec, f0, rms, spk_id=spk, steps=4)
    assert out.shape == (1, 40, NMEL)
    assert bool(mx.isfinite(out).all())
    # merge_lora 后变成普通 Linear,输出仍有限
    m.transformer.merge_lora()
    from usvc.nn.mlx.backends.rift_dit import Attention
    assert isinstance(block0.k_proj, Attention) or not isinstance(block0.k_proj, LoRALinear)
    out2 = m.sample(cvec, f0, rms, spk_id=spk, steps=4)
    assert out2.shape == (1, 40, NMEL)


@needs_mlx
def test_lora_math():
    """LoRALinear 分解 forward == 合并后 forward。"""
    from usvc.nn.mlx.backends.rift_dit import DiT, LoRALinear
    dit = DiT(dim=DIM, depth=1, head_dim=HEAD_DIM,
              n_mel_channels=NMEL, cvec_dim=CVEC, kernel_size=KS)
    dit.eval()
    block = dit.transformer_blocks[0].attn
    lin = block.k_proj
    rank = 8
    lora = LoRALinear(lin, rank, 16.0)
    x = mx.random.normal((1, 10, DIM))
    out_decomp = lora(x)
    # 合并版
    delta = (lora.A @ lora.B.T) * lora.scale
    lin2 = lin  # same module;直接构造等效权重
    w = lin2.weight + delta.T
    out_merged = x @ w.T + lin2.bias
    assert np.max(np.abs(np.asarray(out_decomp) - np.asarray(out_merged))) < 1e-5
