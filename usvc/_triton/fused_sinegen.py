"""Triton fused NSF SineGen（frame-domain 相位累加）：把 sinegen 链条 ~6-8 个小 elementwise
kernel（rad2+fmod+cumsum+pad+reshape+sin+uv+noise_amp*noise）合成一个，消 launch 开销，
让外层 torch.compile CUDA graph 更紧凑。
数值契约：fp32 下与 sovits ``Generator.fast_sinegen`` bit 级一致（torch 2.11 + triton 3.6 实证）：
tl.sin 与 torch.sin、x%1.0 与 torch.remainder 均 fp32 bit-exact；torch.cumsum 在 [B,T,dim] 上是
scan_outer_dim 顺序逐列扫描而非树扫描（tl.cumsum maxdiff ~4e-4 @T=860），故 kernel 保持顺序
T 循环 + 循环携带累加。noise/随机初相由调用方生成传入（固定种子可复现），None 取零；
custom op 注册同 dwconv1d.py（Meta+CUDA，inference-only）。"""
import math

import torch

from ._compat import _HAS_TRITON

if _HAS_TRITON:
    import triton
    import triton.language as tl
else:
    # 无 triton 环境: no-op jit stub（kernel 定义不报错, 实际走 torch fallback）
    class _TritonStub:
        @staticmethod
        def jit(fn=None, **kw):
            return fn if fn is not None else (lambda f: f)
    triton = _TritonStub()
    tl = None


@triton.jit
def _sinegen_kernel(f0_ptr, rand_ini_ptr, noise_ptr, out_ptr, T, L,
                    sr, sine_amp, noise_std,
                    upp: tl.constexpr, dim: tl.constexpr,
                    BLOCK_S: tl.constexpr, S_BLOCKS: tl.constexpr,
                    HAS_RAND: tl.constexpr, HAS_NOISE: tl.constexpr):
    """out[b,t*upp+s,d] = sin(2π·phase)*sa*uv + n_amp*noise;
    phase = cumsum_{t'<t}(rad*upp) + rad*(s+1) + rand_ini[b,d], rad=(f0*(d+1)/sr)%1,
    uv = f0>0, n_amp = uv*ns+(1-uv)*sa/3. 顺序 T 循环匹配 torch scan_outer_dim (fp32)。"""
    pid = tl.program_id(0)
    sb, rest = pid % S_BLOCKS, pid // S_BLOCKS
    d, b = rest % dim, rest // dim
    s0 = sb * BLOCK_S; arange = tl.arange(0, BLOCK_S)
    s = s0 + arange + 1                 # 1..upp (1-based)
    smask = s <= upp; s_f = s.to(tl.float32)
    ri = 0.0
    if HAS_RAND:
        ri = tl.load(rand_ini_ptr + b * dim + d).to(tl.float32)
    # 顺序 cumsum 载体：全 lane 相同的 [BLOCK_S] 向量（Triton 不支持 0-d 索引）
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    f0_base = f0_ptr + b * T
    out_base = out_ptr + b * L * dim + d  # out 布局 [B,L,dim]: (b,l,d)=b*L*dim+l*dim+d
    for t in range(T):
        f0f = tl.load(f0_base + t).to(tl.float32)
        uv = tl.where(f0f > 0.0, 1.0, 0.0)  # torch: (f0>0).to(dtype) after fp32 promotion
        rad = (f0f * (d + 1) / sr) % 1.0  # fp32，与 torch 同 op 顺序
        # rad*upp 精确（upp 为 2 的幂），LLVM 收缩成 FMA 也不改变结果；顺序累加 bit-exact 于 scan_outer_dim
        v = rad * upp
        prev = acc
        acc = acc + v
        # mul.rn/add.rn 阻止 LLVM fp-fusion 收缩成 FMA：torch 对乘、加分别舍入
        rs = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=r,r,r",
            [rad, s_f], dtype=tl.float32, is_pure=True, pack=1)
        phase = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=r,r,r",
            [prev, rs], dtype=tl.float32, is_pure=True, pack=1)
        phase = phase + ri
        sine = tl.sin(phase * 2.0 * 3.141592653589793) * sine_amp
        n_amp = uv * noise_std + (1.0 - uv) * sine_amp / 3.0  # uv∈{0,1}: 精确
        t1 = sine * uv
        if HAS_NOISE:
            nv = tl.load(noise_ptr + b * L * dim + d + (t * upp + s0 + arange) * dim,
                         mask=smask, other=0.0).to(tl.float32)
            t2 = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;", "=r,r,r",
                [n_amp, nv], dtype=tl.float32, is_pure=True, pack=1)
            val = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;", "=r,r,r",
                [t1, t2], dtype=tl.float32, is_pure=True, pack=1)
        else:
            val = t1
        tl.store(out_base + (t * upp + s0 + arange) * dim, val, mask=smask)


def _fused_sinegen_triton(f0, upp, dim, sr, sine_amp, noise_std, rand_ini, noise):
    B, T = f0.shape
    L = T * upp
    f0 = f0.contiguous()
    out = torch.empty((B, L, dim), device=f0.device, dtype=torch.float32)
    BLOCK_S = min(128, triton.next_power_of_2(upp))
    _sinegen_kernel[(B * dim * triton.cdiv(upp, BLOCK_S),)](
        f0,
        rand_ini.contiguous() if rand_ini is not None else None,
        noise.contiguous() if noise is not None else None,
        out, T, L, float(sr), float(sine_amp), float(noise_std),
        upp=upp, dim=dim, BLOCK_S=BLOCK_S, S_BLOCKS=triton.cdiv(upp, BLOCK_S),
        HAS_RAND=rand_ini is not None, HAS_NOISE=noise is not None,
        num_warps=4 if BLOCK_S >= 128 else 2,
    )
    return out


def _fused_sinegen_pytorch(f0, upp, dim, sr, sine_amp, noise_std,
                           rand_ini=None, noise=None):
    """Pure-PyTorch fallback（frame-domain，fp32）：与 Triton kernel 及 sovits fast_sinegen fp32 下 bit 级一致。"""
    B, T = f0.shape
    f0 = f0.to(torch.float32)
    device = f0.device
    rad = (f0.unsqueeze(-1) * torch.arange(1, dim + 1, device=device, dtype=torch.float32)) / sr % 1
    if rand_ini is None:
        rand_ini = torch.zeros(B, dim, device=device, dtype=torch.float32)
    fb = torch.cumsum(rad * upp, dim=1)  # frame-domain 相位边界（顺序扫描，同 torch）
    prev = torch.zeros_like(fb)
    prev[:, 1:, :] = fb[:, :-1, :]
    arange = torch.arange(1, upp + 1, device=device, dtype=torch.float32)
    phase = (prev.unsqueeze(2) + rad.unsqueeze(2) * arange[None, None, :, None]) \
        .reshape(B, T * upp, dim) + rand_ini[:, None, :]
    sines = torch.sin(phase * 2 * math.pi) * sine_amp
    uv = (f0[:, :, None] > 0).to(sines.dtype).unsqueeze(2).expand(B, T, upp, 1).reshape(B, T * upp, 1)
    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    return sines * uv + noise_amp * (noise if noise is not None else
                                     torch.zeros(B, T * upp, dim, device=device, dtype=sines.dtype))


# torch.library custom op（Dynamo 可追踪、CUDA graph 友好）；失败置 False 走 pytorch 回退
_registered_library = None


def _register_custom_op():
    global _registered_library
    if _registered_library is not None:
        return True
    try:
        from torch.library import Library, impl
        lib = Library("usvc", "FRAGMENT")
        lib.define("fused_sinegen(Tensor f0, Tensor? rand_ini, Tensor? noise,"
                   " int upp, int dim, float sr, float sine_amp, float noise_std)"
                   " -> Tensor")

        @impl(lib, "fused_sinegen", "CUDA")
        def _sg_cuda(f0, rand_ini, noise, upp, dim, sr, sine_amp, noise_std):
            return _fused_sinegen_triton(f0, upp, dim, sr, sine_amp, noise_std, rand_ini, noise)

        impl(lib, "fused_sinegen", "Meta")(
            lambda f0, rand_ini, noise, upp, dim, sr, sine_amp, noise_std:
            torch.empty((f0.shape[0], f0.shape[1] * upp, dim), device=f0.device, dtype=torch.float32))
        _registered_library = lib
        return True
    except Exception as e:
        _registered_library = False
        print(f"[fused_sinegen] custom op 注册失败: {e}")
        return False


_register_custom_op()


def fused_sinegen(f0, upp, dim, sr, sine_amp, noise_std,
                  rand_ini=None, noise=None):
    """Fused NSF sine generation。f0 [B,T] Hz；rand_ini [B,dim]（[:,0] 必须为 0）、
    noise [B,T*upp,dim]，任一为 None 走确定性零值。返回 [B,T*upp,dim] fp32。"""
    if _registered_library is not False and f0.is_cuda and f0.dim() == 2:
        try:
            return torch.ops.usvc.fused_sinegen(
                f0, rand_ini, noise, int(upp), int(dim),
                float(sr), float(sine_amp), float(noise_std))
        except Exception:
            pass
    return _fused_sinegen_pytorch(f0, upp, dim, sr, sine_amp, noise_std, rand_ini, noise)


triton_available = _HAS_TRITON
