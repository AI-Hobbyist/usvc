"""Triton depthwise Conv1d（k=7，same padding，dilation），用于 diffusion ConvNext。
替代 native depthwise kernel（仅跑 ~10-15% HBM 带宽；dwconv 纯 memory-bound，fp32 累加近带宽）。
数值契约：fp32 按升序 k 累加，与 native 一致 -> bf16 autocast 下 bit 级相等。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dwconv1d_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride_xb, stride_xc, C, T,
                         DIL: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr):
    """y[b,c,t] = sum_k x[b,c,t+(k-K//2)*DIL]*w[c,k]+b[c]，zero-pad；row=pid//tiles 编码 b*C+c。"""
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(T, BLOCK_T)
    row = pid // num_tiles            # b*C + c
    t0 = (pid % num_tiles) * BLOCK_T
    b, c = row // C, row % C
    x_base = x_ptr + b * stride_xb + c * stride_xc; w_base = w_ptr + c * K
    t = t0 + tl.arange(0, BLOCK_T)
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    for k in range(K):
        off = t + (k - (K // 2)) * DIL
        mask = (t < T) & (off >= 0) & (off < T)
        xv = tl.load(x_base + off, mask=mask, other=0.0).to(tl.float32); wv = tl.load(w_base + k).to(tl.float32)
        acc += xv * wv
    if b_ptr is not None:
        acc += tl.load(b_ptr + c).to(tl.float32)
    tl.store(y_ptr + row * T + t, acc.to(x_ptr.dtype.element_ty), mask=t < T)


@triton.jit
def _dwconv1d_dgrad_kernel(dy_ptr, w_ptr, dx_ptr, stride_yb, stride_yc, C, T,
                           DIL: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr):
    """dx[b,c,t] = sum_k dy[b,c,t-(k-K//2)*DIL]*w[c,k]，zero-pad。"""
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(T, BLOCK_T)
    row = pid // num_tiles
    t0 = (pid % num_tiles) * BLOCK_T
    b, c = row // C, row % C
    y_base = dy_ptr + b * stride_yb + c * stride_yc; w_base = w_ptr + c * K
    t = t0 + tl.arange(0, BLOCK_T)
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    for k in range(K):
        off = t - (k - (K // 2)) * DIL
        mask = (t < T) & (off >= 0) & (off < T)
        yv = tl.load(y_base + off, mask=mask, other=0.0).to(tl.float32); wv = tl.load(w_base + k).to(tl.float32)
        acc += yv * wv
    tl.store(dx_ptr + row * T + t, acc.to(dy_ptr.dtype.element_ty), mask=t < T)


@triton.jit
def _dwconv1d_wgrad_kernel(dy_ptr, x_ptr, dw_ptr, stride_yb, stride_yc, stride_xb, stride_xc,
                           B, C, T, DIL: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr):
    """dw[c,k] = sum_{b,t} dy[b,c,t]*x[b,c,t+(k-K//2)*DIL]，zero-pad。pid=c*K+k。"""
    pid = tl.program_id(0)
    c, k = pid // K, pid % K
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    for bb in range(B):
        y_base = dy_ptr + bb * stride_yb + c * stride_yc; x_base = x_ptr + bb * stride_xb + c * stride_xc
        for t0 in range(0, T, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            off = t + (k - (K // 2)) * DIL
            mask = (t < T) & (off >= 0) & (off < T)
            yv = tl.load(y_base + t, mask=t < T, other=0.0).to(tl.float32)
            xv = tl.load(x_base + off, mask=mask, other=0.0).to(tl.float32)
            acc += yv * xv
    tl.store(dw_ptr + c * K + k, tl.sum(acc).to(dy_ptr.dtype.element_ty))


class _DwConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, dilation):
        ctx.save_for_backward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))
        ctx.dilation = dilation
        return _dwconv1d_fwd(x, weight, bias, dilation)

    @staticmethod
    def backward(ctx, grad_y):
        dx, dw, db = _dwconv1d_bwd(grad_y, *ctx.saved_tensors, ctx.dilation)
        return dx, dw, db, None  # dilation has no grad


def _is_fake_tensor(t):
    """FakeTensor / FunctionalTensor 检测（torch.compile 推断阶段）。"""
    try:
        from torch._subclasses import fake_tensor as ft, functional_tensor as fu
        return isinstance(t, (ft.FakeTensor, fu.FunctionalTensor))
    except Exception:
        return False


def _dwconv1d_fwd(x, weight, bias, dilation):
    """fwd：模仿 autocast（w/b -> x.dtype）。CUDA graph capture 要求 contiguous 输入。"""
    B, C, T = x.shape
    w = weight.to(x.dtype).contiguous()
    b = None if bias is None else bias.to(x.dtype)
    x = x.contiguous()
    BLOCK_T = triton.next_power_of_2(T)
    y = torch.empty_like(x)
    _dwconv1d_fwd_kernel[(B * C * triton.cdiv(T, BLOCK_T),)](
        x, w, b, y, x.stride(0), x.stride(1), C, T, DIL=dilation, K=w.shape[2], BLOCK_T=BLOCK_T)
    return y


def _dwconv1d_bwd(grad_y, x, w, b, dilation):
    """bwd -> (dx, dw, db)。w/b 为已 cast 张量。"""
    B, C, T = x.shape
    K = w.shape[2]
    BLOCK_T = triton.next_power_of_2(T); tiles = B * C * triton.cdiv(T, BLOCK_T)
    dx = torch.empty_like(x)
    _dwconv1d_dgrad_kernel[(tiles,)](
        grad_y, w, dx, grad_y.stride(0), grad_y.stride(1), C, T, DIL=dilation, K=K, BLOCK_T=BLOCK_T)
    dw = torch.zeros_like(w, dtype=torch.float32)
    _dwconv1d_wgrad_kernel[(C * K,)](
        grad_y, x, dw, grad_y.stride(0), grad_y.stride(1), x.stride(0), x.stride(1), B, C, T,
        DIL=dilation, K=K, BLOCK_T=BLOCK_T)
    return dx, dw, None if b is None else grad_y.sum(dim=(0, 2))


# torch.library custom op：Dynamo 可追踪不 graph break，使 dwconv 留在 compile/max-autotune 图内。
# NOTE: register_autograd 会静默拖慢同进程无关的 max-autotune 图（67.9 -> 95.9 ms/step），故 backward 用 allow_in_graph Function。
_registered_library = None


def _register_custom_op():
    global _registered_library
    if _registered_library is not None:
        return True
    try:
        from torch.library import Library, impl, register_autograd
        lib = Library("usvc", "FRAGMENT")
        lib.define("dwconv1d(Tensor x, Tensor w, Tensor b, int dilation) -> Tensor")
        impl(lib, "dwconv1d", "CUDA")(_dwconv1d_fwd)
        impl(lib, "dwconv1d", "Meta")(lambda x, w, b, dilation: torch.empty_like(x))

        def _dwc_setup(ctx, inputs, output):
            x, w, b, dilation = inputs
            ctx.save_for_backward(x, w.to(x.dtype), None if b is None else b.to(x.dtype))
            ctx.dilation = dilation

        @torch._dynamo.disable
        def _dwc_backward(ctx, grad_y):
            x, w, b = ctx.saved_tensors
            if _is_fake_tensor(grad_y):  # compile 推断梯度形状：只回零梯度，不执行 Triton
                return (torch.zeros_like(x), torch.zeros_like(w),
                        None if b is None else torch.zeros_like(b), None)
            dx, dw, db = _dwconv1d_bwd(grad_y, x, w, b, ctx.dilation)
            return dx, dw, db, None  # dilation has no grad

        register_autograd(torch.ops.usvc.dwconv1d.default, _dwc_backward, setup_context=_dwc_setup)
        _registered_library = lib
        return True
    except Exception as e:
        print(f"[dwconv1d] custom op 注册失败: {e}")
        _registered_library = False
        return False


_register_custom_op()


def dwconv1d(x, weight, bias, dilation):
    """公开入口：已注册则走 custom op（compile 友好），否则 eager fn。"""
    if _registered_library is not False:
        try:
            return torch.ops.usvc.dwconv1d(x, weight, bias, dilation)
        except Exception:
            pass
    return _DwConv1d.apply(x, weight, bias, dilation)


class DwConv1dFn:
    @staticmethod
    def apply(x, weight, bias, dilation):
        return dwconv1d(x, weight, bias, dilation)


triton_available = True
