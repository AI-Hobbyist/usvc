"""conv/gemm 的 shape 捕获 + FLOPs 计算。

补齐 nsys 拿不到的能力:conv/gemm 的 tensor shapes。nsys 只有 kernel 名
和 grid/block,算不出 FLOPs。本模块用 torch 的 TorchDispatchMode 在 Python
层 hook aten compute op,记录 (op, shape, count),采集后算 FLOPs。

用法(profile 时):
    cap = ShapeCapture()
    with cap:
        engine.convert(...)   # 所有 conv/gemm 被 hook
    ops = cap.compute_ops      # list[ComputeOp]

FLOPs 公式:
- conv2d: 2 * Cout * Hout * Wout * (Cin/groups * Kh * Kw)
- conv1d: 2 * Cout * Lout * (Cin/groups * K)
- mm:     2 * M * N * K
- bmm:    2 * B * M * N * K
- addmm:  2 * M * N * K
- linear: 2 * B(*...) * out_features * in_features
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComputeOp:
    """一个 compute-heavy op 的聚合统计(按 op+shape 分组)。"""
    op: str                    # aten::conv2d / aten::bmm / ...
    shapes: tuple              # ((in0_shape), (in1_shape), ...)
    count: int = 0
    flops_per_call: Optional[int] = None  # 单次调用的 FLOPs
    # GPU 时间由渲染层从 nsys kernel 数据反查填充(按 conv/gemm kernel 名聚合)

    @property
    def total_flops(self) -> int:
        return (self.flops_per_call or 0) * self.count


_COMPUTE_OPS = {
    "aten::convolution", "aten::conv1d", "aten::conv2d", "aten::conv3d",
    "aten::_convolution", "aten::mm", "aten::bmm", "aten::addmm",
    "aten::baddbmm", "aten::matmul", "aten::linear",
}


class ShapeCapture:
    """TorchDispatchMode hook, 捕获 conv/gemm 的 input shapes。

    profile 时用 `with cap:` 包裹推理, 非 profile 时零开销(不激活)。
    """

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        self._raw: dict[tuple, list] = {}  # (op, shapes) -> [count]
        self._dispatch_cls = TorchDispatchMode
        self._mode = None

    def __enter__(self):
        cap = self

        class _Mode(self._dispatch_cls):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                fname = func.name()
                if fname in _COMPUTE_OPS:
                    cap._record(fname, args)
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        if self._mode is not None:
            self._mode.__exit__(*exc)
        return False

    def _record(self, fname: str, args: tuple):
        """记录一个 compute op 的 shapes。按 (op, shapes) 去重计数。"""
        shapes = []
        for a in args:
            if hasattr(a, "shape"):
                shapes.append(tuple(int(s) for s in a.shape))
            elif isinstance(a, (list, tuple)):
                # stride/padding 等参数, 跳过
                break
        if not shapes:
            return
        key = (fname, tuple(shapes))
        rec = self._raw.setdefault(key, [0])
        rec[0] += 1

    @property
    def compute_ops(self) -> list[ComputeOp]:
        """返回带 FLOPs 的 ComputeOp 列表, 按 total_flops 降序。"""
        out = []
        for (fname, shapes_tuple), (count,) in self._raw.items():
            flops = _compute_flops(fname, list(shapes_tuple))
            out.append(ComputeOp(
                op=fname, shapes=shapes_tuple, count=count,
                flops_per_call=flops,
            ))
        out.sort(key=lambda o: o.total_flops, reverse=True)
        return out


def _compute_flops(op: str, shapes: list[tuple]) -> Optional[int]:
    """从 op 名 + input shapes 算单次调用 FLOPs。

    返回 None 表示无法计算(非 conv/gemm 或 shape 不全)。
    """
    shapes = [s for s in shapes if s]  # 去掉空 shape(标量参数)
    if not shapes:
        return None

    try:
        if op in ("aten::conv2d", "aten::convolution", "aten::_convolution"):
            return _flops_conv(shapes, ndim=2)
        if op in ("aten::conv1d",):
            return _flops_conv(shapes, ndim=1)
        if op in ("aten::conv3d",):
            return _flops_conv(shapes, ndim=3)
        if op in ("aten::mm",):
            # mm: (M,K) x (K,N)
            if len(shapes) >= 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2:
                m, k = shapes[0]; k2, n = shapes[1]
                if k == k2:
                    return 2 * m * n * k
        if op in ("aten::bmm", "aten::baddbmm"):
            # bmm: (B,M,K) x (B,K,N)
            if len(shapes) >= 2 and len(shapes[0]) == 3 and len(shapes[1]) == 3:
                b, m, k = shapes[0]; b2, k2, n = shapes[1]
                if b == b2 and k == k2:
                    return 2 * b * m * n * k
        if op == "aten::addmm":
            # addmm: bias(M,) x (M,K) x (K,N) -> 2*M*N*K
            # shapes[0] 可能是 bias, 找 2D 的两个
            mats = [s for s in shapes if len(s) == 2]
            if len(mats) >= 2:
                m, k = mats[0]; k2, n = mats[1]
                if k == k2:
                    return 2 * m * n * k
        if op == "aten::matmul":
            if len(shapes) >= 2:
                a, b = shapes[0], shapes[1]
                if len(a) == 2 and len(b) == 2:
                    return 2 * a[0] * b[1] * a[1]
                if len(a) == 3 and len(b) == 3:
                    if a[0] == b[0] and a[2] == b[1]:
                        return 2 * a[0] * a[1] * b[2] * a[2]
        if op == "aten::linear":
            # linear: (*, in) x (out, in) -> 2 * prod(*) * out * in
            if len(shapes) >= 2:
                inp, w = shapes[0], shapes[1]
                if len(w) == 2 and len(inp) >= 1 and inp[-1] == w[1]:
                    batch = 1
                    for d in inp[:-1]:
                        batch *= d
                    return 2 * batch * w[0] * w[1]
    except (IndexError, TypeError):
        return None
    return None


def _flops_conv(shapes: list[tuple], ndim: int) -> Optional[int]:
    """conv FLOPs = 2 * Cout * spatial_out * (Cin/groups * K)。

    shapes: [input(N,C,*spatial), weight(Cout,Cin/groups,*kernel)]
    不含 padding/stride(影响 output spatial, 用 input 近似, 误差通常 <5%)。
    """
    if len(shapes) < 2:
        return None
    inp, wt = shapes[0], shapes[1]
    if len(inp) < 2 or len(wt) < 2:
        return None
    n = inp[0]
    cout = wt[0]
    cin_per_group = wt[1]
    # kernel 体积
    kernel = 1
    for k in wt[2:2 + ndim]:
        kernel *= k
    # 输出 spatial (近似 = input spatial, padding/stride 影响小)
    spatial_out = 1
    for s in inp[2:2 + ndim]:
        spatial_out *= s
    return 2 * cout * cin_per_group * kernel * spatial_out * n
