"""usvc profiler 可视化渲染器。

把 NsysReport + occupancy + roofline 反推画成单文件 HTML(plotly)。
四个 panel:
  1. §0 判据仪表盘(occupancy/胶水/传输)
  2. Roofline 散点图(只画可信的 elementwise/reduce;conv/gemm 按耗时大小另区显示)
  3. Kernel 构成瀑布图(按类别堆叠)
  4. Top-N kernel 表(含 occupancy)

参考: docs/PROFILER_DESIGN.md §6
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .nsys_parser import NsysReport, KernelRow
from .occupancy import theoretical_occupancy
from ._kernel_flops import classify_kernel, CUDNN_CUBLAS, ELEMENTWISE_REDUCE, MEMCPY_TRANSFER, TRITON_FUSED, OTHER
from .nsys_flops import estimate_kernel_flops_from_grid, estimate_kernel_bytes_from_grid


# 类别 → 显示颜色
CAT_COLOR = {
    CUDNN_CUBLAS: "#4e79a7",      # 蓝(conv/gemm)
    ELEMENTWISE_REDUCE: "#f28e2b", # 橙(胶水)
    TRITON_FUSED: "#59a14f",      # 绿
    MEMCPY_TRANSFER: "#76b7b2",   # 青
    "sync/idle": "#b07aa1",       # 紫
    OTHER: "#bab0ac",             # 灰
}


def render_html(
    report: NsysReport,
    output_path: str | Path,
    *,
    title: str = "usvc inference profile",
    arch: tuple[int, int] = (8, 6),
    input_duration_s: Optional[float] = None,
) -> Path:
    """渲染单文件 HTML 报告。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    hw = report.hardware
    total_ms = report.total_kernel_ns / 1e6

    # ---- 给每个 kernel 补 category / flops / bytes / occupancy ----
    rows = []
    for k in report.kernels:
        cat = classify_kernel(k.name)
        occ = theoretical_occupancy(int(k.registers_per_thread), k.block, arch)
        flops = estimate_kernel_flops_from_grid(k.name, k.grid, k.block)
        bytes_io = estimate_kernel_bytes_from_grid(k.name, k.grid, k.block)
        rows.append({
            "name": k.name, "cat": cat, "ms": k.total_ns / 1e6,
            "pct": 100 * k.total_ns / report.total_kernel_ns,
            "count": k.count, "regs": k.registers_per_thread,
            "occ": occ, "flops": flops, "bytes": bytes_io,
            "grid": k.grid, "block": k.block,
        })

    # ---- Panel 1: §0 判据(occupancy 加权 + 胶水占比 + 传输占比) ----
    # 加权 occupancy(按 kernel 耗时加权)
    occ_known = [(r["ms"], r["occ"]) for r in rows if r["occ"] >= 0]
    weighted_occ = (sum(m*o for m,o in occ_known) / sum(m for m,_ in occ_known)) if occ_known else 0
    glue_ms = sum(r["ms"] for r in rows if r["cat"] in (ELEMENTWISE_REDUCE, OTHER))
    glue_pct = 100 * glue_ms / total_ms if total_ms else 0
    transfer_pct = 100 * (report.memcpy.total_ns / 1e6) / total_ms if total_ms else 0

    fig1 = go.Figure()
    criteria = [
        ("SM occupancy (理论加权)", weighted_occ, 85, "%", "理论值, 非ncu实测"),
        ("胶水占比 (elementwise+other)", glue_pct, 5, "%", "目标≤5%"),
        ("传输占比 (memcpy)", transfer_pct, 3, "%", "目标<3%"),
    ]
    for i, (name, val, target, unit, note) in enumerate(criteria):
        color = "#2ca02c" if val >= target else ("#ff7f0e" if val >= target*0.7 else "#d62728")
        # occupancy 是"越高越好"(绿=达标), 胶水/传输是"越低越好"(绿=达标), 处理颜色逻辑
        if name.startswith("SM"):
            color = "#2ca02c" if val >= target else "#d62728"
        else:
            color = "#2ca02c" if val <= target else ("#ff7f0e" if val <= target*2 else "#d62728")
        fig1.add_trace(go.Bar(
            x=[name], y=[val], name=name, marker_color=color,
            text=[f"{val:.1f}{unit}<br><sub>{note}</sub>"],
            textposition="outside", hovertemplate=f"{name}: {val:.1f}{unit}<br>{note}<extra></extra>",
        ))
    fig1.update_layout(
        title=f"§0 判据 (GPU {total_ms:.0f}ms 总, {hw.name} {hw.sm_count}SM, {hw.memory_bandwidth_gbs:.0f}GB/s)",
        yaxis_title="%", showlegend=False, height=280,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ---- Panel 2: Roofline 散点(只画可信 elementwise/reduce) ----
    bw_peak = hw.memory_bandwidth_gbs  # GB/s
    # 算力峰值近似: conv/gemm fp16 ~ 65 TFLOPS(A10G). elementwise 无算力瓶颈, 用带宽线为主。
    compute_peak_tflops = 65.0
    fig2 = go.Figure()
    # 可信点(elementwise: 有 bytes, flops)
    plot_pts = [r for r in rows if r["flops"] and r["bytes"] and r["ms"] > 0]
    if plot_pts:
        # 每点: 算力强度(flops/bytes), 实测吞吐(flops/s 或 GB/s)
        ai = [r["flops"]/r["bytes"] for r in plot_pts]  # FLOP/byte
        # 用带宽吞吐(GB/s)衡量, 因为 elementwise 是 bandwidth bound
        gbps = [r["bytes"]/(r["ms"]*1e-3)/1e9 for r in plot_pts]
        sizes = [max(6, min(40, r["pct"]*3)) for r in plot_pts]
        hovers = [f"{r['name'][:60]}<br>占用 {r['pct']:.1f}% ({r['ms']:.1f}ms)<br>"
                  f"occupancy {r['occ']:.0f}%<br>AI={r['flops']//r['bytes']:.1f} FLOP/byte"
                  for r in plot_pts]
        fig2.add_trace(go.Scatter(
            x=ai, y=gbps, mode="markers", marker=dict(size=sizes, color="#f28e2b", opacity=0.7),
            text=hovers, hoverinfo="text", name="elementwise/reduce",
        ))
    # 带宽屋顶线
    fig2.add_hline(y=bw_peak, line_dash="dash", line_color="#d62728",
                   annotation_text=f"带宽峰值 {bw_peak:.0f} GB/s")
    fig2.update_layout(
        title="Roofline (elementwise 类, 从 grid 反推)",
        xaxis_title="算力强度 (FLOP/byte, 对数)", yaxis_title="吞吐 (GB/s, 对数)",
        xaxis_type="log", yaxis_type="log", height=380,
        margin=dict(l=50, r=20, t=60, b=40), showlegend=True,
    )

    # ---- Panel 2.5: conv/gemm 算力利用率 (来自 shapes.json) ----
    gemm_html = ""
    if report.compute_ops:
        # 按 op 聚合 FLOPs, 匹配 nsys 里的 conv/gemm kernel 时间
        conv_gemm_ms = sum(r["ms"] for r in rows if r["cat"] == CUDNN_CUBLAS)
        total_flops = sum(o.total_flops for o in report.compute_ops)
        # 整体利用率
        achieved_tflops = (total_flops / (conv_gemm_ms * 1e-3) / 1e12) if conv_gemm_ms > 0 else 0
        # 按层类型分组, 看哪些层贡献最多 FLOPs
        op_groups = {}
        for o in report.compute_ops:
            key = o.op.replace("aten::", "")
            g = op_groups.setdefault(key, {"flops": 0, "count": 0})
            g["flops"] += o.total_flops
            g["count"] += o.count
        groups_sorted = sorted(op_groups.items(), key=lambda x: -x[1]["flops"])[:8]
        fig_g = go.Figure()
        for name, g in groups_sorted:
            gflops = g["flops"] / 1e9
            fig_g.add_trace(go.Bar(
                x=[name], y=[gflops], name=name,
                text=[f"{gflops:.0f}G\nx{g['count']}"], textposition="inside",
            ))
        # 计算屋顶线(整体达到的算力)
        fig_g.add_hline(y=achieved_tflops, line_dash="dot", line_color="#1f77b4",
                        annotation_text=f"整体 {achieved_tflops:.1f} TFLOPS")
        fig_g.update_layout(
            title=f"conv/gemm FLOPs 分布 (总 {total_flops/1e12:.1f} TFLOPs, kernel {conv_gemm_ms:.0f}ms → {achieved_tflops:.1f} TFLOPS)",
            yaxis_title="GFLOPs", barmode="stack", height=320, showlegend=False,
            margin=dict(l=50, r=20, t=60, b=60),
        )
        # 利用率文字
        util_tf32 = 100 * achieved_tflops / 34.1 if achieved_tflops else 0
        util_fp16 = 100 * achieved_tflops / 68.0 if achieved_tflops else 0
        gemm_html = f"""
<h2>2.5 conv/gemm 算力 (shapes 来自 torch dispatcher)</h2>
{fig_g.to_html(full_html=False, include_plotlyjs=False)}
<p class="note">整体算力 {achieved_tflops:.1f} TFLOPS = 总FLOPs/conv-gemm kernel 时间。
利用率: <b>{util_tf32:.0f}% of TF32 peak(34.1)</b> | {util_fp16:.0f}% of FP16 peak(68)。
注: 粗粒度(op的FLOPs vs 同类kernel时间), 若含小矩阵 op 会被高估的 kernel 时间稀释。</p>
"""

    # ---- Panel 3: kernel 构成(类别堆叠条) ----
    cat_ms = {}
    for r in rows:
        cat_ms[r["cat"]] = cat_ms.get(r["cat"], 0) + r["ms"]
    cats_sorted = sorted(cat_ms.items(), key=lambda x: -x[1])
    fig3 = go.Figure()
    for cat, ms in cats_sorted:
        fig3.add_trace(go.Bar(
            x=[cat], y=[ms], name=cat, marker_color=CAT_COLOR.get(cat, "#999"),
            text=[f"{ms:.0f}ms ({100*ms/total_ms:.0f}%)"], textposition="inside",
        ))
    fig3.update_layout(
        title=f"Kernel 构成 (总 {total_ms:.0f}ms)", yaxis_title="ms",
        barmode="stack", height=320, showlegend=False,
        margin=dict(l=50, r=20, t=60, b=60),
    )

    # ---- Panel 3.5: NVTX 阶段拆解(extract/run_model/stitch)----
    stage_html = ""
    if report.stages:
        st_total = sum(s.total_ms for s in report.stages)
        # 阶段占比条
        fig_s = go.Figure()
        for s in report.stages:
            fig_s.add_trace(go.Bar(
                x=[s.name], y=[s.total_ms], name=s.name,
                text=[f"{s.total_ms:.0f}ms ({100*s.total_ms/st_total:.0f}%) x{s.instances}"],
                textposition="inside",
            ))
        fig_s.update_layout(
            title=f"阶段拆解 (NVTX, 总 {st_total:.0f}ms)", yaxis_title="ms",
            barmode="stack", height=300, showlegend=False,
            margin=dict(l=50, r=20, t=60, b=40),
        )
        # 每 stage 的 top 5 kernel 表
        stage_tables = ""
        for s in report.stages:
            stage_tables += f"<h3 style='margin:15px 0 5px'>{s.name} " \
                f"<span style='color:#666;font-weight:normal'>({s.total_ms:.0f}ms, {s.instances}次)</span></h3>"
            if not s.kernels:
                stage_tables += "<p class='note'>无 GPU kernel (纯 CPU 阶段)</p>"
                continue
            stage_tables += "<table style='border-collapse:collapse;font-size:11px;width:100%'>"
            stage_tables += "<tr style='background:#f5f5f5'><th style='padding:3px;text-align:left'>kernel</th>" \
                "<th>ms</th><th>count</th></tr>"
            for k in s.kernels[:5]:
                kname = k.name[:65]
                kms = k.total_ns / 1e6
                stage_tables += (
                    "<tr style='border-bottom:1px solid #eee'>"
                    "<td style='padding:2px;max-width:450px'>" + kname + "</td>"
                    "<td style='text-align:right'>" + format(kms, ".1f") + "</td>"
                    "<td style='text-align:right'>" + str(k.count) + "</td></tr>"
                )
            stage_tables += "</table>"
        stage_html = f"""
<h2>3. 阶段拆解 (extract / run_model / stitch)</h2>
{fig_s.to_html(full_html=False, include_plotlyjs=False)}
<p class="note">由 NVTX range 标记拆分(需 engine 插入 nvtx 标记)。
run_model 含 forward; extract 含特征提取(contentvec/f0); stitch 含 crossfade 拼接。</p>
{stage_tables}
"""

    # ---- Panel 4: Top kernel 表(HTML) ----
    top = rows[:15]
    table_html = "<table style='border-collapse:collapse;font-size:12px;width:100%'>"
    table_html += "<tr style='background:#f0f0f0'><th style='padding:4px;text-align:left'>kernel</th>" \
                  "<th>ms</th><th>%</th><th>regs</th><th>occupancy</th><th>grid</th><th>block</th></tr>"
    for r in top:
        occ_s = f"{r['occ']:.0f}%" if r['occ'] >= 0 else "?"
        occ_color = "#2ca02c" if r['occ'] >= 85 else ("#ff7f0e" if r['occ'] >= 50 else "#d62728")
        table_html += (
            f"<tr style='border-bottom:1px solid #eee'>"
            f"<td style='padding:3px;max-width:400px;overflow:hidden;text-overflow:ellipsis'>{r['name'][:70]}</td>"
            f"<td style='text-align:right'>{r['ms']:.1f}</td>"
            f"<td style='text-align:right'>{r['pct']:.1f}</td>"
            f"<td style='text-align:right'>{r['regs']:.0f}</td>"
            f"<td style='text-align:right;color:{occ_color}'>{occ_s}</td>"
            f"<td style='text-align:right'>{r['grid'][0]}×{r['grid'][1]}×{r['grid'][2]}</td>"
            f"<td style='text-align:right'>{r['block'][0]}×{r['block'][1]}×{r['block'][2]}</td>"
            f"</tr>")
    table_html += "</table>"

    dur_note = f" (输入 {input_duration_s:.0f}s)" if input_duration_s else ""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;max-width:1200px}}
h2{{border-bottom:2px solid #333;padding-bottom:5px;margin-top:30px}}
.note{{color:#666;font-size:13px}}</style>
</head><body>
<h1>{title}{dur_note}</h1>
<p class="note">硬件: {hw.name} | {hw.sm_count} SM | 带宽峰值 {hw.memory_bandwidth_gbs:.0f} GB/s |
GPU 总耗时 {total_ms:.0f}ms | kernel 种类 {len(report.kernels)} |
memcpy {report.memcpy.total_bytes/1e9:.2f}GB ({report.memcpy.total_ns/1e6:.0f}ms, {transfer_pct:.1f}%)</p>

<h2>1. §0 判据</h2>
{fig1.to_html(full_html=False, include_plotlyjs='cdn')}
<p class="note">SM occupancy 为理论计算值(从 nsys registers+block, 非ncu实测)。
带宽/传输来自 nsys 真实 memcpy bytes。</p>

<h2>2. Roofline</h2>
{fig2.to_html(full_html=False, include_plotlyjs=False)}
<p class="note">橙色点 = elementwise/reduce(从 grid 反推 FLOPs/Bytes, 可信)。
点大小 = 耗时占比。</p>
{gemm_html}
<h2>3. Kernel 构成</h2>
{fig3.to_html(full_html=False, include_plotlyjs=False)}
{stage_html}
<h2>4. Top 15 kernel</h2>
{table_html}
<p class="note">occupancy 颜色: 绿≥85% | 橙50-85% | 红&lt;50%。
深挖单 kernel: 用 Nsight Systems UI 打开 .nsys-rep 文件。</p>
</body></html>"""

    out.write_text(html, encoding="utf-8")
    return out
