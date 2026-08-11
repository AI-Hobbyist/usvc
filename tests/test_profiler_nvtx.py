"""NVTX 阶段解析的单元测试。

用造的小 CSV 测 _parse_ranges / _parse_kern_sum, 不依赖真实 nsys-rep。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usvc._profiler.nsys_nvtx import _parse_ranges, _parse_kern_sum


# 模拟 nsys stats 的输出(含 log 前缀行, 真实场景就是这样)
RANGE_CSV = """Generating SQLite file /tmp/x.sqlite from /tmp/x.nsys-rep
Processing [/tmp/x.sqlite] with [.../nvtx_pushpop_trace.py]...
Start (ns),End (ns),Duration (ns),DurChild (ns),DurNonChild (ns),Name,PID,TID,Lvl,NumChild,RangeId,ParentId,RangeStack,NameTree
1000,2000,1000,0,1000,extract,1,1,0,0,1,,:1,extract
3000,5000,2000,0,2000,run_model,1,1,0,0,2,,:2,run_model
6000,6500,500,0,500,run_model,1,1,0,0,3,,:3,run_model
7000,7100,100,0,100,stitch,1,1,0,0,4,,:4,stitch
"""

KERN_CSV = """Generating SQLite file /tmp/x.sqlite from /tmp/x.nsys-rep
Processing [/tmp/x.sqlite] with [.../nvtx_kern_sum.py]...
NVTX Range,Style,PID,TID,NVTX Inst,Kern Inst,Total Time (ns),Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Kernel Name
extract,PushPop,1,1,1,10,1000,100,100,90,110,5,kernel_a
run_model,PushPop,1,1,2,20,4000,200,200,150,250,20,kernel_b
run_model,PushPop,1,1,2,5,500,100,100,80,120,10,kernel_c
"""


def test_parse_ranges_skips_log_prefix():
    ranges = _parse_ranges(RANGE_CSV)
    # 应跳过 "Generating/Processing" 行
    assert set(ranges.keys()) == {"extract", "run_model", "stitch"}


def test_parse_ranges_aggregates_instances():
    ranges = _parse_ranges(RANGE_CSV)
    # run_model 出现 2 次, 各 2000/500 ns
    assert ranges["run_model"][0] == 2          # instances
    assert ranges["run_model"][1] == 2500       # total ns


def test_parse_ranges_durations():
    ranges = _parse_ranges(RANGE_CSV)
    assert ranges["extract"] == (1, 1000)
    assert ranges["stitch"] == (1, 100)


def test_parse_kern_sum_groups_by_stage():
    ks = _parse_kern_sum(KERN_CSV)
    assert "extract" in ks and "run_model" in ks
    assert len(ks["run_model"]) == 2  # kernel_b + kernel_c


def test_parse_kern_sum_sorted_by_time():
    ks = _parse_kern_sum(KERN_CSV)
    # run_model 内 kernel_b(4000ns) 应排在 kernel_c(500ns) 前
    assert ks["run_model"][0].name == "kernel_b"
    assert ks["run_model"][0].total_ns == 4000


def test_parse_kern_sum_fields():
    ks = _parse_kern_sum(KERN_CSV)
    k = ks["extract"][0]
    assert k.name == "kernel_a"
    assert k.count == 10
    assert k.total_ns == 1000
    assert abs(k.avg_ns - 100.0) < 0.01


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
