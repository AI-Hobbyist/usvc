"""usvc module."""
from __future__ import annotations

# import usvc 时检查 triton 可用性（缺失仅提示，不阻断 import）。
from ._envcheck import check_once

check_once()
