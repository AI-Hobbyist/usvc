"""Offline inference: features -> backend model -> vocoder -> wav."""
from __future__ import annotations

from .engine import InferenceEngine

__all__ = ["InferenceEngine"]
