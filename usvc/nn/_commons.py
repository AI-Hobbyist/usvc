"""Shared torch nn helpers (W1 dedup).

Single definitions of the so-vits commons ``get_padding``/``init_weights``
(previously inlined in backends/sovits/model.py and nn/vocoder/nsf_hifigan.py)
and the LYNX/Conformer ``calc_same_padding`` (inlined in ddsp6/model.py and
reflow_vae/model.py). All four call sites import from here; sovits.model
re-exports get_padding/init_weights for rvc.model's existing import.
"""
from __future__ import annotations

__all__ = ["calc_same_padding", "get_padding", "init_weights"]


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)