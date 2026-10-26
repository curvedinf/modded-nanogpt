"""Combine prepared geometry operands with exact BF16 gather and row32 VJP.

Reuse both tested implementations. Forward equations and all selected product
tiles stay original; only the previously validated geometry reduction grouping
differs from the original64-row backward. No new mathematical approximation.
"""
from wr_reproduction import ks3_prepared_geometry as prepared
from wr_reproduction import ks3_bf16_gather_vjp32 as backward

bind = prepared.bind
prepare = prepared.prepare
EXECUTED = prepared.EXECUTED
CACHE_BYTES = prepared.CACHE_BYTES


def install():
    backward.install()
    prepared.install()
