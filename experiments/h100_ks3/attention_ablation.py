"""Skip one attention operation while preserving bank shapes and initialization.

The forward's existing attention-free branch retains residual scaling and
the original input/ngram injections. Keep the last MUDD attention intact.
"""
import os


def install(layer):
    from track_1_short.model import gpt
    layer=int(layer)
    if layer not in (0,1,2,3,5,8):
        raise ValueError('Attention ablation must select a live non-MUDD attention layer')
    assert layer in gpt.ATTN_LAYERS
    # All bank/group tuples were created from the original topology at import.
    # Only the forward skip branch consumes this subsequently changed tuple.
    gpt.NO_ATTN_LAYERS=tuple(sorted((*gpt.NO_ATTN_LAYERS,layer)))
    if os.environ.get('RANK','0')=='0':
        print(f'[h100-ablation] attention operation skipped at layer {layer}; original banks and initialization retained',flush=True)
