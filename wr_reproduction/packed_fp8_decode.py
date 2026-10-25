"""Exact four-byte FN/E5 to half-value native-FNUZ register decoding.

Stored tensors and scales are unchanged. Each byte's magnitude is at most127,
so additions by127 or4 cannot carry into the next byte. The invalid flags are
0/128 per byte; subtracting their right-shifted0/1 cannot borrow across bytes.
"""
import triton as tr
import triton.language as tl


def select_packed(m,n,k,a_stride,b_stride,grouped=False):
    """Retain scalar decoding where paired evidence showed no clear gain."""
    if grouped or m<1024 or n<768 or k<128 or a_stride[1]!=1 or b_stride[0]!=1:
        return False
    # Sampled head weight gradients improved only0.39%; retain scalar there.
    return not (n==768 and k==4096)


def decode_word(word, source_e5):
    """CPU SWAR expression, independently checked against a byte oracle."""
    magnitude = word & 0x7f7f7f7f
    canonical = word & ((magnitude + 0x7f7f7f7f) | 0x7f7f7f7f)
    flags = (magnitude + (0x04040404 if source_e5 else 0x01010101)) & 0x80808080
    mask = flags | (flags - (flags >> 7))
    return (canonical & ~mask) | flags


@tr.jit
def native_packed(bits, E5:tl.constexpr, SOURCE_E5:tl.constexpr):
    # AMD LLVM rejects a <4 x i8> output for the VGPR inline-asm constraint.
    # Form explicit uint32 words so the assembly ABI uses supported scalars.
    # Split/join use adjacent logical bytes; correctness never assumes a
    # particular physical register grouping selected by Triton.
    even, odd = tl.split(tl.reshape(bits,(bits.numel//2,2)))
    b0,b2 = tl.split(tl.reshape(even,(bits.numel//4,2)))
    b1,b3 = tl.split(tl.reshape(odd,(bits.numel//4,2)))
    word = b0.to(tl.uint32) | (b1.to(tl.uint32)<<8) | (b2.to(tl.uint32)<<16) | (b3.to(tl.uint32)<<24)
    bias:tl.constexpr = '0x04040404' if SOURCE_E5 else '0x01010101'
    assembly:tl.constexpr = (
        'v_and_b32 $1, 0x7f7f7f7f, $3\n'
        'v_add_u32 $2, 0x7f7f7f7f, $1\n'
        'v_or_b32 $2, 0x7f7f7f7f, $2\n'
        'v_and_b32 $0, $3, $2\n'
        'v_add_u32 $1, ' + bias + ', $1\n'
        'v_and_b32 $1, 0x80808080, $1\n'
        'v_lshrrev_b32 $2, 7, $1\n'
        'v_sub_u32 $2, $1, $2\n'
        'v_or_b32 $2, $1, $2\n'
        'v_bfi_b32 $0, $2, $1, $0\n')
    value, scratch1, scratch2 = tl.inline_asm_elementwise(
        assembly, constraints='=&v,=&v,=&v,v', args=[word],
        dtype=(tl.uint32,tl.uint32,tl.uint32), is_pure=True, pack=1)
    even=tl.interleave(value.to(tl.uint8),(value>>16).to(tl.uint8))
    odd=tl.interleave((value>>8).to(tl.uint8),(value>>24).to(tl.uint8))
    value=tl.reshape(tl.interleave(even,odd),bits.shape)
    if E5:return value.to(tl.float8e5b16,bitcast=True)
    else:return value.to(tl.float8e4b8,bitcast=True)
