import triton as tr
import triton.language as tl
from triton.language.extra import libdevice

@tr.jit
def _native(bits, E5:tl.constexpr, SOURCE_E5:tl.constexpr):
    # Preserve invalid values as invalid, rather than interpreting FN NaN bytes
    # as the additional finite FNUZ codes. Finite values retain identical bits.
    magnitude=bits&127
    if SOURCE_E5:invalid=magnitude>=124
    else:invalid=magnitude==127
    bits=tl.where(invalid,128,tl.where(bits==128,0,bits)).to(tl.uint8)
    if E5:return bits.to(tl.float8e5b16,bitcast=True)
    else:return bits.to(tl.float8e4b8,bitcast=True)

