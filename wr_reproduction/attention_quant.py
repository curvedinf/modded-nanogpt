"""Original packed cache kernel with masked loads replacing cross-allocation pointer select.
Adapted from the pinned PR393/record360 kernel; original licenses accompany upstream/.
This avoids gfx942 compiler pointer-canonicalization failure without changing quantization.
"""
import triton
import triton.language as tl

@triton.jit
def _quantize_dual_layout_packed_batched_kernel(
    first, second, row, transposed, scale_ptr,
    M, N,
    FIRST_ROWS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr,
    SECOND_STRIDE_B: tl.constexpr,
    ROW_STRIDE_B: tl.constexpr,
    TRANSPOSED_STRIDE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Quantize a logical row-pack directly from its two source allocations: one read of each source,
    two fp8 writes (row-major, and transposed via a register-tile tl.trans)."""
    batch = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    in_first = offs_m[:, None] < FIRST_ROWS
    first_offsets = offs_m[:, None] * N + offs_n[None, :]
    second_rows = tl.maximum(offs_m - FIRST_ROWS, 0)
    second_offsets = second_rows[:, None] * N + offs_n[None, :]
    first_values=tl.load(first+batch*FIRST_STRIDE_B+first_offsets,mask=valid&in_first,other=0.)
    second_values=tl.load(second+batch*SECOND_STRIDE_B+second_offsets,mask=valid&~in_first,other=0.)
    values=first_values.to(tl.float32)+second_values.to(tl.float32)
    scaled = values / tl.load(scale_ptr + batch)
    quantized = tl.maximum(tl.minimum(scaled, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(row + batch * ROW_STRIDE_B + offs_m[:, None] * N + offs_n[None, :], quantized, mask=valid)
    transposed_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(transposed + batch * TRANSPOSED_STRIDE_B + offs_n[:, None] * M + offs_m[None, :],
             tl.trans(quantized), mask=transposed_mask)

