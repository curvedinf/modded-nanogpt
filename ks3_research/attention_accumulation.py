"""Ordered, non-atomic accumulation of grouped attention K/V gradients.

Each output element owns its reverse chunk loop. BF16 rounds after EVERY
addition, as the existing sequence of slice.add_ calls does. No reassociation
or FP32 accumulator changes the overlap reduction semantics.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _ordered_add(DEST, SOURCE, BOUNDS,
                 START, END, COUNT,
                 HEADS: tl.constexpr, WIDTH: tl.constexpr,
                 D0, D1, D2, S0, S1, S2, S3,
                 BLOCK: tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=index<(END-START)*HEADS*WIDTH
    row=START+index//(HEADS*WIDTH)
    head=index//WIDTH%HEADS
    dim=index%WIDTH
    offset=row*D0+head*D1+dim*D2
    value=tl.load(DEST+offset,valid,other=0.)
    # Offsets/strides vary with each document packing: keep them runtime values
    # so new documents never cause compilation. The cast
    # remains inside the loop, retaining each BF16 store's rounding point.
    for reverse in range(COUNT):
        chunk=COUNT-1-reverse
        lo=tl.load(BOUNDS+chunk*2)
        hi=tl.load(BOUNDS+chunk*2+1)
        inside=valid & (row>=lo) & (row<hi)
        add=tl.load(SOURCE+chunk*S0+head*S1+(row-lo)*S2+dim*S3,inside,other=0.)
        updated=(value.to(tl.float32)+add.to(tl.float32)).to(DEST.dtype.element_ty)
        value=tl.where(inside,updated,value)
    tl.store(DEST+offset,value,valid)


def ordered_add(destination,source,chunks,bounds=None):
    """Add [chunk,head,padded_row,dim] into strided [row,head,dim]."""
    if not chunks:
        return
    if destination.device.type!='cuda':
        for i in reversed(range(len(chunks))):
            _,hi,lo=chunks[i]
            destination[lo:hi].add_(source[i,:,:hi-lo,:destination.shape[-1]].transpose(0,1))
        return
    if destination.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('Ordered GPU accumulation supports FP32 and BF16')
    if bounds is None:
        bounds=torch.tensor([(lo,hi) for _,hi,lo in chunks],device=destination.device)
    start=min(lo for _,_,lo in chunks)
    end=max(hi for _,hi,_ in chunks)
    heads,width=destination.shape[1:]
    _ordered_add[(tr.cdiv((end-start)*heads*width,256),)](
        destination,source,bounds,start,end,len(chunks),heads,width,
        *destination.stride(),*source.stride(),256)
