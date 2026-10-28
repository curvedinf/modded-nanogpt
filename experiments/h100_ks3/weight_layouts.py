"""Coalesced weight-cache construction with Hopper TMA transpose stores."""
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@tr.jit
def kernel(Q,UP,UPT,DOWN,DOWNT,SCALES,N:tl.constexpr,D:tl.constexpr,
           TILE:tl.constexpr,PACK_SCALE:tl.constexpr):
    slot=tl.program_id(1);down=slot%16>=8;expert=slot//16*8+slot%8
    rows=tl.where(down,D,N);cols=tl.where(down,N,D)
    row=tl.program_id(0)//(cols//TILE)*TILE
    col=tl.program_id(0)%(cols//TILE)*TILE
    rr=row+tl.arange(0,TILE);cc=col+tl.arange(0,TILE)
    stride=N*D+(64 if PACK_SCALE else 0)
    values=tl.load(Q+slot*stride+rr[:,None]*cols+cc[None,:])
    if down:
        DOWN.store([expert*D+row,col],values)
        DOWNT.store([expert*N+col,row],tl.trans(values))
    else:
        UP.store([expert*N+row,col],values)
        UPT.store([expert*D+col,row],tl.trans(values))
    if PACK_SCALE and tl.program_id(0)==0:
        trailer=(Q+slot*stride+N*D).to(tl.pointer_type(tl.float32))
        tl.store(SCALES+slot,tl.load(trailer))


def build(q,up,upt,down,downt,n,scales,*,packed=False,tile=128):
    d=768
    assert n%tile==0 and d%tile==0
    desc=[TensorDescriptor.from_tensor(a.view(-1,width),[tile,tile])
          for a,width in [(up,d),(upt,n),(down,n),(downt,d)]]
    kernel[(n*d//(tile*tile),80)](q,*desc,scales,n,d,tile,packed,num_warps=4)
