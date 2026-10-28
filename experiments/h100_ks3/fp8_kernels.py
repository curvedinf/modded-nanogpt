"""Native H100 FP8 producers, dual layouts and grouped weight gradients."""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale

@tr.jit
def pack_input(A,O,OT,SRC,SCALE,M:tl.constexpr,ROWS:tl.constexpr,D:tl.constexpr=768,BM:tl.constexpr=128):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*32+tl.arange(0,32)
    source=tl.load(SRC+rows,rows<M,other=ROWS)
    value=tl.load(A+source[:,None].to(tl.int64)*D+cols[None,:],(rows[:,None]<M)&(source[:,None]<ROWS)&(cols[None,:]<D),other=0.).to(tl.float32)
    s=_half_scale(tl.max(tl.abs(value),1))
    value=tl.minimum(tl.maximum(libdevice.nearbyint(value/s[:,None]),-127.),127.)*s[:,None]
    value=tl.minimum(tl.maximum(value/tl.load(SCALE),-448.),448.).to(tl.float8e4nv)
    mask=(rows[:,None]<M)&(cols[None,:]<D)
    # Requires the corrected ptxas >=13.3, enforced by the FP8 model overlay.
    tl.store(O+rows[:,None]*D+cols[None,:],value,mask)
    tl.store(OT+cols[None,:]*M+rows[:,None],value,mask)

@tr.jit
def publish(BANK,Q,SCALE,PART,LO:tl.constexpr,N:tl.constexpr,D:tl.constexpr=768,BR:tl.constexpr=128):
    slot=LO+tl.program_id(1);down=(slot%16)>=8
    r=tl.where(down,D,N);c=tl.where(down,N,D)
    tile=tl.program_id(0);rows=(tile//(c//32))*BR+tl.arange(0,BR);cols=(tile%(c//32))*32+tl.arange(0,32)
    mask=(rows[:,None]<r)&(cols[None,:]<c)
    off=tl.where(down,cols[None,:]*D+rows[:,None],rows[:,None]*D+cols[None,:])
    a=tl.load(BANK+slot*N*D+off,mask,other=0.).to(tl.float32)
    s=_half_scale(tl.max(tl.abs(a),1))
    value=tl.minimum(tl.maximum(libdevice.nearbyint(a/s[:,None]),-127.),127.)*s[:,None]
    q=tl.minimum(tl.maximum(value/tl.load(SCALE+slot),-448.),448.).to(tl.float8e4nv)
    tl.store(Q+slot*N*D+rows[:,None]*c+cols[None,:],q,mask)
    tl.store(PART+slot*(N*D//(BR*32))+tile,tl.max(tl.max(tl.abs(value),1),0))

@tr.jit
def layouts(Q,UP,UPT,DOWN,DOWNT,N:tl.constexpr,D:tl.constexpr=768,B:tl.constexpr=1024):
    slot=tl.program_id(1);down=(slot%16)>=8;e=slot//16*8+slot%8
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N*D
    q=tl.load(Q+slot*N*D+i,valid,other=0.)
    if down:
        tl.store(DOWN+e*N*D+i,q,valid)
        tl.store(DOWNT+e*N*D+(i%N)*D+i//N,q,valid)
    else:
        tl.store(UP+e*N*D+i,q,valid)
        tl.store(UPT+e*N*D+(i%D)*N+i//D,q,valid)

@tr.jit
def weight_gradient(A,B,O,OFFS,SA,SB,FLAG,H:tl.constexpr,D:tl.constexpr,M:tl.constexpr,
                    BM:tl.constexpr=128,BN:tl.constexpr=128,BK:tl.constexpr=128):
    e=tl.program_id(2);lo=0
    if e>0:lo=tl.load(OFFS+e-1)
    hi=tl.load(OFFS+e)
    h=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(lo,hi,BK):
        k=start+kk
        a=tl.load(A+h[:,None]*M+k[None,:],(h[:,None]<H)&(k[None,:]<hi),other=0.)
        b=tl.load(B+d[None,:]*M+k[:,None],(d[None,:]<D)&(k[:,None]<hi),other=0.)
        acc=tl.dot(a,b,acc)
    acc=acc*tl.load(SA)*tl.load(SB)
    offsets=e*H*D+h[:,None]*D+d[None,:];mask=(h[:,None]<H)&(d[None,:]<D)
    if tl.load(FLAG):acc+=tl.load(O+offsets,mask,other=0.).to(tl.float32)
    tl.store(O+offsets,acc,mask)

@tr.jit
def gather_gate(G,Y,SRC,SCALE,GG,GT,DS,GS,ROWS:tl.constexpr,M:tl.constexpr,D:tl.constexpr=768,BM:tl.constexpr=4):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.arange(0,1024)
    source=tl.load(SRC+rows,rows<M,other=ROWS);valid=(rows<M)&(source<ROWS)
    grad=tl.load(G+source[:,None].to(tl.int64)*D+cols[None,:],valid[:,None]&(cols[None,:]<D),other=0.).to(tl.float32)
    y=tl.load(Y+rows[:,None]*D+cols[None,:],valid[:,None]&(cols[None,:]<D),other=0.).to(tl.float32)
    scale=tl.load(SCALE+source,valid,other=0.).to(tl.bfloat16).to(tl.float32)
    value=grad*scale[:,None]/tl.load(GS)
    packed=tl.minimum(tl.maximum(value,-57344.),57344.).to(tl.float8e5)
    mask=(rows[:,None]<M)&(cols[None,:]<D)
    tl.store(GG+rows[:,None]*D+cols[None,:],packed,mask)
    tl.store(GT+cols[None,:]*M+rows[:,None],packed,mask)
    tl.store(DS+rows,tl.sum(grad*y,1),rows<M)
