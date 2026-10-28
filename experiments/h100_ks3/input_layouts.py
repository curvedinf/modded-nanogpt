"""Hopper TMA stores for the GS32 input and routed-gradient dual layouts."""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from ks3_research.vendor.ks3_int8 import _half_scale


@tr.jit
def input_kernel(A,O,OT,SRC,SCALE,M:tl.constexpr,ROWS:tl.constexpr,BM:tl.constexpr=128,D:tl.constexpr=768):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*32+tl.arange(0,32)
    source=tl.load(SRC+rows,rows<M,other=ROWS)
    value=tl.load(A+source[:,None].to(tl.int64)*D+cols[None,:],
                  (rows[:,None]<M)&(source[:,None]<ROWS),other=0.).to(tl.float32)
    s=_half_scale(tl.max(tl.abs(value),1))
    value=tl.minimum(tl.maximum(libdevice.nearbyint(value/s[:,None]),-127.),127.)*s[:,None]
    value=tl.minimum(tl.maximum(value/tl.load(SCALE),-448.),448.).to(tl.float8e4nv)
    O.store([tl.program_id(0)*BM,tl.program_id(1)*32],value)
    OT.store([tl.program_id(1)*32,tl.program_id(0)*BM],tl.trans(value))


@tr.jit
def gradient_kernel(G,Y,SRC,SCALE,GG,GT,DS,GS,ROWS:tl.constexpr,M:tl.constexpr,
                    D:tl.constexpr=768,BM:tl.constexpr=64,BD:tl.constexpr=128):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    source=tl.load(SRC+rows,rows<M,other=ROWS)
    valid=(rows<M)&(source<ROWS)
    scale=tl.load(SCALE+source,valid,other=0.).to(tl.bfloat16).to(tl.float32)
    inverse=1./tl.load(GS)
    ds=tl.zeros((BM,),tl.float32)
    for d0 in tl.static_range(0,D,BD):
        cols=d0+tl.arange(0,BD)
        grad=tl.load(G+source[:,None].to(tl.int64)*D+cols[None,:],valid[:,None],other=0.).to(tl.float32)
        y=tl.load(Y+rows[:,None]*D+cols[None,:],valid[:,None],other=0.).to(tl.float32)
        ds+=tl.sum(grad*y,1)
        q=tl.minimum(tl.maximum(grad*scale[:,None]*inverse,-57344.),57344.).to(tl.float8e5)
        # Retain the original row-major store to anchor the reduction layout.
        tl.store(GG+rows[:,None]*D+cols[None,:],q,rows[:,None]<M)
        GT.store([d0,tl.program_id(0)*BM],tl.trans(q))
    tl.store(DS+rows,ds,rows<M)


def pack(rows,primary,dual,src,scale):
    m=src.numel()
    assert m%128==0 and primary.shape==(m,768)
    a=TensorDescriptor.from_tensor(primary,[128,32])
    b=TensorDescriptor.from_tensor(dual,[32,128])
    input_kernel[(m//128,24)](rows,a,b,src,scale,m,rows.shape[0],num_warps=4)


def gather(grad,y,src,scale,primary,dual,dgate,gs,rows,m):
    assert m%64==0 and primary.shape==(m,768)
    b=TensorDescriptor.from_tensor(dual,[128,64])
    gradient_kernel[(m//64,)](grad,y,src,scale,primary,b,dgate,gs,rows,m,num_warps=4,enable_fp_fusion=False)
