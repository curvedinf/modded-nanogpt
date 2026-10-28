"""Lossless GS32 code/scale transport; all expert products remain BF16."""
import torch,triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale

@tr.jit
def encode(BANK,Q,S,LO:tl.constexpr,BR:tl.constexpr=32,N:tl.constexpr=1024,D:tl.constexpr=768):
    slot=LO+tl.program_id(1);down=(slot%16)>=8
    rows=tl.program_id(0)*BR+tl.arange(0,BR);cols=tl.program_id(2)*32+tl.arange(0,32)
    R=tl.where(down,D,N);C=tl.where(down,N,D)
    mask=(rows[:,None]<R)&(cols[None,:]<C)
    offset=tl.where(down,cols[None,:]*D+rows[:,None],rows[:,None]*D+cols[None,:])
    a=tl.load(BANK+slot*N*D+offset,mask,other=0.).to(tl.float32)
    scale=_half_scale(tl.max(tl.abs(a),1))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(a/scale[:,None]),-127.),127.)
    tl.store(Q+slot*N*D+rows[:,None]*C+cols[None,:],q,mask)
    tl.store(S+slot*N*D//32+rows*(C//32)+tl.program_id(2),scale,(rows<R)&(tl.program_id(2)<C//32))

@tr.jit
def decode(Q,S,UP,DOWN,N:tl.constexpr=1024,D:tl.constexpr=768,B:tl.constexpr=1024):
    slot=tl.program_id(1);down=(slot%16)>=8
    i=tl.program_id(0)*B+tl.arange(0,B)
    q=tl.load(Q+slot*N*D+i,i<N*D,other=0).to(tl.float32)
    scale=tl.load(S+slot*N*D//32+i//32,i<N*D,other=1).to(tl.float32)
    expert=(slot//16)*8+slot%8
    if down:tl.store(DOWN+expert*N*D+i,q*scale,i<N*D)
    else:tl.store(UP+expert*N*D+i,q*scale,i<N*D)

def attach(cls,br=32):
    import torch.distributed as dist
    from track_1_short.perf.cuda_graphs import fp8_refresh_graphs as refresh
    class Packed(cls):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            n,d=self.ks3_up.shape[-2:]
            self.register_buffer('ks3_transport_q',torch.empty(80,n*d,device=self.device,dtype=torch.int8),persistent=False)
            self.register_buffer('ks3_transport_s',torch.empty(80,n*d//32,device=self.device,dtype=torch.float16),persistent=False)
        @torch.no_grad()
        def _quantize_moe(self,lo,hi,exact):
            n,d=self.ks3_up.shape[-2:]
            encode[(tr.cdiv(max(n,d),br),hi-lo,tr.cdiv(max(n,d),32))](self.moe_bank,self.ks3_transport_q,self.ks3_transport_s,lo,BR=br,N=n,D=d,num_warps=4)
            if lo==0 and hi==80:self.build_moe_cols()
        def gather_moe_f8(self):
            lo,hi=self._moe_lo,self._moe_hi;q,s=self.ks3_transport_q,self.ks3_transport_s
            return [('moe_bank',dist.all_gather_into_tensor(q,q[lo:hi],async_op=True).get_future()),('moe_scales',dist.all_gather_into_tensor(s,s[lo:hi],async_op=True).get_future())]
        @torch.no_grad()
        def build_moe_cols(self):
            n,d=self.ks3_up.shape[-2:]
            decode[(tr.cdiv(n*d,1024),80)](self.ks3_transport_q,self.ks3_transport_s,self.ks3_up,self.ks3_down,N=n,D=d,num_warps=4)
    refresh.MOE_LOCAL_OUTPUTS=('ks3_transport_q','ks3_transport_s')
    refresh.MOE_COLS_OUTPUTS=('ks3_up','ks3_down')
    return Packed
