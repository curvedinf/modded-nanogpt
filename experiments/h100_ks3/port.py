"""Distributed H100 transport for the verified MI100 GS32/BF16 KS3 recipe.

The existing KS3 math/autograd and optimizer are reused. Each ANVIL owner
publishes its GS32-rounded BF16 matrices, then gathers them before readers.
Shared attention, dense MLPs, head, data and schedule remain upstream PR393.
"""
import os
import torch
import torch.distributed as dist
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale

@tr.jit
def _publish(BANK, PACKED, LO:tl.constexpr, N:tl.constexpr=1024,D:tl.constexpr=768):
    slot=LO+tl.program_id(1)
    down=(slot%16)>=8
    rows=tl.program_id(0)*8+tl.arange(0,8)
    cols=tl.program_id(2)*32+tl.arange(0,32)
    R=tl.where(down,D,N);C=tl.where(down,N,D)
    mask=(rows[:,None]<R)&(cols[None,:]<C)
    offset=tl.where(down,cols[None,:]*D+rows[:,None],rows[:,None]*D+cols[None,:])
    a=tl.load(BANK+slot*N*D+offset,mask,other=0.).to(tl.float32)
    scale=_half_scale(tl.max(tl.abs(a),1))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(a/scale[:,None]),-127.),127.)*scale[:,None]
    tl.store(PACKED+slot*N*D+rows[:,None]*C+cols[None,:],q,mask)

def publish(bank,packed,lo,hi):
    n,d=bank.shape[-2:]
    _publish[(tr.cdiv(max(n,d),8),hi-lo,tr.cdiv(max(n,d),32))](bank,packed,lo,N=n,D=d,num_warps=4)

def install(optimization=None):
    from wr_reproduction import ks3_moe as ks,ks3_bf16_gemm as gemm
    if os.environ.get('KS3_H100_SKIP_ATTENTION') is not None:
        from experiments.h100_ks3.attention_ablation import install as skip_attention
        skip_attention(os.environ['KS3_H100_SKIP_ATTENTION'])
    if os.environ.get('KS3_H100_GEOMETRY_TAIL','0') == '1':
        from experiments.h100_ks3.geometry_tail import install as install_geometry_tail
        install_geometry_tail()
    if os.environ.get('KS3_H100_FUSED_ANVIL','0') == '1':
        from experiments.h100_ks3.moe_optimizer import install as install_fused_optimizer
        install_fused_optimizer()
    # Retain the selected tile shapes; omit AMD-only compiler arguments.
    from wr_reproduction.bf16_tiles import select_projection_tiles
    def mm(a,w,tiles,dtype=torch.float32):
        m,k=a.shape;_,_,n=w.shape
        bm,bn,bk,warps,stages,_=select_projection_tiles(m,n,k,w.stride(2)==1)
        out=torch.empty((m,n),device=a.device,dtype=dtype)
        ks._mm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a,w,out,tiles,m,n,k,*w.stride(),BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
        return out
    def dw(a,b,out,offs):
        e,h,d=out.shape
        ks._dw[(tr.cdiv(h,128),tr.cdiv(d,128),e)](a,b,out,offs,h,d,False,BM=128,BN=128,BK=64,num_warps=8,num_stages=1)
    gemm.run=mm;gemm.dw=dw
    width=int(os.environ.get('KS3_H100_WIDTH','1024'))
    rank=int(os.environ.get('KS3_H100_RANK','1'))
    if rank not in (1,2,4) or width%rank:raise ValueError('Invalid H100 KS3 rank/width')
    geometry_lr=float(os.environ.get('KS3_H100_GEOMETRY_LR','.125'))
    projection_lr=float(os.environ.get('KS3_H100_PROJECTION_LR','1.'))
    base=ks.install('bf16',bf16_kernel_mode='tuned',charts=width//rank,subspace_rank=rank,projection_lr_multiplier=projection_lr,geometry_config={'lr_mul':geometry_lr,'betas':[.9,.99],'raw_scale':.75,'raw_shift':0.,'coefficients':[0.]*6})
    class H100KS3(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.register_buffer('ks3_packed_bank',torch.empty((80,width,768),device=self.device,dtype=torch.bfloat16),persistent=False)
        @torch.no_grad()
        def _quantize_moe(self,lo,hi,exact):
            publish(self.moe_bank,self.ks3_packed_bank,lo,hi)
            if lo==0 and hi==80:self.build_moe_cols()
        def gather_moe_f8(self):
            p=self.ks3_packed_bank
            return [('moe_bank',dist.all_gather_into_tensor(p,p[self._moe_lo:self._moe_hi],async_op=True).get_future())]
        @torch.no_grad()
        def build_moe_cols(self):
            bank=self.ks3_packed_bank.view(5,2,8,width*768)
            self.ks3_up.copy_(bank[:,0].view(5,8,width,768))
            self.ks3_down.copy_(bank[:,1].view(5,8,768,width))
    # Graph capture must snapshot/check the actual KS3 producer outputs.
    from track_1_short.perf.cuda_graphs import fp8_refresh_graphs as refresh
    refresh.MOE_LOCAL_OUTPUTS=('ks3_packed_bank',)
    refresh.MOE_COLS_OUTPUTS=('ks3_up','ks3_down')
    selected = optimization if optimization is not None else os.environ.get('KS3_H100_OPT', '')
    if selected:
        if selected=='fp8-native':
            if rank!=1 and os.environ.get('KS3_FP8_FUSE_GEOMETRY','0')=='1':
                raise ValueError('Fused FP8 up projection currently requires rank1')
            from experiments.h100_ks3.fp8 import attach
            return attach(H100KS3)
        from experiments.h100_ks3.optimized import install as install_optimized
        install_optimized(selected)
        if os.environ.get('KS3_H100_TRANSPORT', 'bf16') == 'codes':
            from experiments.h100_ks3.transport import attach
            return attach(H100KS3, br=int(os.environ.get('KS3_H100_PACK_ROWS', '32')))
    return H100KS3
