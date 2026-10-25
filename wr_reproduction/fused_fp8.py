"""MI300 pointer-tiled FP8 GEMMs with original record fused epilogues.

Original FN/E5 byte encodings remain in caches. In registers, the same finite
nonzero byte interpreted as FNUZ represents exactly half its original value.
Negative zero is canonicalized; multiplying the dot by four restores values.
No full-tensor format conversion or additional quantization pass is performed.
"""
import torch
import triton as tr
import triton.language as tl
from wr_reproduction.packed_fp8_decode import native_packed,select_packed

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

@tr.jit
def _gemm(A,B,C,CT,PRE,POST,AUX,SA,SB,PS,DS,AMAX,TILES,
          M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          A0:tl.constexpr,A1:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,
          AE5:tl.constexpr,BE5:tl.constexpr,MODE:tl.constexpr,
          GROUPED:tl.constexpr,EMIT_T:tl.constexpr,STORE_PRE:tl.constexpr,STORE_POST:tl.constexpr,
          BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=64,
          PARTIAL_AMAX:tl.constexpr=False,PREPARED_B:tl.constexpr=False,PREPARED_A:tl.constexpr=False,
          PACKED_DECODE:tl.constexpr=False):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    c=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    expert=0
    if GROUPED:expert=tl.load(TILES+tl.program_id(0)*BM//128)
    acc=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(K,BK)):
        k=base*BK+kk
        ab=tl.load(A+r[:,None]*A0+k[None,:]*A1,(r[:,None]<M)&(k[None,:]<K),other=0)
        bb=tl.load(B+k[:,None]*B0+(c[None,:]+expert*N)*B1,(k[:,None]<K)&(c[None,:]<N),other=0)
        # gfx942 mixed-format MFMA lowering in this Triton build reverses
        # the operand format selectors. Swap type tags, not stored bits.
        # Independent four-format numerical tests guard this workaround.
        if PREPARED_B:
            # The fused weight producer already canonicalized native bits.
            # Retain the mixed-format tag swap used by this gfx942 compiler.
            if AE5:b_native=bb.to(tl.float8e5b16,bitcast=True)
            else:b_native=bb.to(tl.float8e4b8,bitcast=True)
        elif PACKED_DECODE:b_native=native_packed(bb,AE5,BE5)
        else:b_native=_native(bb,AE5,BE5)
        if PREPARED_A:
            if BE5:a_native=ab.to(tl.float8e5b16,bitcast=True)
            else:a_native=ab.to(tl.float8e4b8,bitcast=True)
        elif PACKED_DECODE:a_native=native_packed(ab,BE5,AE5)
        else:a_native=_native(ab,BE5,AE5)
        acc=tl.dot(a_native,b_native,acc)
    acc=acc*(4.*tl.load(SA+expert)*tl.load(SB))
    valid=(r[:,None]<M)&(c[None,:]<N)
    if MODE==1:
        pre=acc.to(tl.bfloat16)
        if STORE_PRE:tl.store(PRE+r[:,None]*N+c[None,:],pre,valid)
        pos=tl.maximum(pre.to(tl.float32),0.)
        values=(pos*pos).to(tl.bfloat16).to(tl.float32)
        if STORE_POST:tl.store(POST+r[:,None]*N+c[None,:],values,valid)
        out=tl.minimum(values*(1./tl.load(PS)),448.).to(tl.float8e4nv)
    elif MODE==2:
        aux=tl.load(AUX+r[:,None]*N+c[None,:],valid,other=0.)
        values=(2.*acc*tl.sqrt(aux.to(tl.float32)*tl.load(PS))).to(tl.bfloat16).to(tl.float32)
        out=tl.maximum(tl.minimum(values*(1./tl.load(DS)),57344.),-57344.).to(tl.float8e5)
    else:
        out=acc
    tl.store(C+r[:,None]*N+c[None,:],out,valid)
    if EMIT_T:tl.store(CT+c[None,:]*M+r[:,None],out,valid)
    if MODE!=0:
        largest=tl.max(tl.max(tl.where(valid,tl.abs(values),0.),1),0)
        if PARTIAL_AMAX:
            tl.store(AMAX+tl.program_id(0)*tl.cdiv(N,BN)+tl.program_id(1),largest)
        else:
            tl.atomic_max(AMAX,largest)


@tr.jit
def _reduce_partial_amax(PARTIAL,AMAX,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    indices=tl.arange(0,BLOCK)
    values=tl.load(PARTIAL+indices,indices<COUNT,other=0.)
    # Tile maxima are abs-values, so their sign bits are clear. This Triton
    # build lowers positive float atomic_max to signed integer atomic max.
    # Retain its exact bit ordering, including positive NaNs and infinities.
    largest=tl.max(values.to(tl.int32,bitcast=True),0).to(tl.float32,bitcast=True)
    tl.atomic_max(AMAX,largest)

_constant_scales={}
TILE_MODE='legacy'  # Only an explicitly configured KS3 research run enables tuning.
HEAD_MODE='legacy'  # Research-only packed-decoder head pipeline.
SHARED_PIPELINE_MODE='legacy'  # Research-only attention/dense producer pipelines.
DECODE_MODE='scalar'  # Research-only register decode; stored record bytes stay unchanged.
AMAX_MODE='atomic'  # Research-only opt-in; preserve the original control path.

def run(a,b,sa,sb,*,out_dtype=torch.float32,mode=0,post_scale=None,dpre_scale=None,
        amax=None,aux=None,emit_t=False,store_pre=False,store_post=False,tiles=None,n=None):
    if a.dtype not in (torch.float8_e4m3fn,torch.float8_e5m2) or b.dtype not in (torch.float8_e4m3fn,torch.float8_e5m2):
        raise TypeError('Record FP8 operands required')
    m,k=a.shape;n=b.shape[1] if n is None else n
    if b.shape[0]!=k:raise ValueError('Inconsistent GEMM shape')
    def scalar(s):
        if isinstance(s,torch.Tensor):return s
        key=(a.device,float(s))
        if key not in _constant_scales:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm up scalar constants before capture')
            _constant_scales[key]=torch.tensor(s,device=a.device,dtype=torch.float32)
        return _constant_scales[key]
    sa,sb=scalar(sa),scalar(sb)
    dtype=out_dtype if not mode else (torch.float8_e4m3fn if mode==1 else torch.float8_e5m2)
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    trans=torch.empty((n,m),device=a.device,dtype=dtype) if emit_t else out
    pre=torch.empty((m,n),device=a.device,dtype=torch.bfloat16) if store_pre else out
    post=torch.empty_like(pre) if store_post else out
    dummy=sa
    if m and n:
        bm,bn,bk=32,64,64
        options=dict(num_warps=4,enable_fp_fusion=False,matrix_instr_nonkdim=16)
        if TILE_MODE!='legacy':
            if TILE_MODE not in ('mi300-v1','mi300-v2'):raise ValueError('Unknown shared FP8 tile policy')
            from wr_reproduction.shared_fp8_tiles import select_tiles,select_tiles_v2
            selector=select_tiles_v2 if TILE_MODE=='mi300-v2' else select_tiles
            selected=selector(m,n,k,a.stride(),b.stride(),tiles is not None)
            if selected is not None:
                bm,bn,bk,warps,stages,mfma=selected
                options.update(num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
        if AMAX_MODE not in ('atomic','partial'):raise ValueError('Unknown shared FP8 amax policy')
        if DECODE_MODE not in ('scalar','packed4','packed4-selected'):raise ValueError('Unknown shared FP8 decode policy')
        packed_decode=tiles is None and (DECODE_MODE=='packed4' or
            (DECODE_MODE=='packed4-selected' and select_packed(m,n,k,a.stride(),b.stride())))
        if SHARED_PIPELINE_MODE not in ('legacy','packed-v1'):raise ValueError('Unknown shared FP8 pipeline policy')
        if SHARED_PIPELINE_MODE=='packed-v1' and packed_decode:
            from wr_reproduction.shared_fp8_tiles import select_packed_shared_tiles
            selected=select_packed_shared_tiles(m,n,k,a.stride(),b.stride(),tiles is not None)
            if selected is not None:
                bm,bn,bk,warps,stages,mfma=selected
                options.update(num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
        if HEAD_MODE not in ('legacy','packed-v1'):raise ValueError('Unknown shared FP8 head policy')
        if HEAD_MODE=='packed-v1' and packed_decode:
            from wr_reproduction.shared_fp8_tiles import select_packed_head_tiles
            selected=select_packed_head_tiles(m,n,k,a.stride(),b.stride(),tiles is not None)
            if selected is not None:
                bm,bn,bk,warps,stages,mfma=selected
                options.update(num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
        partial_mode=AMAX_MODE=='partial' and mode!=0 and tiles is None and amax is not None
        grid=(tr.cdiv(m,bm),tr.cdiv(n,bn));partial_count=grid[0]*grid[1]
        partial=torch.empty((partial_count,),device=a.device,dtype=torch.float32) if partial_mode else None
        _gemm[grid](a.view(torch.uint8),b.view(torch.uint8),out,trans,pre,post,
            aux if aux is not None else dummy,sa,sb,post_scale if post_scale is not None else dummy,
            dpre_scale if dpre_scale is not None else dummy,partial if partial_mode else (amax if amax is not None else dummy),
            tiles if tiles is not None else dummy,m,n,k,*a.stride(),*b.stride(),
            a.dtype==torch.float8_e5m2,b.dtype==torch.float8_e5m2,mode,tiles is not None,
            emit_t,store_pre,store_post,BM=bm,BN=bn,BK=bk,PARTIAL_AMAX=partial_mode,
            PACKED_DECODE=packed_decode,**options)
        if partial_mode:
            _reduce_partial_amax[(1,)](partial,amax,partial_count,tr.next_power_of_2(partial_count),num_warps=4)
    return out,trans if emit_t else None,pre if store_pre else None,post if store_post else None

@tr.jit
def _dw(A,B,C,OFFS,SA,SB,FLAG,H:tl.constexpr,N:tl.constexpr,
        A0:tl.constexpr,A1:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,
        AE5:tl.constexpr,BE5:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=64,PREPARED_A:tl.constexpr=False,PREPARED_B:tl.constexpr=False):
    e=tl.program_id(2)
    lo=0
    if e>0:lo=tl.load(OFFS+e-1)
    hi=tl.load(OFFS+e)
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    c=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(hi-lo,BK)):
        k=lo+base*BK+kk
        av=tl.load(A+r[:,None]*A0+k[None,:]*A1,(r[:,None]<H)&(k[None,:]<hi),other=0)
        bv=tl.load(B+c[None,:]*B0+k[:,None]*B1,(c[None,:]<N)&(k[:,None]<hi),other=0)
        if PREPARED_A:
            if BE5:a_native=av.to(tl.float8e5b16,bitcast=True)
            else:a_native=av.to(tl.float8e4b8,bitcast=True)
        else:a_native=_native(av,BE5,AE5)
        if PREPARED_B:
            if AE5:b_native=bv.to(tl.float8e5b16,bitcast=True)
            else:b_native=bv.to(tl.float8e4b8,bitcast=True)
        else:b_native=_native(bv,AE5,BE5)
        acc=tl.dot(a_native,b_native,acc)
    acc=acc*(4.*tl.load(SA)*tl.load(SB))
    valid=(r[:,None]<H)&(c[None,:]<N)
    dest=C+(e*H+r[:,None])*N+c[None,:]
    if tl.load(FLAG):acc+=tl.load(dest,valid,other=0.).to(tl.float32)
    tl.store(dest,acc,valid)


def grouped_dw(a,b,offs,sa,sb,experts,out,acc_flag):
    h,n=a.shape[0],b.shape[0]
    _dw[(tr.cdiv(h,32),tr.cdiv(n,64),experts)](a.view(torch.uint8),b.view(torch.uint8),out,
        offs,sa,sb,acc_flag,h,n,*a.stride(),*b.stride(),a.dtype==torch.float8_e5m2,
        b.dtype==torch.float8_e5m2,num_warps=4,enable_fp_fusion=False,matrix_instr_nonkdim=16)

_OUTPUT_DTYPES=(torch.float32,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e5m2)

@torch.library.custom_op('wr_mi300::scaled_mm',mutates_args=())
def scaled_mm_op(a:torch.Tensor,b:torch.Tensor,sa:torch.Tensor,sb:torch.Tensor,otype:int,fast:bool)->torch.Tensor:
    return run(a,b,sa,sb,out_dtype=_OUTPUT_DTYPES[otype])[0]

@scaled_mm_op.register_fake
def _(a,b,sa,sb,otype,fast):
    return a.new_empty((a.shape[0],b.shape[1]),dtype=_OUTPUT_DTYPES[otype])

@torch.library.custom_op('wr_mi300::lrs',mutates_args={'amax'})
def lrs_op(a:torch.Tensor,b:torch.Tensor,scale:torch.Tensor,post_scale:torch.Tensor,dpre_scale:torch.Tensor,
           amax:torch.Tensor,aux:torch.Tensor,forward:bool,emit_t:bool,store_pre:bool,store_post:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    q,qt,pre,post=run(a,b,scale,1.,mode=1 if forward else 2,post_scale=post_scale,dpre_scale=dpre_scale,
        amax=amax,aux=aux,emit_t=emit_t,store_pre=store_pre,store_post=store_post)
    return q,qt if qt is not None else a.new_empty((0,)),pre if pre is not None else a.new_empty((0,),dtype=torch.bfloat16),post if post is not None else a.new_empty((0,),dtype=torch.bfloat16)

@lrs_op.register_fake
def _(a,b,scale,post_scale,dpre_scale,amax,aux,forward,emit_t,store_pre,store_post):
    m,n=a.shape[0],b.shape[1];dt=torch.float8_e4m3fn if forward else torch.float8_e5m2
    return (a.new_empty((m,n),dtype=dt),a.new_empty((n,m) if emit_t else (0,),dtype=dt),
        a.new_empty((m,n) if store_pre else (0,),dtype=torch.bfloat16),
        a.new_empty((m,n) if store_post else (0,),dtype=torch.bfloat16))
