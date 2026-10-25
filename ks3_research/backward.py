"""FP32 BLAS alternatives for KS3's VJP matrix products.

Backward orchestration adapted from curvedinf/k-splanifolds ks3_int8.py,
ea7a9c1121efe711345bd9e6cf22e69401c0c7b4, CC-BY-SA-4.0; attributed source and
license remain in vendor/. Forward and fused geometry VJP use that unchanged
source. The optional all-products path also separates the first feature-gradient
GEMM from the unchanged surrogate in geometry_backward.py.
"""
import torch
import torch.distributed as dist
import triton as tr
from torch.autograd.function import once_differentiable
from ks3_research.vendor import ks3_int8 as original


def unpack_fp32(codes,scales):
    return codes.float()*scales.float().repeat_interleave(32,-1)[...,:codes.shape[-1]]


def unpack_bf16(codes,scales):
    from ks3_research.unpack import unpack_gs32
    return unpack_gs32(codes,scales,dtype=torch.bfloat16)


def matrix_product(a,b,bf16=False,fp8=False):
    if fp8:
        if bf16:raise ValueError('Select FP8 or BF16 products, not both')
        from ks3_research.fp8_blas import matrix_product as fp8_mm
        return fp8_mm(a,b)
    if bf16:
        with torch.autocast('cuda',enabled=False):
            return torch.mm(a.to(torch.bfloat16),b.to(torch.bfloat16),out_dtype=torch.float32)
    return a@b


def project_features_blas(qx,sx,qd,sd,p,ps,scale,shift,lut,rank,nonlinear,unpack=unpack_fp32,dynamic_rows=False,bf16_products=False,fp8_products=False,prepared=None):
    """FP32 input product followed by the original FP16/GS32 geometry kernel.

    A different product summation can cross a later quantization boundary; this
    is an explicitly measured numerical variant, not a bitwise replacement.
    """
    with torch.autocast('cuda',enabled=False):
        if prepared is not None:
            from ks3_research.fp8_blas import product
            coords=product(prepared[0],prepared[1],transpose_b=True)
        else:
            coords=matrix_product(unpack(qx,sx),unpack(qd,sd).T,bf16_products,fp8_products)
    m,n=coords.shape
    clipped=coords.clamp(-65504.,65504.)
    mean=clipped if rank==1 else clipped.reshape(m,n//rank,rank).mean(-1)
    h=torch.empty((m,n),device=coords.device,dtype=torch.int8)
    hs=torch.empty((m,tr.cdiv(n,32)),device=coords.device,dtype=original.SCALE_DTYPE)
    geometry_quant = original._geometry_quant
    if dynamic_rows:
        from control393.ks3_dynamic import _geometry_quant
        geometry_quant = _geometry_quant
    geometry_quant[(tr.cdiv(m,16),tr.cdiv(n,32))](
        coords,mean,p,ps,scale,shift,lut,h,hs,m,n,rank,nonlinear,BM=16,BN=32,num_warps=4)
    # Rank-one backward obtains the clipped mean from coords inside its kernel.
    # Keep the forward clipping temporary only through geometry_quant, rather
    # than retaining another M*N FP32 tensor for every FFN until backward.
    return h,hs,coords,coords if rank==1 else mean


class _BLASKS3Train(torch.autograd.Function):
    @staticmethod
    def forward(ctx,*args):
        return original._KS3Train.forward(ctx,*args)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        rank=ctx.subspace_rank
        bf16_products=getattr(ctx,'bf16_products',False)
        fp8_products=getattr(ctx,'fp8_products',False)
        unpack=unpack_fp32
        if getattr(ctx,'fused_unpack',False):
            from ks3_research.unpack import unpack_gs32
            unpack=unpack_gs32
        if bf16_products:
            unpack=unpack_bf16
        if ctx.empty:
            dx,ddown,dup,dcoeff,dscale,dshift=[torch.zeros_like(t) for t in ctx.saved_tensors]
        else:
            (qx,sx,qd,sd,qu,su,h,hs,coords,p,ps,scale,shift,
             raw_scale,raw_shift,mean)=ctx.saved_tensors
            grad=grad.reshape(-1,qu.shape[0]).contiguous()
            m,local=coords.shape
            all_products=getattr(ctx,'all_products',False)
            if all_products:
                grad_product=grad.to(torch.bfloat16 if bf16_products else torch.float32)
                if fp8_products:
                    from ks3_research.fp8_blas import Operand,product
                    fx,fd,fu,fh=ctx.fp8_operands
                    fg=Operand(grad)
                    dc=product(fg,fu)
                else:
                    dc=matrix_product(grad_product,unpack(qu,su),bf16_products)
            else:
                dc=torch.empty_like(coords)
            dmean=torch.empty_like(coords) if rank>1 else dc
            row_tile=64 if all_products else 16
            partial=torch.empty((tr.cdiv(m,row_tile),local,8),device=grad.device,dtype=torch.float32)
            if all_products:
                from ks3_research.geometry_backward import geometry_vjp
                if getattr(ctx,'dynamic_rows',False):
                    from control393.ks3_dynamic import geometry_vjp
                geometry_vjp[(tr.cdiv(m,row_tile),tr.cdiv(local,32))](
                    dc,coords,p,ps,scale,shift,raw_scale,raw_shift,partial,mean,dmean,
                    rank,m,local,BM=row_tile,BN=32,num_warps=4)
            else:
                original._backward_mm[(tr.cdiv(m,16),tr.cdiv(local,32))](
                    grad,qu,su,dc,coords,p,ps,scale,shift,raw_scale,raw_shift,partial,mean,dmean,rank,
                    m,local,qu.shape[0],*grad.stride(),32,True,BM=16,BN=32,BK=32,num_warps=4)
            dcoeff=torch.empty((local,6),device=grad.device,dtype=torch.float32)
            dscale=torch.empty((local,),device=grad.device,dtype=torch.float32)
            dshift=torch.empty_like(dscale)
            if rank>1:
                shared=dmean.reshape(m,local//rank,rank).mean(-1)
                original._finish_coordinate_gradient[(tr.cdiv(m*local,256),)](
                    coords,dc,shared,m,local,rank,BLOCK=256)
            # The optional BF16 operand variant retains FP32 GEMM outputs and masters.
            # Separate GEMMs let BLAS tile long weight-gradient reductions instead
            # of sharing one persistent launch with the input gradient.
            if fp8_products:
                fc=Operand(dc)
                dx=product(fc,fd)
                ddown=product(fc,fx,transpose_a=True)
                dup=product(fg,fh,transpose_a=True)
            else:
                dx=matrix_product(dc,unpack(qd,sd),bf16_products)
                ddown=matrix_product(dc.T,unpack(qx,sx),bf16_products)
                dup=matrix_product((grad_product if all_products else grad.float()).T,unpack(h,hs),bf16_products)
            reduce_geometry = original._reduce_geometry
            if getattr(ctx,'dynamic_rows',False):
                from control393.ks3_dynamic import _reduce_geometry
                reduce_geometry = _reduce_geometry
            reduce_geometry[(local,)](partial,dcoeff,dscale,dshift,
                local,partial.shape[0],tr.next_power_of_2(partial.shape[0]),num_warps=4)
            if rank>1:
                dscale=dscale.reshape(-1,rank).sum(-1)
                dshift=dshift.reshape(-1,rank).sum(-1)
            dcoeff=dcoeff.reshape(ctx.coefficient_shape)
            dx=dx.reshape(ctx.input_shape)
        if ctx.world>1 and dx.numel():
            if ctx.reducer is None:
                dist.all_reduce(dx)
            else:
                dx=ctx.reducer(dx)
        return dx.to(ctx.input_dtype),ddown,dup,dcoeff,dscale,dshift,None,None,None,None,None


class _AllBLASKS3Train(_BLASKS3Train):
    @staticmethod
    def forward(ctx,*args):
        ctx.all_products=True
        return original._KS3Train.forward(ctx,*args)


class _OutputBLASKS3Train(_BLASKS3Train):
    """Same GS32 operands/input geometry; BLAS sums the output projection.

    Forward orchestration follows the attributed original._KS3Train.forward.
    Unlike the backward-only variants, this can change final rounding: unpacked
    FP32 products replace per-group INT32 dots followed by FP32 scale products.
    """
    @staticmethod
    def forward(ctx,inputs,down,up,coefficients,raw_scale,raw_shift,
                lut,nonlinear,world,reducer,subspace_rank):
        ctx.all_products=True
        tensors=(down,up,coefficients,raw_scale,raw_shift)
        if any(t.device!=inputs.device or t.dtype!=torch.float32 for t in tensors):
            raise ValueError('KS3 training requires FP32 master parameters on the input device.')
        if inputs.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError('KS3 inputs must be FP32, FP16 or BF16.')
        ctx.input_dtype=inputs.dtype
        output_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else inputs.dtype
        if not inputs.is_cuda or inputs.shape[-1]!=down.shape[1]:
            raise ValueError('KS3 training requires GPU inputs with the configured feature width.')
        x=inputs.reshape(-1,inputs.shape[-1]).contiguous()
        raw_scale,raw_shift=raw_scale.contiguous(),raw_shift.contiguous()
        m,local=x.shape[0],down.shape[0]
        ctx.input_shape,ctx.world,ctx.reducer=inputs.shape,world,reducer
        ctx.subspace_rank,ctx.coefficient_shape=subspace_rank,coefficients.shape
        ctx.empty=not m or not local
        if ctx.empty:
            ctx.save_for_backward(inputs,down,up,coefficients,raw_scale,raw_shift)
            y=torch.zeros((m,up.shape[0]),device=x.device,dtype=torch.float32)
        else:
            pack_training_tensors = original._pack_training_tensors
            if getattr(ctx,'dynamic_rows',False):
                from control393.ks3_dynamic import _pack_training_tensors
                pack_training_tensors = _pack_training_tensors
            qx,sx,qd,sd,qu,su,p,ps,scale,shift=pack_training_tensors(
                x,down,up,coefficients,raw_scale,raw_shift,subspace_rank)
            unpack=unpack_fp32
            if getattr(ctx,'fused_unpack',False):
                from ks3_research.unpack import unpack_gs32
                unpack=unpack_gs32
            if getattr(ctx,'bf16_products',False):
                unpack=unpack_bf16
            prepared=None
            if getattr(ctx,'fp8_products',False):
                from ks3_research.fp8_blas import Operand,product
                fx,fd,fu=Operand(qx,sx),Operand(qd,sd),Operand(qu,su)
                prepared=(fx,fd)
            if getattr(ctx,'input_blas',False):
                h,hs,coords,mean=project_features_blas(qx,sx,qd,sd,p,ps,scale,shift,lut,
                                                      subspace_rank,nonlinear,unpack,
                                                      dynamic_rows=getattr(ctx,'dynamic_rows',False),
                                                      bf16_products=getattr(ctx,'bf16_products',False),
                                                      fp8_products=getattr(ctx,'fp8_products',False),prepared=prepared)
            else:
                h,hs,coords,mean=original._project_features(qx,sx,qd,sd,p,ps,scale,shift,lut,
                                                          subspace_rank,nonlinear,save=True)
            with torch.autocast('cuda',enabled=False):
                if getattr(ctx,'fp8_products',False):
                    fh=Operand(h,hs)
                    y=product(fh,fu,transpose_b=True)
                    ctx.fp8_operands=(fx,fd,fu,fh)
                else:
                    y=matrix_product(unpack(h,hs),unpack(qu,su).T,getattr(ctx,'bf16_products',False))
            ctx.save_for_backward(qx,sx,qd,sd,qu,su,h,hs,coords,p,ps,scale,shift,
                                  raw_scale,raw_shift,mean)
        if world>1 and y.numel():
            if reducer is None:
                dist.all_reduce(y)
            else:
                y=reducer(y)
        return y.reshape(*inputs.shape[:-1],up.shape[0]).to(output_dtype)


class _BothBLASKS3Train(_OutputBLASKS3Train):
    @staticmethod
    def forward(ctx,*args):
        ctx.input_blas=True
        return _OutputBLASKS3Train.forward(ctx,*args)


class _FusedUnpackKS3Train(_BothBLASKS3Train):
    @staticmethod
    def forward(ctx,*args):
        ctx.fused_unpack=True
        return _BothBLASKS3Train.forward(ctx,*args)


class _DynamicRowsKS3Train(_FusedUnpackKS3Train):
    @staticmethod
    def forward(ctx,*args):
        ctx.dynamic_rows=True
        return _FusedUnpackKS3Train.forward(ctx,*args)


class _BF16DynamicRowsKS3Train(_DynamicRowsKS3Train):
    @staticmethod
    def forward(ctx,*args):
        ctx.bf16_products=True
        return _DynamicRowsKS3Train.forward(ctx,*args)


class _FP8DynamicRowsKS3Train(_DynamicRowsKS3Train):
    @staticmethod
    def forward(ctx,*args):
        ctx.fp8_products=True
        return _DynamicRowsKS3Train.forward(ctx,*args)


def ks3_blas(inputs,down,up,coefficients,raw_scale,raw_shift,*,subspace_rank=1,
             nonlinear='fp16',tanh_lut=None,all_products=False,output_blas=False,input_blas=False,
             fused_unpack=False,dynamic_rows=False,bf16_products=False,fp8_products=False):
    modes={'fp32':0,'fp16':1,'int8':2}
    if nonlinear not in modes:
        raise ValueError('Unknown KS3 nonlinearity')
    charts=raw_scale.numel()
    if (subspace_rank<1 or raw_scale.ndim!=1 or raw_shift.ndim!=1 or raw_shift.numel()!=charts
            or down.shape[0]!=charts*subspace_rank or down.ndim!=2 or up.ndim!=2
            or up.shape[1]!=down.shape[0] or coefficients.numel()!=6*down.shape[0]):
        raise ValueError('Inconsistent KS3 geometry or projection shapes')
    if torch.get_float32_matmul_precision()!='highest':
        raise ValueError('The FP32 BLAS ablation requires existing highest matmul precision')
    if tanh_lut is None:
        tanh_lut=(torch.tanh(torch.arange(-128,128,device=inputs.device).float()*(3.2/127.))*127).round().clamp(-127,127).to(torch.int8)
    implementation=_AllBLASKS3Train if all_products else _BLASKS3Train
    if output_blas:
        if not all_products:
            raise ValueError('The BLAS-output adapter uses the all-products BLAS backward')
        implementation=_OutputBLASKS3Train
    if input_blas:
        if not output_blas:
            raise ValueError('The BLAS-input adapter also requires BLAS-output')
        implementation=_BothBLASKS3Train
    if fused_unpack:
        if not input_blas:
            raise ValueError('Fused unpacking requires both BLAS forward projections')
        implementation=_FusedUnpackKS3Train
    if dynamic_rows:
        if not fused_unpack or subspace_rank != 1:
            raise ValueError('Runtime MoE rows require rank1 fused-unpack KS3')
        implementation=_DynamicRowsKS3Train
    if bf16_products:
        if not dynamic_rows:
            raise ValueError('BF16 operands require the verified runtime-row rank1 path')
        implementation=_BF16DynamicRowsKS3Train
    if fp8_products:
        if not dynamic_rows or bf16_products:raise ValueError('FP8 products require runtime-row rank1 path without BF16 products')
        implementation=_FP8DynamicRowsKS3Train
    return implementation.apply(inputs,down,up,coefficients,raw_scale,raw_shift,tanh_lut,
                                 modes[nonlinear],1,None,subspace_rank)


def ks3_blas_all(*args,**kwargs):
    return ks3_blas(*args,**kwargs,all_products=True)


def ks3_blas_output(*args,**kwargs):
    return ks3_blas(*args,**kwargs,all_products=True,output_blas=True)


def ks3_blas_both(*args,**kwargs):
    return ks3_blas(*args,**kwargs,all_products=True,output_blas=True,input_blas=True)


def ks3_blas_fused_unpack(*args,**kwargs):
    return ks3_blas(*args,**kwargs,all_products=True,output_blas=True,input_blas=True,fused_unpack=True)
