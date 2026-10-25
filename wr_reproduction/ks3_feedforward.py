"""KS3 at every dense and routed feedforward site, with explicit edge hybrids.

The dense KS3 projections retain the original mlp_bank ANVIL label, master
mantissas, alternating2x output LR, sharding/padding, and gradient accumulation.
Hybrid MLPs use a separately padded original-width bank with identical policy.
"""
import math
import torch
import triton as tr
from wr_reproduction import ks3_moe as core
from wr_reproduction.matched_architecture import DENSE,RETAIN
from wr_reproduction.ks3_feedforward_source import forward_tree,refresh_tree,compile_method,method

COVERAGE={'forward':set(),'backward':set()}

@torch.library.custom_op('wr_ks3_dense::forward',mutates_args=())
def dense_forward(rows:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,slot:int,gs32:bool=True)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    COVERAGE['forward'].add(slot)
    m=tr.cdiv(rows.shape[0],128)*128
    src=torch.arange(m,device=rows.device,dtype=torch.int32)
    tiles=torch.zeros(m//128,device=rows.device,dtype=torch.int32)
    offs=torch.full((1,),m,device=rows.device,dtype=torch.int32)
    if gs32:
        y,x,c,h=core.forward(rows,src,tiles,up.unsqueeze(0),down.unsqueeze(0),g.unsqueeze(0))
    else:
        d,n=rows.shape[1],up.shape[0]
        x=torch.empty((m,d),device=rows.device,dtype=torch.bfloat16)
        core._pack[(tr.cdiv(m,128),tr.cdiv(d,32))](rows,x,m,d,*rows.stride(),
            SRC=src,INPUT_ROWS=rows.shape[0],ROWS_PER_TILE=128,GS32=False,num_warps=4)
        c=core.mm(x,up.T.unsqueeze(0),tiles);h=torch.empty_like(c,dtype=torch.bfloat16)
        core._geometry[(tr.cdiv(m,64),tr.cdiv(n,32))](c,g.unsqueeze(0),h,tiles,m,n,
            BM=64,GS32=False,RANK=core.KS3_RANK,num_warps=4)
        y=core.mm(h,down.T.unsqueeze(0),tiles,dtype=rows.dtype)
    return y,x,c,h,src,tiles,offs

@dense_forward.register_fake
def _(rows,up,down,g,slot,gs32=True):
    m=tr.cdiv(rows.shape[0],128)*128;n=up.shape[0];d=rows.shape[1]
    return (rows.new_empty((m,d)),rows.new_empty((m,d)),rows.new_empty((m,n),dtype=torch.float32),
            rows.new_empty((m,n)),rows.new_empty((m,),dtype=torch.int32),
            rows.new_empty((m//128,),dtype=torch.int32),rows.new_empty((1,),dtype=torch.int32))

@torch.library.custom_op('wr_ks3_dense::backward',mutates_args=())
def dense_backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,
                   src:torch.Tensor,tiles:torch.Tensor,offs:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,slot:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    COVERAGE['backward'].add(slot)
    bank=torch.empty((2,up.shape[0],up.shape[1]),device=up.device,dtype=up.dtype)
    gate=torch.ones(grad.shape[0],device=grad.device,dtype=torch.float32)
    dx,_,dg=core.backward(grad.contiguous(),y,x,c,h,gate,src,offs,tiles,
                          up.unsqueeze(0),down.unsqueeze(0),g.unsqueeze(0),bank,0)
    return dx[:grad.shape[0]].contiguous(),bank,dg.squeeze(0)

@dense_backward.register_fake
def _(grad,y,x,c,h,src,tiles,offs,up,down,g,slot):
    return torch.empty_like(grad),up.new_empty((2,*up.shape)),torch.empty_like(g)

class DenseKS3(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rows,w1,w2,up,down,g,slot,gs32=True):
        ctx.argument_count=len(ctx.needs_input_grad)
        y,x,c,h,src,tiles,offs=dense_forward(rows,up,down,g,slot,gs32)
        ctx.save_for_backward(y,x,c,h,src,tiles,offs,up,down,g);ctx.slot=slot
        return y[:rows.shape[0]]
    @staticmethod
    def backward(ctx,grad):
        dx,bank,dg=dense_backward(grad,*ctx.saved_tensors,ctx.slot)
        return (dx,bank[0],bank[1],None,None,dg,None,None)[:ctx.argument_count]

def install(base,recipe):
    if recipe['precision']!='bf16':raise ValueError('Full-feedforward implementation currently verified for BF16 KS3 only')
    dense_precision=recipe.get('dense_precision','bf16')
    if dense_precision not in ('bf16','fp8'):raise ValueError('Unknown dense KS3 precision')
    dense_gs32=recipe.get('dense_training_gs32',True)
    if type(dense_gs32) is not bool:raise ValueError('Dense GS32 selection must be boolean')
    if not dense_gs32 and recipe['rank']!=1:raise ValueError('Direct dense encoding requires rank one')
    if dense_precision=='fp8' and recipe['rank']!=1:raise ValueError('Dense FP8 requires rank one')
    from track_1_short.model import gpt
    from track_1_short import training
    from track_1_short.optim import anvil
    arch=recipe['architecture'];keep=tuple(sorted(DENSE[l] for l in RETAIN[arch]))
    mapping={slot:i for i,slot in enumerate(keep)}
    scale_mapping={slot:i for i,slot in enumerate((*keep,*(s for s in range(5) if s not in keep)))}
    charts,rank=recipe.get('dense_charts',recipe['charts']),recipe['rank'];features=charts*rank
    if features not in (512,768,1024,1280,1536,2816):raise ValueError('Unsupported dense KS3 width')
    geometry=recipe['geometry']
    multiplier=recipe.get('dense_projection_lr_multiplier',recipe.get('projection_lr_multiplier',1.))
    if not math.isfinite(multiplier) or multiplier<=0:
        raise ValueError('Dense KS3 projection LR multiplier must be finite and positive')
    geometry_multiplier=recipe.get('dense_geometry_lr_multiplier',geometry['lr_mul'])
    if not math.isfinite(geometry_multiplier) or geometry_multiplier<0:
        raise ValueError('Dense KS3 geometry LR multiplier must be finite and nonnegative')
    optimizer_policy={}
    geometry_policy={}
    transformed_forward,forward_sha=compile_method(forward_tree(),gpt.__dict__)
    refresh,_=compile_method(refresh_tree(len(keep),dense_precision=='fp8'),gpt.__dict__)
    # Reuse the exact original allocation/layout definitions, only with fewer
    # retained dense slots and remapped fold-lambda source indices.
    namespace=dict(gpt.__dict__,NUM_USED_MLP_SLOTS=len(keep),
                   MLP_SLOT_OF_LAYER={layer:mapping[slot] for layer,slot in DENSE.items() if slot in mapping and layer!=8})
    init_retained,_=compile_method(method('init_mlp_fp8'),namespace)
    class FullKS3GPT(base):
        _matched_forward_sha256=forward_sha
        retained_slots=keep
        forward=transformed_forward
        _quantize_mlp_weights_and_scales=refresh
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            original=self.mlp_bank
            bank=original.detach()[:,:,:features,:].clone()
            for slot in keep:bank[slot].zero_()
            self.mlp_bank=torch.nn.Parameter(bank)
            self.mlp_bank.label='mlp_bank';self.mlp_bank.reshape=(16,features,768)
            self.mlp_bank.frozen_matrices=frozenset(2*s+j for s in (*keep,5,6,7) for j in (0,1))
            if keep:
                retained=original.new_zeros((4,2,2816,768))
                for slot,index in mapping.items():retained[index].copy_(original.detach()[slot])
                self.retained_mlp_bank=torch.nn.Parameter(retained)
                self.retained_mlp_bank.label='retained_mlp_bank';self.retained_mlp_bank.reshape=(8,2816,768)
                self.retained_mlp_bank.frozen_matrices=frozenset(2*s+j for s in range(len(keep),4) for j in (0,1))
            init_retained(self,768)
            self.register_buffer('_mlp_ks3_up',torch.zeros((5,features,768),device=self.device,dtype=torch.bfloat16),persistent=False)
            self.register_buffer('_mlp_ks3_down',torch.zeros((5,768,features),device=self.device,dtype=torch.bfloat16),persistent=False)
            geo=torch.zeros((5,charts,6*rank+2),device=self.device)
            for coordinate in range(rank):geo[...,6*coordinate:6*coordinate+6]=torch.tensor(geometry['coefficients'],device=self.device)
            geo[...,6*rank]=geometry['raw_scale'];geo[...,6*rank+1]=geometry['raw_shift']
            self.ks3_dense_geometry=torch.nn.Parameter(geo);self.ks3_dense_geometry.label='ks3_dense_geometry'
            if dense_precision=='fp8':
                from wr_reproduction.ks3_dense_fp8 import init_buffers
                init_buffers(self,features,training_gs32=dense_gs32)
        def cast_matrix_weights_bf16(self):
            super().cast_matrix_weights_bf16()
            if keep:self.retained_mlp_bank.data=self.retained_mlp_bank.data.bfloat16()
        @torch.no_grad()
        def _refresh_dense_ks3(self):
            for slot in range(5):
                if slot in keep:continue
                up=self.mlp_bank[slot,0];down=self.mlp_bank[slot,1].T
                for source,out in ((up,self._mlp_ks3_up[slot]),(down,self._mlp_ks3_down[slot])):
                    r,c=source.shape
                    core._pack[(tr.cdiv(r,128),tr.cdiv(c,32))](source,out,r,c,*source.stride(),ROWS_PER_TILE=128,
                        GS32=(dense_precision=='fp8' or not self.training or dense_gs32),num_warps=4)
            if dense_precision=='fp8':
                from wr_reproduction.ks3_dense_fp8 import refresh_weights
                refresh_weights(self,self.mlp_quantize_calls<=gpt.FP8_EXACT_SCALE_CALLS)
        def _matched_dense(self,slot,normed,w1,w2,post,use_fp8,retained_all):
            if slot in mapping:
                k=mapping[slot];a,b=retained_all[2*k:2*k+2]
                if use_fp8:
                    x_f8=gpt.quantize_dual_layout_fused(normed.detach().view(-1,normed.size(-1)),self._mlp_x_scale,fmt=torch.float8_e4m3fn)
                    args=self._mlp_fp8_args(k,self._mlp_up_f8[k],x_f8,post)
                else:args=None
                return gpt.GPT._mlp(self,normed,a,b,args)
            rows=normed.reshape(-1,normed.shape[-1])
            if self.training and dense_precision=='fp8':
                from wr_reproduction.ks3_dense_fp8 import apply
                y=apply(self,slot,scale_mapping[slot],rows,w1,w2)
            else:
                y=DenseKS3.apply(rows,w1,w2,self._mlp_ks3_up[slot],self._mlp_ks3_down[slot],self.ks3_dense_geometry[slot],slot,dense_gs32 if self.training else True)
            if post is not None:y=y*post
            return y.reshape(normed.shape)
        def _mlp(self,*a,**kw):
            raise RuntimeError('Unmapped conventional feedforward path in full-KS3 overlay')
        def assert_feedforward_inventory(self,rows):
            assert tuple(self.mlp_bank.shape)==(8,2,features,768)
            assert tuple(self.ks3_dense_geometry.shape)==(5,charts,6*rank+2)
            actual={b['slot'] for r in rows for b in r['feedforward'] if b['type']=='mlp'}
            assert actual==set(keep)
            assert bool(keep)==hasattr(self,'retained_mlp_bank')
        def feedforward_execution_report(self):
            expected=set(range(5))-set(keep)
            for kind in ('forward','backward'):assert COVERAGE[kind]==expected,(kind,COVERAGE,expected)
            assert core.EXECUTED_MOE_BACKWARDS==set(range(5)),core.EXECUTED_MOE_BACKWARDS
            return dict(dense_ks3_forward_slots=sorted(COVERAGE['forward']),dense_ks3_backward_slots=sorted(COVERAGE['backward']),
                        routed_ks3_backward_layers=[1,3,5,6,8],
                        retained_original_mlp_slots=list(keep),all_feedforward_paths_mapped=True,
                        original_mlp_fallback_is_fatal=True,transformed_forward_sha256=forward_sha,
                        actual_graph_warmup=True,dense_projection_optimizer=dict(optimizer_policy),
                        dense_geometry_optimizer=dict(geometry_policy),
                        dense_training_precision=dense_precision,dense_validation_precision='bf16',
                        dense_training_projection_gs32=dense_gs32,dense_validation_projection_gs32=True,
                        dense_geometry_coefficient_gs32=True,
                        dense_fp8_activation_scale_slots=scale_mapping if dense_precision=='fp8' else None)
    # Extend the original per-matrix output-LR/padding policy to the retained
    # bank. All existing labels execute the byte-for-byte original method.
    original_build=anvil.AnvilAndAdam._build_param_cfg
    def build(optimizer,param,label):
        if label!='retained_mlp_bank':return original_build(optimizer,param,label)
        saved=optimizer.param_table['mlp_bank']
        try:
            optimizer.param_table['mlp_bank']=optimizer.param_table[label]
            original_build(optimizer,param,'mlp_bank')
            optimizer.param_cfgs[param].label=label
        finally:optimizer.param_table['mlp_bank']=saved
    anvil.AnvilAndAdam._build_param_cfg=build
    original_optimizer=training.AnvilAndAdam
    def optimizer(*a,**kw):
        table=kw['param_table']
        table['ks3_dense_geometry']=dict(optim='adam',comms='replicated',adam_betas=[.9,.99],lr_mul=geometry_multiplier,wd_mul=0.)
        if keep:table['retained_mlp_bank']=dict(table['mlp_bank'])
        table['mlp_bank']=dict(table['mlp_bank'],lr_mul=multiplier)
        for key in ('scatter_order','work_order'):
            kw[key].append('ks3_dense_geometry')
            if keep:kw[key].insert(kw[key].index('mlp_bank')+1,'retained_mlp_bank')
        result=original_optimizer(*a,**kw)
        dense=[cfg for cfg in result.param_cfgs.values() if cfg.label=='mlp_bank']
        assert len(dense)==1
        cfg=dense[0];shape_scale=max(1.,features/768)**.5
        assert cfg.lr_mul==shape_scale*multiplier
        optimizer_policy.update(multiplier=multiplier,effective_multiplier=cfg.lr_mul,
                                shape_scale=shape_scale,base_lr=cfg.lr,weight_decay=cfg.weight_decay)
        dense_geometry=[cfg for cfg in result.param_cfgs.values() if cfg.label=='ks3_dense_geometry']
        assert len(dense_geometry)==1
        geom=dense_geometry[0]
        assert geom.lr_mul==geometry_multiplier
        geometry_policy.update(multiplier=geom.lr_mul,base_lr=geom.lr,
                               adam_betas=list(geom.adam_betas),weight_decay_multiplier=geom.wd_mul,
                               eps=geom.eps,routed_geometry_multiplier=geometry['lr_mul'])
        return result
    training.AnvilAndAdam=optimizer
    return FullKS3GPT
