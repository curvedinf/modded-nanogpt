"""Explicit kernel-only in-memory dispatch overlay for immutable record sources.

No trainer is launched by importing this module. The import hook changes one
binding to the unavailable FA3 binary. NVRTC kernel objects are deferred during
import, then their actual CE calls are replaced by the tested ROCm kernel.
Any unported NVRTC invocation fails; it never silently falls back.
"""
import ast
import importlib.abc
import importlib.util
import sys
import torch
from wr_reproduction.contracts import ROOT,audit
from wr_reproduction import fp8,ce_kernel,attention_kernel,mlp_kernels,moe_kernels

class DeferredCUDAKernel:
    def set_shared_memory_config(self,*args):pass
    def __call__(self,*args,**kwargs):raise RuntimeError('Unported NVRTC call: reproduction cannot proceed')

class AttentionLoader(importlib.abc.Loader):
    def __init__(self,path):self.path=path
    def create_module(self,spec):return None
    def exec_module(self,module):
        source=self.path.read_text();tree=ast.parse(source);replaced=0
        for i,node in enumerate(tree.body):
            if isinstance(node,ast.ImportFrom) and node.module=='kernels':
                assert [n.name for n in node.names]==['get_kernel']
                tree.body[i]=ast.Assign(targets=[ast.Name(id='get_kernel',ctx=ast.Store())],
                                        value=ast.Name(id='_wr_unavailable_hopper',ctx=ast.Load()))
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='flash_attn_interface':
                assert isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='load_flash_attn3'
                node.value=ast.Name(id='_wr_attention_kernel',ctx=ast.Load());replaced+=1
        assert replaced==1
        module.__dict__['_wr_attention_kernel']=attention_kernel
        def unavailable(*a,**kw):raise RuntimeError('Hopper binary is replaced by the audited MI300 kernel')
        module.__dict__['_wr_unavailable_hopper']=unavailable
        module.__dict__['__file__']=str(self.path)
        exec(compile(ast.fix_missing_locations(tree),str(self.path),'exec'),module.__dict__)

class AttentionFinder(importlib.abc.MetaPathFinder):
    def __init__(self,path):self.path=path
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='track_1_short.model.attention':
            return importlib.util.spec_from_loader(fullname,AttentionLoader(self.path))

def _scaled_mm(a,b,scale_a,scale_b,*,out_dtype=None,use_fast_accum=False,**kwargs):
    if kwargs:raise RuntimeError(f'Unverified GEMM arguments: {tuple(kwargs)}')
    return fp8.mm(a,b,scale_a,scale_b,out_dtype=out_dtype or torch.bfloat16,use_fast_accum=use_fast_accum)

def install(version='pr393'):
    audit(version)
    if 'track_1_short.model.gpt' in sys.modules:raise RuntimeError('Install before importing original model')
    source=ROOT/version/'upstream';sys.path.insert(0,str(source))
    from track_1_short.perf.kernels import fp8_attention_quant
    from wr_reproduction.attention_quant import _quantize_dual_layout_packed_batched_kernel
    fp8_attention_quant._quantize_dual_layout_packed_batched_kernel=_quantize_dual_layout_packed_batched_kernel
    sys.meta_path.insert(0,AttentionFinder(source/'track_1_short/model/attention.py'))
    torch._scaled_mm=_scaled_mm
    compile_kernel=torch.cuda._compile_kernel
    torch.cuda._compile_kernel=lambda *a,**kw:DeferredCUDAKernel()
    try:
        from track_1_short.perf.kernels import cross_entropy as ce
        from track_1_short.perf.kernels import sampled_cross_entropy as sampled
        if version=='pr393':
            from track_1_short.perf.kernels import cplm
    finally:torch.cuda._compile_kernel=compile_kernel
    def ce_call(logits,targets,mtp_weights,prefix_targets,prefix_weight,losses,grad_input,
                n_rows,n_predict,A,B,C,grad_s):
        assert (A,B,C)==(23.,5.,7.5)
        out=ce_kernel.run(logits.view(torch.float8_e4m3fn),targets,mtp_weights,prefix_targets,prefix_weight,grad_s)
        losses.copy_(out[0]);grad_input.copy_(out[1])
    ce.ce_fwd_bwd.register_kernel('cuda',ce_call)
    # Sampled CE has the same mathematical kernel, with explicit candidate width.
    if hasattr(sampled,'sampled_ce_fwd_bwd'):sampled.sampled_ce_fwd_bwd.register_kernel('cuda',ce_call)
    if version=='pr393':
        def cplm_call(logits,targets,mtp_weights,prefix_targets,prefix_weight,p_copy,a_sink,copy_col,
                      losses,grad_input,d_pc,d_as,n_rows,n_predict,A,B,C,grad_s):
            assert (A,B,C)==(23.,5.,7.5)
            out=ce_kernel.run(logits.view(torch.float8_e4m3fn),targets,mtp_weights,prefix_targets,prefix_weight,
                              grad_s,p_copy=p_copy,a_sink=a_sink,copy_col=copy_col)
            for dst,src in zip((losses,grad_input,d_pc,d_as),out):dst.copy_(src)
        cplm.cplm_ce_fwd_bwd.register_kernel('cuda',cplm_call)
        from track_1_short.perf.kernels import moe_mlp
        moe_kernels.install(moe_mlp)
        torch._grouped_mm=moe_kernels.grouped_mm_validation
    from track_1_short.perf.kernels import mlp
    mlp_kernels.install(mlp)
    from track_1_short.optim import anvil
    from wr_reproduction import optimizer_kernel
    anvil.XTX=optimizer_kernel.gram_tall
    anvil.XXT=optimizer_kernel.gram_wide
    anvil.ba_plus_cAA=optimizer_kernel.poly_gram
    if version=='pr393':
        from track_1_short.perf.kernels import anvil_opt
        anvil_opt.gemm_xb_add=optimizer_kernel.gemm_xb_add
        # Hopper launch tiles exceed MI300's 64 KiB LDS. Change only kernel tiling.
        anvil_opt._GRAM_CFG={key:(64,64,32,4,1) for key in anvil_opt._GRAM_CFG}
