"""Numerical checks for direct chunk-gradient accumulation in KS3 attention.

CPU tests run normally. GPU checks are invoked only inside a registered pilot or
smoke by verify_attention_buffers, before that diagnostic builds the full model.
"""
import unittest
import torch

from ks3_research.attention import (buffered_attention, BufferedSelfAttention,
                                    grouped_attention, GroupedSelfAttention,
                                    gathered_attention, GatheredSelfAttention,
                                    gathered_attention32,Gathered32SelfAttention,
                                    vectorized_gathered_attention,VectorizedGatheredSelfAttention,
                                    fused_accumulated_attention,FusedAccumulationSelfAttention,chunks_for,
                                    fused64_attention,Fused64SelfAttention)
from mi300.portable import varlen_attention
from mi300.model.attention import AttnArgs, CausalSelfAttention, Yarn


def errors(actual, expected, dtype):
    actual, expected = actual.detach(), expected.detach()
    delta = actual.float()-expected.float()
    absolute = float(delta.abs().max())
    relative_rms = float(delta.norm()/expected.float().norm().clamp_min(1e-12))
    if dtype == torch.bfloat16:
        assert relative_rms <= .015, (absolute, relative_rms)
        assert absolute <= max(2e-4, .03*float(expected.float().abs().max())), (absolute,relative_rms)
    else:
        torch.testing.assert_close(actual,expected,atol=4e-6,rtol=4e-5)
    return dict(max_absolute=absolute,relative_rms=relative_rms)


def direct_check(device, dtype, qdim, vdim, window, implementation=buffered_attention, n=529):
    torch.manual_seed(81)
    heads = 3
    # Noncontiguous token/head strides, more than one query chunk, repeated
    # terminal seqlens padding, and overlapping backward key/value ranges.
    q = torch.randn(heads,n,qdim,device=device,dtype=dtype).transpose(0,1).detach().requires_grad_()
    k = torch.randn_like(q).requires_grad_()
    v = torch.randn(heads,n,vdim,device=device,dtype=dtype).transpose(0,1).detach().requires_grad_()
    boundaries = torch.tensor([0,17,289,n,n],device=device,dtype=torch.int32)
    old = varlen_attention(q,k,v,boundaries,window,.13)
    new = implementation(q,k,v,boundaries,window,.13)
    forward = errors(new,old,dtype)
    upstream = torch.randn_like(old)/old.numel()**.5
    original_grads = torch.autograd.grad(old,(q,k,v),upstream)
    new_grads = torch.autograd.grad(new,(q,k,v),upstream)
    gradients = []
    for i,(a,b) in enumerate(zip(new_grads,original_grads)):
        if window==0 and i<2:
            # A one-key softmax is exactly one: Q/K derivatives are analytically
            # zero. Flash SDPA can leave sub-1e-8 cancellation residue, for which
            # relative error is ill-conditioned. Test BOTH paths against zero
            # directly; this is much tighter than the general BF16 abs tolerance.
            for grad in (a,b):
                torch.testing.assert_close(grad,torch.zeros_like(grad),atol=1e-8,rtol=0)
            gradients.append(dict(expected_zero=True,max_absolute=float((a-b).abs().max()),
                                  actual_max_absolute=float(a.abs().max()),
                                  reference_max_absolute=float(b.abs().max())))
        else:
            gradients.append(errors(a,b,dtype))
    if window==0:
        # Analytically identity through V; SDPA's finite-precision softmax
        # backward can introduce about 1e-6 relative FP32 roundoff even in the
        # original path. Compare to identity with a tighter bound than the
        # general nonzero-gradient equivalence check above.
        for grad in (original_grads[2],new_grads[2]):
            torch.testing.assert_close(grad,upstream,atol=2e-8,rtol=4e-6)
    with torch.no_grad():
        torch.testing.assert_close(implementation(q,k,v,boundaries,window,.13),
                                   varlen_attention(q,k,v,boundaries,window,.13),atol=0,rtol=0)
    return dict(case='chunk_gradients',implementation=implementation.__name__,tokens=n,
                dtype=str(dtype),qdim=qdim,vdim=vdim,window=window,
                forward=forward,gradients=gradients,evaluation_bitwise_equal=True)


def wrapper_check(device,dtype,paired,qdim,vdim,layer_type=BufferedSelfAttention):
    torch.manual_seed(83)
    n, heads, dim = 544, 6, 24
    params = []
    def parameter(shape,scale=.03,kind=None):
        value = (torch.randn(shape,device=device,dtype=kind or dtype)*scale).requires_grad_()
        params.append(value)
        return value
    x = parameter((1,n,dim),1.)
    qk_w = parameter((2*heads*qdim,dim))
    v_w = parameter((heads*vdim,dim))
    o_w = parameter((dim,heads*vdim))
    aux = parameter((1,n,heads*128))
    gates = parameter((1,n,heads),.2,torch.float32)
    xsa = None if paired else parameter((1,n,heads),.1,torch.float32)
    lambdas = torch.tensor([.8,.9],device=device,requires_grad=True)
    gain = torch.tensor(.7,device=device,requires_grad=True)
    params.extend((lambdas,gain))
    yarn = Yarn(qdim,n,paired=paired,attn_scale=.12,device=device)
    args = AttnArgs(sa_lambdas=lambdas,seqlens=torch.tensor([0,16,288,n,n],device=device,dtype=torch.int32),
                    bm_size=137,yarn=yarn,key_offset=qdim>64,attn_gate_w=gates,aux_v=aux,
                    xsa_alpha=xsa,train_max_seq_len=512,o_gain=gain)
    old_layer = CausalSelfAttention(heads,128,qdim,vdim,n,paired=paired)
    new_layer = layer_type(heads,128,qdim,vdim,n,paired=paired)
    old = old_layer(x,args,qk_w,v_w,o_w)
    new = new_layer(x,args,qk_w,v_w,o_w)
    forward = errors(new,old,dtype)
    upstream = torch.randn_like(old)/old.numel()**.5
    original_grads = torch.autograd.grad(old,params,upstream)
    new_grads = torch.autograd.grad(new,params,upstream)
    names=['input','qk_projection','v_projection','output_projection','aux_values','gates']
    if not paired:names.append('xsa')
    names.extend(('sa_lambdas','output_gain'))
    gradients=[]
    for name,a,b in zip(names,new_grads,original_grads):
        try:gradients.append(dict(name=name,**errors(a,b,dtype)))
        except AssertionError as error:raise AssertionError(f'{name}: {error}') from error
    old_layer.eval(); new_layer.eval()
    with torch.no_grad():
        torch.testing.assert_close(old_layer(x,args,qk_w,v_w,o_w),new_layer(x,args,qk_w,v_w,o_w),atol=0,rtol=0)
    return dict(case='whole_attention_layer',implementation=layer_type.__name__,dtype=str(dtype),paired=paired,qdim=qdim,vdim=vdim,
                forward=forward,gradients=gradients,evaluation_bitwise_equal=True)


def ordered_accumulation_checks(device,dtype):
    from ks3_research.attention_accumulation import ordered_add
    torch.manual_seed(84)
    n,heads=7969,3
    chunks=chunks_for([0,17,1051,3741,n],1408)
    results=[]
    for width in (64,128):
        # Nonzero initial buffers, overlapping groups, multiple documents,
        # padding, unequal padded widths, and noncontiguous source/destination.
        got=torch.randn(heads,n,width,device=device,dtype=dtype).transpose(0,1)
        want=got.clone(memory_format=torch.preserve_format)
        groups=[chunks[i:i+32] for i in range(0,len(chunks),32)]
        for group in reversed(groups):
            length=max(hi-lo for _,hi,lo in group)
            source=torch.randn(len(group),length,heads,width+7,device=device,dtype=dtype).transpose(1,2)
            for i in reversed(range(len(group))):
                _,hi,lo=group[i]
                want[lo:hi].add_(source[i,:,:hi-lo,:width].transpose(0,1))
            ordered_add(got,source,group)
        torch.testing.assert_close(got,want,atol=0,rtol=0)
        results.append(dict(case='ordered_accumulation',dtype=str(dtype),width=width,
                            groups=len(groups),chunks=len(chunks),bitwise_equal=True))
    return results


def run_device_checks(device,backend='buffered_sdpa'):
    from control393.attention128 import fused128_attention,Fused128SelfAttention
    from control393.attention256 import fused256_attention,Fused256SelfAttention
    device = torch.device(device)
    dtypes = (torch.float32,torch.bfloat16) if device.type=='cuda' else (torch.float32,)
    results = []
    implementation,layer_type = dict(buffered_sdpa=(buffered_attention,BufferedSelfAttention),
        grouped_sdpa8=(grouped_attention,GroupedSelfAttention),
        gathered_sdpa8=(gathered_attention,GatheredSelfAttention),
        gathered_sdpa32=(gathered_attention32,Gathered32SelfAttention),
        gathered_sdpa32_vectorized=(vectorized_gathered_attention,VectorizedGatheredSelfAttention),
        gathered_sdpa32_fused=(fused_accumulated_attention,FusedAccumulationSelfAttention),
        gathered_sdpa64_fused=(fused64_attention,Fused64SelfAttention),
        gathered_sdpa128_fused=(fused128_attention,Fused128SelfAttention),
        gathered_sdpa256_evalbucket=(fused256_attention,Fused256SelfAttention))[backend]
    assert layer_type.attention is implementation, 'Wrapper must use the requested attention implementation'
    for dtype in dtypes:
        if backend in ('gathered_sdpa32_fused','gathered_sdpa64_fused','gathered_sdpa128_fused','gathered_sdpa256_evalbucket'):
            results.extend(ordered_accumulation_checks(device,dtype))
        for qdim,vdim,window in ((64,128,17),(128,64,384),(64,64,0)):
            results.append(direct_check(device,dtype,qdim,vdim,window,implementation))
        for paired,qdim,vdim in ((False,128,64),(True,64,128)):
            results.append(wrapper_check(device,dtype,paired,qdim,vdim,layer_type))
        if backend in ('grouped_sdpa8','gathered_sdpa8','gathered_sdpa32','gathered_sdpa32_vectorized','gathered_sdpa32_fused','gathered_sdpa64_fused','gathered_sdpa128_fused','gathered_sdpa256_evalbucket'):
            # Nine/33 chunks exercise two groups, final partial group, padding, and
            # overlapping K/V ranges across a group boundary.
            n=(65313 if backend.startswith('gathered_sdpa256') else
               32545 if backend.startswith('gathered_sdpa128') else
               16161 if backend.startswith('gathered_sdpa64') else
               7969 if backend.startswith('gathered_sdpa32') else 1825)
            results.append(direct_check(device,dtype,64,128,384,implementation,n=n))
    return dict(device=str(device),backend=backend,checks=results,passed=True)


class AttentionBufferTests(unittest.TestCase):
    def test_against_independent_dense_document_window_attention(self):
        torch.manual_seed(80)
        n,h,qd,vd = 273,2,8,16
        q=torch.randn(n,h,qd,dtype=torch.float64,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        v=torch.randn(n,h,vd,dtype=torch.float64,requires_grad=True)
        boundaries=torch.tensor([0,13,n,n],dtype=torch.int32)
        got=buffered_attention(q,k,v,boundaries,37,.13)
        pos=torch.arange(n)
        docs=pos>=13
        allowed=(pos[:,None]>=pos[None,:]) & (pos[:,None]-pos[None,:]<=37) & (docs[:,None]==docs[None,:])
        logits=torch.einsum('thd,shd->hts',q,k)*.13
        want=torch.einsum('hts,shd->thd',logits.masked_fill(~allowed,float('-inf')).softmax(-1),v)
        torch.testing.assert_close(got,want,atol=1e-12,rtol=1e-12)
        gradient=torch.randn_like(want)
        a=torch.autograd.grad(got,(q,k,v),gradient)
        b=torch.autograd.grad(want,(q,k,v),gradient)
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)

    def test_chunk_and_whole_layer_equivalence(self):
        self.assertTrue(run_device_checks('cpu')['passed'])

    def test_grouped_chunk_and_whole_layer_equivalence(self):
        self.assertTrue(run_device_checks('cpu','grouped_sdpa8')['passed'])

    def test_gathered_chunk_and_whole_layer_equivalence(self):
        self.assertTrue(run_device_checks('cpu','gathered_sdpa8')['passed'])

    def test_gathered32_chunk_and_whole_layer_equivalence(self):
        self.assertTrue(run_device_checks('cpu','gathered_sdpa32')['passed'])

    def test_vectorized_gathered_chunk_and_whole_layer_equivalence(self):
        self.assertTrue(run_device_checks('cpu','gathered_sdpa32_vectorized')['passed'])

    def test_fused_accumulation_chunk_and_whole_layer_equivalence(self):
        self.assertTrue(run_device_checks('cpu','gathered_sdpa32_fused')['passed'])

    def test_fused64_chunk_and_whole_layer_equivalence(self):
        self.assertTrue(run_device_checks('cpu','gathered_sdpa64_fused')['passed'])

    def test_fused_accumulation_requires_new_timing_pilot(self):
        import json
        from ks3_research.planning import load_config,plan
        config=load_config('configs/ks3/ablation_b64k_fast_trial.json')
        with open(config['pilot_path']) as handle:
            pilot=json.load(handle)
        config['attention_backend']='gathered_sdpa32_fused'
        with self.assertRaisesRegex(ValueError,'Run a pilot'):
            plan(config,pilot)

    def test_grouped_against_independent_dense_document_window_attention(self):
        torch.manual_seed(82)
        n=273
        q=torch.randn(n,2,8,dtype=torch.float64,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        v=torch.randn(n,2,16,dtype=torch.float64,requires_grad=True)
        boundaries=torch.tensor([0,13,n,n],dtype=torch.int32)
        got=grouped_attention(q,k,v,boundaries,37,.13,group_size=2)
        pos=torch.arange(n)
        docs=pos>=13
        mask=(pos[:,None]>=pos[None,:]) & (pos[:,None]-pos[None,:]<=37) & (docs[:,None]==docs[None,:])
        logits=torch.einsum('thd,shd->hts',q,k)*.13
        want=torch.einsum('hts,shd->thd',logits.masked_fill(~mask,float('-inf')).softmax(-1),v)
        torch.testing.assert_close(got,want,atol=1e-12,rtol=1e-12)
        gradient=torch.randn_like(want)
        a=torch.autograd.grad(got,(q,k,v),gradient)
        b=torch.autograd.grad(want,(q,k,v),gradient)
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)


if __name__=='__main__':
    unittest.main(verbosity=2)
