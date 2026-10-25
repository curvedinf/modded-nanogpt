"""Bitwise dual-layout and multi-update scale check, plus paired GPU timing."""
import statistics
import torch
import triton as tr
from wr_reproduction.ks3_moe_fp8 import init_buffers,refresh_weights


@torch.no_grad()
def check(charts=1024,gs32=True):
    m=torch.nn.Module();m.device=torch.device('cuda');m.ks3_batched_weight_pack=True
    m.ks3_training_gs32=gs32
    m.ks3_geometry=torch.zeros((5,8,charts,8),device=m.device)
    m.moe_bank=(torch.randn(80,charts,768,device=m.device)*.025).bfloat16()
    init_buffers(m)
    names=['ks3_f8_up','ks3_f8_up_t','ks3_f8_down','ks3_f8_down_t',
           'ks3_weight_scales','ks3_weight_amax']
    checks=[]
    for iteration,exact in enumerate((True,False,False,True)):
        if iteration:m.moe_bank.add_(torch.randn_like(m.moe_bank)*.001)
        old_amax=m.ks3_weight_amax.clone()
        m.ks3_batched_weight_pack=False;refresh_weights(m,exact)
        expected={name:getattr(m,name).clone() for name in names}
        m.ks3_weight_amax.copy_(old_amax)
        m.ks3_batched_weight_pack=True;refresh_weights(m,exact)
        for name in names:
            got=getattr(m,name);want=expected[name]
            assert torch.equal(got.view(torch.uint8),want.view(torch.uint8)),(iteration,name)
        checks.append(dict(iteration=iteration,exact_scale=exact,bitwise_equal=names))
    # Refresh graphs replay with changing live weights and maxima; validate the
    # new side-effect buffers explicitly, independent of upstream buffer names.
    graph=torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):refresh_weights(m,False)
    for iteration in range(2):
        m.moe_bank.add_(torch.randn_like(m.moe_bank)*.001)
        old_amax=m.ks3_weight_amax.clone()
        m.ks3_batched_weight_pack=False;refresh_weights(m,False)
        expected={name:getattr(m,name).clone() for name in names}
        m.ks3_weight_amax.copy_(old_amax)
        m.ks3_batched_weight_pack=True;graph.replay()
        for name in names:
            got=getattr(m,name);want=expected[name]
            assert torch.equal(got.view(torch.uint8),want.view(torch.uint8)),('replay',iteration,name)
        checks.append(dict(iteration=iteration,captured_replay=True,exact_scale=False,bitwise_equal=names))
    samples={name:[] for name in ('loop','batched')}
    for repeat in range(3):
        for name in (('loop','batched') if repeat%2==0 else ('batched','loop')):
            m.ks3_batched_weight_pack=name=='batched'
            samples[name].append(tr.testing.do_bench_cudagraph(lambda:refresh_weights(m,False),rep=100))
    return dict(charts=charts,gs32=gs32,checks=checks,delayed_refresh_ms=samples,
                median_ms={k:statistics.median(v) for k,v in samples.items()},
                scope='Weight refresh only; whole-model speed not established')
