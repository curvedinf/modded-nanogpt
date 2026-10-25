"""Short component timing; no optimizer or training trial."""
import sys,json,torch,triton as tr
from wr_reproduction.contracts import ROOT
sys.path.insert(0,str(ROOT/'pr393/upstream'))
from track_1_short.perf.kernels import moe_mlp as moe
from wr_reproduction import ks3_moe as bf,ks3_moe_fp8 as fp

def main():
    torch.manual_seed(169);torch.set_num_threads(1)
    t,e,n,d=16384,8,1024,768;device='cuda';f8=torch.float8_e4m3fn
    x=torch.randn(t,d,device=device,dtype=torch.bfloat16);grad=torch.randn_like(x)*.02
    expert=torch.randint(e,(t,),device=device);dest,src,offs,tiles,counts=moe.moe_dispatch(expert,e)
    up=(torch.randn(e,n,d,device=device)*.025).bfloat16();down=(torch.randn(e,d,n,device=device)*.02).bfloat16()
    g=torch.randn(e,n,8,device=device)*.01;g[...,6]=.75
    bu=bf.pack(up.reshape(-1,d)).reshape_as(up);bd=bf.pack(down.reshape(-1,n)).reshape_as(down)
    scalar=lambda v:torch.tensor([v],device=device,dtype=torch.float32)
    caches=[];scales=[]
    for w in (up,down):
        q=torch.empty_like(w,dtype=f8);qt=torch.empty(w.shape[0],w.shape[2],w.shape[1],device=device,dtype=f8)
        s=w.float().abs().amax((1,2))/448
        for i in range(e):
            a=w[i];r,c=a.shape
            bf._pack[(tr.cdiv(r,8),tr.cdiv(c,32))](a,q[i],r,c,*a.stride(),SCALE=s[i:i+1],OT=qt[i],FP8=True)
        caches.append((q,qt));scales.append(s)
    (u,ut),(w,wt)=caches;us,ws=scales
    xs,ps,gs,ds=map(scalar,(2**-4,.01,2**-6,.0001));pa,da=scalar(0),scalar(0)
    bank=torch.zeros(16,n,d,device=device,dtype=torch.bfloat16);flag=torch.zeros(1,device=device,dtype=torch.int32)
    gate=torch.ones(t,device=device)
    bf_calls=[];fp_calls=[];dw_calls=[]
    bm,fr,fd=bf.mm,fp.run,fp.grouped_dw
    def trace_b(*a,**kw):bf_calls.append((a,kw));return bm(*a,**kw)
    def trace_f(*a,**kw):fp_calls.append((a,kw));return fr(*a,**kw)
    def trace_d(*a,**kw):dw_calls.append((a,kw));return fd(*a,**kw)
    bf.mm=trace_b;fp.run=trace_f;fp.grouped_dw=trace_d
    def bforward():return bf.forward(x,src,tiles,bu,bd,g)
    by,bx,bc,bh=bforward()
    def bbackward():return bf.backward(grad,by,bx,bc,bh,gate,src,offs,tiles,bu,bd,g,bank,0)
    bbackward()
    def fforward():return fp.forward(x,src,tiles,u,w,us,ws,xs,ps,pa,g)
    fy,fc,fx,fh=fforward()
    def fbackward():return fp.backward(grad,fy,fc,fx,fh,gate,src,offs,tiles,ut,wt,us,ws,xs,gs,ps,ds,da,bank,flag,g,0)
    fbackward();bf.mm=bm;fp.run=fr;fp.grouped_dw=fd
    bench=lambda fn:tr.testing.do_bench_cudagraph(fn,rep=100)
    result={'tokens':t,'experts':e,'charts':n,'block_ms':{},'matrix_products':[]}
    for name,fn in [('bf16_forward',bforward),('bf16_backward',bbackward),('fp8_forward',fforward),('fp8_backward',fbackward)]:result['block_ms'][name]=bench(fn)
    m=src.numel();qx=torch.empty((m,d),device=device,dtype=f8);qxt=torch.empty((d,m),device=device,dtype=f8)
    qh=torch.empty((m,n),device=device,dtype=f8);qht=torch.empty((n,m),device=device,dtype=f8)
    hb=torch.empty((m,n),device=device,dtype=torch.bfloat16)
    result['producer_ms']={
      'bf16_input_gather_pack':bench(lambda:bf.pack(torch.where((src<t)[:,None],x[src.clamp(0,t-1).long()],0).contiguous())),
      'fp8_input_gather_pack_dual':bench(lambda:bf._pack[(tr.cdiv(m,8),tr.cdiv(d,32))](x,qx,m,d,*x.stride(),SCALE=xs,OT=qxt,SRC=src,INPUT_ROWS=t,FP8=True)),
      'bf16_geometry_pack':bench(lambda:bf._geometry[(tr.cdiv(m,16),tr.cdiv(n,32))](bc,g,hb,tiles,m,n,num_warps=4)),
      'fp8_geometry_pack_dual_legacy_atomic':bench(lambda:bf._geometry[(tr.cdiv(m,16),tr.cdiv(n,32))](fc,g,qh,tiles,m,n,HT=qht,SCALE=ps,AMAX=pa,FP8=True,num_warps=4))}
    for i,label in enumerate(('input_projection','output_projection','feature_gradient','input_gradient')):
        ba,bk=bf_calls[i];fa,fk=fp_calls[i]
        result['matrix_products'].append({'name':label,'bf16_operands':[str(ba[0].dtype),str(ba[1].dtype)],'fp8_operands':[str(fa[0].dtype),str(fa[1].dtype)],'bf16_ms':bench(lambda:bm(*ba,**bk)),'fp8_ms':bench(lambda:fr(*fa,**fk))})
    for i,(a,kw) in enumerate(dw_calls):
        result['matrix_products'].append({'name':('output_weight_gradient','input_weight_gradient')[i],'fp8_operands':[str(a[0].dtype),str(a[1].dtype)],'fp8_ms':bench(lambda:fd(*a,**kw))})
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
