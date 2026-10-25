"""Detached exactness and paired timing checks for register-only FP8 decode."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import torch
import triton as tr
import triton.language as tl
from wr_reproduction import fused_fp8
from wr_reproduction.packed_fp8_decode import native_packed
from wr_reproduction.check_shared_fp8_tiles import operands,oracle,check,execute,CASES,LATE_CASES,F4,F5
from wr_reproduction.check_shared_fp8_amax import exact


@tr.jit
def decode_check(X,SCALAR,PACKED,N:tl.constexpr,E5:tl.constexpr,SOURCE_E5:tl.constexpr,B:tl.constexpr=1024):
    i=tl.program_id(0)*B+tl.arange(0,B)
    bits=tl.load(X+i,i<N,other=0)
    left=fused_fp8._native(bits,E5,SOURCE_E5).to(tl.uint8,bitcast=True)
    right=native_packed(bits,E5,SOURCE_E5).to(tl.uint8,bitcast=True)
    tl.store(SCALAR+i,left,i<N);tl.store(PACKED+i,right,i<N)


def call(a,b,sa,sb,mode,aux,amax,decode,stores=True,out_dtype=torch.float32):
    fused_fp8.DECODE_MODE=decode
    return (*fused_fp8.run(a,b,sa,sb,out_dtype=out_dtype,mode=mode,post_scale=sa,
        dpre_scale=sb,amax=amax,aux=aux,emit_t=bool(mode),
        store_pre=mode==1 and stores,store_post=mode==1 and stores),amax)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.manual_seed(169);torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    assert fused_fp8.DECODE_MODE=='scalar' and fused_fp8.TILE_MODE=='legacy'
    fused_fp8.TILE_MODE='mi300-v2';fused_fp8.AMAX_MODE='partial'
    def emit(**row):
        row['utc']=datetime.now(timezone.utc).isoformat()
        with args.out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    formats=[];timings=[]
    try:
        # All byte pairs plus random words expose inter-byte carries/borrows.
        pairs=torch.arange(65536,device='cuda',dtype=torch.int64)
        patterned=torch.stack((pairs&255,pairs>>8,(pairs&255)^255,(pairs>>8)^255),1).to(torch.uint8).flatten()
        data=torch.cat((patterned,torch.randint(0,256,(1048579,),device='cuda',dtype=torch.uint8)))
        left=torch.empty_like(data);right=torch.empty_like(data)
        for source_e5 in (False,True):
            magnitude=data&127
            invalid=magnitude>=124 if source_e5 else magnitude==127
            expected=torch.where(invalid,128,torch.where(data==128,0,data)).to(torch.uint8)
            for tag_e5 in (False,True):
                decode_check[(tr.cdiv(data.numel(),1024),)](data,left,right,data.numel(),tag_e5,source_e5)
                assert torch.equal(left,expected) and torch.equal(right,expected)
        emit(event='packed_decode_bytes',passed=True,bytes_per_case=data.numel(),source_formats=2,target_tags=2,all_byte_pairs=True,tail_mask=True)
        sa=torch.tensor(.1173,device='cuda');sb=torch.tensor(.0831,device='cuda')
        # Non-divisible tails, both physical layouts, each source pair and each
        # epilogue, checked both exactly and against independently decoded math.
        for mode in (0,1,2):
            for ad in (F4,F5):
                for bd in (F4,F5):
                    for layout in ('rc','cr'):
                        a,b=operands(33,65,131,ad,bd,layout)
                        aux=torch.rand(33,65,device='cuda').to(F4) if mode==2 else None
                        old=call(a,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'scalar')
                        new=call(a,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'packed4')
                        exact(old,new)
                        errors=check(new,oracle(a,b,sa,sb,torch.float32,mode,aux),mode)
                        formats.append(dict(mode=mode,a_dtype=str(ad),b_dtype=str(bd),layout=layout,errors=errors))
        emit(event='packed_decode_formats',passed=True,cases=len(formats),exact=True,independent_oracles=True)
        # Exercise invalid/negative-zero bytes inside tuned GEMM for both
        # operands and source formats. NaN locations must also be identical.
        for ad in (F4,F5):
            for bd in (F4,F5):
                for side in ('a','b'):
                    a,b=operands(1025,257,131,ad,bd)
                    if side=='a':a.view(torch.uint8)[:,0]=(torch.arange(1025,device='cuda')%256).to(torch.uint8)
                    else:b.view(torch.uint8)[0,:]=(torch.arange(257,device='cuda')%256).to(torch.uint8)
                    exact(call(a,b,sa,sb,0,None,torch.zeros((),device='cuda'),'scalar'),
                          call(a,b,sa,sb,0,None,torch.zeros((),device='cuda'),'packed4'))
        for mode in (1,2):
            a,b=operands(1025,2816,768,F4 if mode==1 else F5,F4)
            aux=torch.rand(1025,2816,device='cuda').to(F4) if mode==2 else None
            for initial in (0.,1.e10,float('inf'),float('nan')):
                old=torch.tensor(initial,device='cuda');new=old.clone()
                for multiplier in (1.,.25,2.,.5):
                    aa=(a.float()*multiplier).to(a.dtype)
                    exact(call(aa,b,sa,sb,mode,aux,old,'scalar'),call(aa,b,sa,sb,mode,aux,new,'packed4'))
            for bad in (float('nan'),float('inf'),float('-inf')):
                aa=a.float().clone();aa[0,0]=bad;aa=aa.to(a.dtype)
                exact(call(aa,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'scalar'),
                      call(aa,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'packed4'))
        emit(event='packed_decode_invalid_and_delayed_amax',passed=True,all_byte_values_in_both_operands=True,prior_maxima_preserved=True,nonfinite_checked=True)
        for name,m,n,k,ad,bd,dtype,mode in CASES+LATE_CASES:
            a,b=operands(m,n,k,ad,bd)
            aux=torch.rand(m,n,device='cuda').to(F4) if mode==2 else None
            old=torch.zeros((),device='cuda');new=old.clone()
            baseline=lambda:call(a,b,sa,sb,mode,aux,old,'scalar',stores=False,out_dtype=dtype)
            candidate=lambda:call(a,b,sa,sb,mode,aux,new,'packed4',stores=False,out_dtype=dtype)
            exact(baseline(),candidate())
            base_samples=[];samples=[]
            for order in ((baseline,candidate),(candidate,baseline),(baseline,candidate)):
                for fn in order:
                    value=tr.testing.do_bench_cudagraph(fn,rep=50)
                    (samples if fn is candidate else base_samples).append(value)
            base=statistics.median(base_samples);value=statistics.median(samples)
            row=dict(name=name,m=m,n=n,k=k,mode=mode,baseline_ms=base,candidate_ms=value,
                     speedup=base/value,baseline_samples_ms=base_samples,candidate_samples_ms=samples,exact=True)
            timings.append(row);emit(event='packed_decode_timing',**row)
            del a,b,aux,old,new
        fused_fp8.DECODE_MODE='packed4'
        from wr_reproduction.check_mlp import main as check_mlp
        with torch.enable_grad():check_mlp(version='pr393',tokens=1024)
        artifact=args.out.parent/'packed-fp8-decode.json'
        artifact.write_text(json.dumps(dict(format_checks=formats,timing=timings,all_outputs_exact=True),indent=2,allow_nan=False)+'\n')
        from wr_reproduction.research import sha
        emit(event='kernel_correctness',passed=True,scope='Packed four-byte register decoder, original FP8 formats/caches/layouts/scales/epilogues',
             all_outputs_exact=True,original_dense_mlp_autograd_checked=True,
             full_convergence_verified=False,whole_update_speed_measured=False,
             artifact=str(artifact),artifact_sha256=sha(artifact),timing=timings)
    finally:
        fused_fp8.TILE_MODE='legacy';fused_fp8.AMAX_MODE='atomic';fused_fp8.DECODE_MODE='scalar'


if __name__=='__main__':main()
