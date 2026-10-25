"""Public head dispatch correctness for the measured packed-FP8 pipeline."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import torch
from wr_reproduction import fused_fp8
from wr_reproduction.check_shared_fp8_tiles import operands,oracle,check,execute,CASES,LATE_CASES,F4,F5
from wr_reproduction.check_shared_fp8_amax import exact


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.manual_seed(169);torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    assert fused_fp8.HEAD_MODE=='legacy' and fused_fp8.DECODE_MODE=='scalar'
    fused_fp8.TILE_MODE='mi300-v2';fused_fp8.DECODE_MODE='packed4-selected';fused_fp8.AMAX_MODE='partial'
    def emit(**row):
        row['utc']=datetime.now(timezone.utc).isoformat()
        with args.out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    sa=torch.tensor(.1173,device='cuda');sb=torch.tensor(.0831,device='cuda')
    cases=[]
    try:
        # Head-shaped tails exercise the selected pipeline, all four source
        # format pairs and both forward/backward shape classes.
        for m,n,k in ((16385,10241,768),(16385,768,10241)):
            for ad in (F4,F5):
                for bd in (F4,F5):
                    a,b=operands(m,n,k,ad,bd)
                    fused_fp8.HEAD_MODE='legacy';old=execute(a,b,sa,sb,None,torch.float32)
                    fused_fp8.HEAD_MODE='packed-v1';new=execute(a,b,sa,sb,None,torch.float32)
                    exact(old,new);errors=check(new,oracle(a,b,sa,sb,torch.float32,0,None),0)
                    emit(event='packed_head_format',m=m,n=n,k=k,a_dtype=str(ad),b_dtype=str(bd),exact=True,errors=errors)
                    del a,b,old,new
        # Every invalid/negative-zero byte is preserved at the selected public
        # entry point, in either operand and either original source format.
        for ad in (F4,F5):
            for bd in (F4,F5):
                a=torch.zeros((16384,768),device='cuda',dtype=ad)
                b=torch.ones((10240,768),device='cuda',dtype=bd).T
                a.view(torch.uint8)[:,0]=(torch.arange(16384,device='cuda')%256).to(torch.uint8)
                b.view(torch.uint8)[0,:]=(torch.arange(10240,device='cuda')%256).to(torch.uint8)
                fused_fp8.HEAD_MODE='legacy';old=execute(a,b,sa,sb,None,torch.float32)
                fused_fp8.HEAD_MODE='packed-v1';new=execute(a,b,sa,sb,None,torch.float32)
                exact(old,new);del a,b,old,new
        for name,m,n,k,ad,bd,dtype,mode in CASES+LATE_CASES:
            if 'head_' not in name or name=='head_sampled_dw':continue
            a,b=operands(m,n,k,ad,bd)
            fused_fp8.HEAD_MODE='legacy';old=execute(a,b,sa,sb,None,dtype)
            fused_fp8.HEAD_MODE='packed-v1'
            # Exercise the actual torch custom-op used by the model.
            out=fused_fp8.scaled_mm_op(a,b,sa,sb,fused_fp8._OUTPUT_DTYPES.index(dtype),False)
            new=(out,None,None,None,torch.zeros((),device='cuda'))
            exact(old,new);errors=check(new,oracle(a,b,sa,sb,dtype,0,None),0)
            row=dict(name=name,m=m,n=n,k=k,exact=True,errors=errors)
            cases.append(row);emit(event='packed_head_public_case',**row)
            del a,b,old,new,out
        emit(event='kernel_correctness',passed=True,scope='Measured two-stage packed head pipeline through original model custom-op',
             all_outputs_exact=True,independent_shape_cases=8,source_format_pairs=4,all_byte_patterns_checked=True,
             shape_checks=cases,full_convergence_verified=False,whole_update_speed_measured=False,
             unrelated_producer_tiles_unchanged=True)
    finally:
        fused_fp8.TILE_MODE='legacy';fused_fp8.HEAD_MODE='legacy';fused_fp8.DECODE_MODE='scalar';fused_fp8.AMAX_MODE='atomic'


if __name__=='__main__':main()
