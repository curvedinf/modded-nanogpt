"""Independent checks through the actual opt-in shared-FP8 public dispatch."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import torch
from wr_reproduction import fused_fp8
from wr_reproduction.check_shared_fp8_tiles import execute,oracle,check,operands,CASES,LATE_CASES,F4,F5


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tile-policy',choices=['mi300-v1','mi300-v2'],default='mi300-v1')
    args=parser.parse_args();torch.manual_seed(169);torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    def emit(**row):
        row['utc']=datetime.now(timezone.utc).isoformat()
        with args.out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    sa=torch.tensor(.1173,device='cuda');sb=torch.tensor(.0831,device='cuda')
    assert fused_fp8.TILE_MODE=='legacy','Default control dispatch changed'
    fused_fp8.TILE_MODE=args.tile_policy
    if args.tile_policy=='mi300-v2':
        from wr_reproduction.shared_fp8_tiles import select_tiles,select_tiles_v2
        assert select_tiles_v2(1024,257,131,(131,1),(1,131))!=select_tiles(1024,257,131,(131,1),(1,131))
    try:
        formats=[]
        for ad in (F4,F5):
            for bd in (F4,F5):
                a,b=operands(1024,257,131,ad,bd)
                errors=check(execute(a,b,sa,sb,None,torch.float32),oracle(a,b,sa,sb,torch.float32,0,None),0)
                formats.append(dict(a_dtype=str(ad),b_dtype=str(bd),errors=errors))
        # Force tuned dispatch while exercising every stored FP8 byte pattern.
        for ad in (F4,F5):
            bits=torch.zeros((1024,128),device='cuda',dtype=torch.uint8)
            bits[:,0]=(torch.arange(1024,device='cuda')%256).to(torch.uint8)
            a=bits.view(ad);b=torch.zeros((128,128),device='cuda');b[0]=1
            b=b.to(F4).T.contiguous().T
            actual=execute(a,b,sa,sb,None,torch.float32)[0]
            expected=a[:,0].float();finite=expected.isfinite()
            assert torch.equal(actual.isfinite(),finite[:,None].expand_as(actual))
            torch.testing.assert_close(actual[finite],(expected[finite]*sa*sb)[:,None].expand_as(actual[finite]),rtol=1e-6,atol=1e-8)
        emit(event='shared_fp8_public_formats',passed=True,checks=formats,all_byte_patterns_checked=True)
        cases=[]
        with torch.no_grad():
            for name,m,n,k,ad,bd,dtype,mode in (CASES+LATE_CASES if args.tile_policy=='mi300-v2' else CASES):
                a,b=operands(m,n,k,ad,bd)
                aux=torch.randn(m,n,device='cuda').abs().to(F4) if mode==2 else None
                reference=oracle(a,b,sa,sb,dtype,mode,aux)
                errors=check(execute(a,b,sa,sb,None,dtype,mode,aux),reference,mode)
                case=dict(name=name,m=m,n=n,k=k,mode=mode,relative_errors=errors)
                cases.append(case);emit(event='shared_fp8_public_case',**case)
                del a,b,aux,reference
        # Existing independently decoded oracle checks the unchanged upstream
        # dense MLP autograd with enough rows to exercise the tuned dispatch.
        from wr_reproduction.check_mlp import main as check_mlp
        check_mlp(version='pr393',tokens=1024)
        evaluation_check=None
        if args.tile_policy=='mi300-v2':
            from wr_reproduction.check_research_evaluation import check_gpu
            evaluation_check=check_gpu()
        emit(event='kernel_correctness',passed=True,scope='Actual shared FP8 dispatch plus unchanged upstream dense MLP autograd',
             tile_policy=args.tile_policy,accepted_evaluation_gpu_check=evaluation_check,
             default_dispatch_legacy=True,all_byte_patterns_checked=True,source_format_pairs=4,
             independent_shape_cases=len(cases),shape_errors=cases,original_dense_mlp_autograd_checked=True,
             dense_mlp_check_seed=393,dense_mlp_check_tokens=1024,full_convergence_verified=False,whole_update_speed_measured=False)
    finally:fused_fp8.TILE_MODE='legacy'


if __name__=='__main__':main()
