import argparse,hashlib,importlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('tree',choices=['original','clean']);p.add_argument('kind',choices=['init','block']);a=p.parse_args()
root=Path('/root/modded-nanogpt');snapshot=root/'logs/wr-reproduction/h100-research/20261029T003007Z'
if a.tree=='original':
 env=json.loads((snapshot/'settings.json').read_text())['environment'];os.environ.update(env)
 sys.path[:0]=[str(snapshot/'source'),str(snapshot/'overlay')]
 module='experiments.h100_ks3';core_module='wr_reproduction.ks3_moe'
else:
 sys.path.insert(0,'/root/modded-nanogpt-ks3-pr');module='track_1_short.ks3';core_module=module+'.core'
import torch
import triton as tr
torch.set_num_threads(1);torch.manual_seed(471)
cls=importlib.import_module(module+'.port').install()
out=root/'research_results/wr_reproduction/h100-research/clean-port';out.mkdir(exist_ok=True)
if a.kind=='init':
 m=cls(vocab_size=50257,num_layers=11,num_heads=4,head_dim=128,model_dim=512,max_seq_len=8192,ngram_dim=512,world_size=8,rank=0,device=torch.device('cpu'))
 hashes={name:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()} for name,v in m.named_parameters()}
 hashes['next_rng']=torch.rand(16).tolist()
 (out/(a.tree+'-init.json')).write_text(json.dumps(hashes,indent=2)+'\n');print(a.tree,len(hashes),'initialized entries')
else:
 torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
 core=importlib.import_module(core_module);fp8=importlib.import_module(module+'.fp8')
 t,n,d=16384,1536,512
 scalar=lambda x:torch.tensor([x],device='cuda')
 rows=torch.randn(t,d,device='cuda',dtype=torch.bfloat16).requires_grad_()
 router=(torch.randn(8,d,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_();post=scalar(.8).requires_grad_()
 w1=torch.randn(8,n,d,device='cuda',dtype=torch.bfloat16)*.02;w2=torch.randn_like(w1)*.02
 up=core.pack(w1.reshape(-1,d)).reshape_as(w1);down=torch.stack([core.pack(w.T) for w in w2])
 g=torch.randn(8,n,8,device='cuda')*.01;g[...,6]=.75;g[...,4:6]=0.;g.requires_grad_()
 bank=torch.empty(16,n,d,device='cuda',dtype=torch.bfloat16);flag=torch.zeros((),device='cuda',dtype=torch.int32)
 grad=torch.randn_like(rows)*.02;gp=torch.randn(t,8,device='cuda')*.002
 us=up.float().abs().amax((1,2)).clamp_min(1e-12)/448;ws=down.float().abs().amax((1,2)).clamp_min(1e-12)/448
 uf=(up.float()/us[:,None,None]).clamp(-448,448).to(torch.float8_e4m3fn);wf=(down.float()/ws[:,None,None]).clamp(-448,448).to(torch.float8_e4m3fn)
 caches=(uf,uf.transpose(1,2).contiguous(),wf,wf.transpose(1,2).contiguous())
 xs,gs,ps,pa,ds,da=scalar(8/448),scalar(2**-6),scalar(1.),scalar(0.),scalar(2**-6),scalar(0.)
 scales=(us,ws,xs,gs,ps,pa,ds,da)
 for _ in range(3):
  for z in (rows,router,post,g):z.grad=None
  y,probs,counts=fp8.Routed.apply(rows,router,post,w1,w2,g,caches,scales,bank,flag,0)
  torch.autograd.backward((y,probs),(grad,gp))
  ps.copy_(pa*1.03/448);ds.copy_(da*1.25/57344)
 tensors={k:v.detach().cpu() for k,v in zip(('output','dx','router','post','geometry','weights','post_amax','dpre_amax'),(y,rows.grad,router.grad,post.grad,g.grad,bank,pa,da))}
 # Evaluate the original BF16 path, independently of the training cache.
 core.BF16_TUNED_KERNELS=False
 from track_1_short.perf.kernels.moe_mlp import moe_dispatch
 _,experts=probs.max(-1);dest,src,offs,tiles,_=moe_dispatch(experts,8)
 tensors['eval']=core.forward(rows.detach(),src,tiles,up,down,g.detach())[0].cpu()
 torch.save(tensors,out/(a.tree+'-block.pt'));print(a.tree,{k:list(v.shape) for k,v in tensors.items()})
