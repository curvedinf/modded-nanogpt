"""Temporary held-out evaluation without consuming any training state."""
from contextlib import contextmanager
from dataclasses import replace
import glob
import hashlib
import json
import random
import time
import numpy as np
import torch
from wr_reproduction.probe_study import signature,digest,DEFINITION

class FixedProbe:
    def __init__(self,runner,protocol,backend_hash,*,reserved=False):
        self.runner=runner;self.protocol=protocol;self.reserved=reserved
        self.signature=signature(protocol,runner.research_config['validation_data'],backend_hash,reserved=reserved)
        self.signature_hash=digest(self.signature)
        self.batches=None;self.fixed_yarns=None;self.cumulative=0.;self.diagnostics=[]

    def prepare(self):
        """All allocations, input reads and fixed-table construction are charged at first use."""
        if self.batches is not None:return
        from track_1_short.data import BOS_ID,ngram_row_ids
        from track_1_short.model.attention import Yarn
        m=self.runner.model;p=self.signature;n=p['chunk_tokens']
        path=sorted(glob.glob(self.runner.hp.val_files))[0]
        header=np.fromfile(path,dtype=np.int32,count=256)
        assert header[0]==20240520 and header[1]==1
        assert p['offset']+p['tokens']+1<=int(header[2])
        tokens=np.fromfile(path,dtype=np.uint16,offset=1024+2*p['offset'],count=p['tokens']+1).astype(np.int32)
        self.batches=[]
        for start in range(0,p['tokens'],n):
            inp=torch.from_numpy(tokens[start:start+n].copy())
            target=torch.from_numpy(tokens[start+1:start+n+1].astype(np.int64))
            boundaries=torch.nonzero(inp==BOS_ID)[:,0]
            # Same original evaluation segmentation, with explicit fixed chunk edges.
            cu=torch.full((max(128,((len(boundaries)+2+127)//128)*128),),n,dtype=torch.int32)
            cu[0]=0;cu[1:len(boundaries)+1]=boundaries
            ids=ngram_row_ids(inp)
            self.batches.append(tuple(t.cuda() for t in (inp,target,cu,ids)))
        self.fixed_yarns=[]
        h=hashlib.sha256()
        for index,source in enumerate(m.yarns):
            y=Yarn(source.head_dim,n,paired=source.paired,attn_scale=source.base_attn_scale,device=source.device)
            windows=(384,896,1408) if index==2 else (128,384,640)
            y.apply(windows[0],windows[1]);y.apply(windows[1],windows[2])
            h.update(y.angular_freq.detach().cpu().contiguous().numpy().tobytes())
            h.update(json.dumps([y.head_dim,y.paired,y.attn_scale]).encode())
            self.fixed_yarns.append(y)
        assert h.hexdigest()==p['yarn_sha256'],h.hexdigest()

    @contextmanager
    def temporary_state(self):
        m=self.runner.model;n=self.signature['chunk_tokens']
        modes=[(module,module.training) for module in m.modules()]
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state(),random.getstate(),np.random.get_state())
        tensors=[(m.ngram_cache[:2*n],m.ngram_cache[:2*n].clone()),(m.cplm_diag,m.cplm_diag.clone())]
        # Evaluation reads freshly packed raw KS3 weights but never consumes delayed FP8 refreshes.
        for name in ('ks3_up','ks3_down','_mlp_ks3_up','_mlp_ks3_down'):
            if hasattr(m,name):
                t=getattr(m,name);tensors.append((t,t.clone()))
        yarns=[]
        for y,fixed in zip(m.yarns,self.fixed_yarns,strict=True):
            yarns.append((y,y.attn_scale,y.valid_rows))
            for t in (y.angular_freq,y.factor1[:n],y.factor2[:n]):tensors.append((t,t.clone()))
        try:
            m.eval()
            for y,f in zip(m.yarns,self.fixed_yarns,strict=True):
                y.angular_freq.copy_(f.angular_freq);y.attn_scale=f.attn_scale
                y.factor1[:n].copy_(f.factor1);y.factor2[:n].copy_(f.factor2)
            if hasattr(m,'prepare_ks3_eval'):m.prepare_ks3_eval()
            if hasattr(m,'_refresh_dense_ks3'):m._refresh_dense_ks3()
            yield
        finally:
            for target,saved in tensors:target.copy_(saved)
            for y,scale,valid in yarns:y.attn_scale=scale;y.valid_rows=valid
            m.train(modes[0][1])
            for module,mode in modes:module.training=mode
            torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);random.setstate(rng[2]);np.random.set_state(rng[3])

    @torch.no_grad()
    def score(self):
        from wr_reproduction.probe_diagnostics import activations,activation_report
        self.prepare();r=self.runner;m=r.model;losses=[];diagnostics=[]
        with self.temporary_state():
            cfg=replace(r.tm.get_forward_args(),ws_short=640,ws_long=1408,train_max_seq_len=2560,sampled_loss=None)
            for index,(inp,target,cu,ids) in enumerate(self.batches):
                if r.observations:r.observations.check()
                slots=r.tables.load(ids)
                with activations(r.ks3 and index==0 and not self.reserved,diagnostics):
                    loss=m(inp,target,cu,slots,cfg)
                losses.append(loss.float().mean())
        self.diagnostics=activation_report(diagnostics)
        return torch.stack(losses).cpu().tolist()

    def observe(self,step,clock_start):
        start=time.perf_counter();torch.cuda.synchronize();checkpoint=time.perf_counter()
        losses=self.score()
        torch.cuda.synchronize();end=time.perf_counter();seconds=end-start;self.cumulative+=seconds
        return dict(event='reserved_confirmation' if self.reserved else 'probe',measurement=DEFINITION,
                    loss=sum(losses)/len(losses),chunk_losses=losses,probe_signature=self.signature,
                    signature_hash=self.signature_hash,step=step,checkpoint_elapsed_seconds=checkpoint-clock_start,probe_start_seconds=start-clock_start,
                    probe_completion_seconds=end-clock_start,probe_seconds=seconds,
                    cumulative_probe_seconds=self.cumulative,setup_and_restore_included=True,
                    train_seconds=end-clock_start,final_validation=False,ks3_activation_diagnostics=self.diagnostics)
