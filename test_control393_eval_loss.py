"""Full-vocabulary canonical CPLM parity and paired reduction timings."""
import statistics
import time
import unittest
import torch
from control393.eval_loss import softcap_table,canonical_cplm_loss


def oracle(z,inputs,targets,mask,pc,asin,copy_col):
    logits=(23*torch.sigmoid((z+5)/7.5)).float()
    shifts=torch.arange(8,device=z.device,dtype=torch.uint8)
    dropped=(mask[inputs,:,None]>>shifts&1).view(logits.shape).bool()
    dropped[:,copy_col]=False
    logits=logits.masked_fill(dropped,-60.)
    lse=logits.logsumexp(-1)
    pl=(logits.gather(1,targets[:,None])[:,0]-lse).exp()
    alpha=(logits[:,copy_col]-lse).exp()
    p=pl/(1-alpha).clamp_min(1e-6)*(1-alpha*(1-asin))+alpha*pc
    return -p.clamp_min(1e-30).log()


class EvalLossTests(unittest.TestCase):
    def test_exhaustive_bf16_rounding_table(self):
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16)
        z=bits.view(torch.bfloat16)
        table=softcap_table('cpu')
        want=(23*torch.sigmoid((z+5)/7.5)).float()
        torch.testing.assert_close(table[bits.long()&65535],want,rtol=0,atol=0,equal_nan=True)

    def test_copy_survives_all_dropped_mask(self):
        z=torch.zeros(3,16,dtype=torch.bfloat16)
        mask=torch.full((1,2),255,dtype=torch.uint8)
        loss=oracle(z,torch.zeros(3,dtype=torch.long),torch.tensor([15,4,4]),mask,
                    torch.tensor([0.,1.,0.]),torch.tensor([1.,0.,0.]),15)
        self.assertEqual(float(loss[1]),0.)
        self.assertGreater(float(loss[2]),60.)


@torch.no_grad()
def run_device_checks(device,block=4096):
    torch.manual_seed(393126)
    table=softcap_table(device);checks=[]
    for n,vocab,scale in ((1,128,1.),(17,1024,30.),(65,50304,7.),(257,50304,1.)):
        copy_col=min(50257,vocab-1)
        z=(torch.randn(n,vocab,device=device)*scale).bfloat16()
        # Exercise saturation and the original intermediate-rounding path.
        z[0,:8]=torch.tensor([-1000.,-100.,-5.,0.,5.,20.,100.,1000.],device=device)
        inputs=torch.randint(0,37,(n,),device=device)
        targets=torch.randint(0,vocab,(n,),device=device);targets[0]=copy_col
        mask=torch.randint(0,256,(37,vocab//8),device=device,dtype=torch.uint8)
        pc=torch.rand(n,device=device);asin=torch.rand(n,device=device)
        for variant in ('random','all_dropped','none_dropped'):
            if variant=='all_dropped':mask.fill_(255);pc.zero_();asin.fill_(1.)
            if variant=='none_dropped':mask.zero_();pc.fill_(.25);asin.zero_()
            expected=oracle(z,inputs,targets,mask,pc,asin,copy_col)
            actual=canonical_cplm_loss(z,inputs,targets,mask,pc,asin,table,copy_col=copy_col,block=block)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-5)
            mean_error=float((actual.double()-expected.double()).mean().abs())
            assert mean_error<=2e-6,mean_error
            checks.append(dict(rows=n,vocab=vocab,variant=variant,
                               max_absolute=float((actual-expected).abs().max()),mean_error=mean_error))
    n,vocab=4096,50304
    z=torch.randn(n,vocab,device=device,dtype=torch.bfloat16)
    inputs=torch.randint(0,37,(n,),device=device);targets=torch.randint(0,vocab,(n,),device=device)
    mask=torch.randint(0,256,(37,vocab//8),device=device,dtype=torch.uint8)
    pc=torch.rand(n,device=device);asin=torch.rand(n,device=device)
    funcs=dict(eager=lambda:oracle(z,inputs,targets,mask,pc,asin,50257),
               fused=lambda:canonical_cplm_loss(z,inputs,targets,mask,pc,asin,table,block=block))
    samples={key:[] for key in funcs}
    for repetition in range(7):
        for name in (('eager','fused') if repetition%2==0 else ('fused','eager')):
            torch.cuda.synchronize();start=time.perf_counter();out=funcs[name]()
            torch.cuda.synchronize();duration=time.perf_counter()-start
            if repetition>=2:samples[name].append(duration)
            del out
    medians={key:statistics.median(value) for key,value in samples.items()}
    return dict(passed=True,checks=checks,block=block,benchmark=dict(rows=n,vocab=vocab,
        samples_seconds=samples,median_seconds=medians,speedup=medians['eager']/medians['fused']))


if __name__=='__main__':unittest.main()
