"""Optional ordinary torch.compile fusion of the existing CPLM slab expression.

Head GEMMs, positive-gradient row sampling, and checkpoint boundaries remain in
control393.loss. The copy-column index is a device scalar because its sampled
vocabulary position changes with the negative sweep; it must not specialize a
new graph on each position. No extra compiler or inductor flags are used.
"""
import torch
import torch.nn.functional as F


def slab_loss(logits,ys,pys,p,a,mtp,prefix,copy_column):
    logp=F.log_softmax(23*torch.sigmoid((logits.float()+5)/7.5),-1)
    pl=logp.gather(1,ys[:,0,None])[:,0].exp()
    alpha=logp.gather(1,copy_column.expand(logp.shape[0],1))[:,0].exp()
    mixed=pl*(1+alpha*a/(1-alpha).clamp_min(1e-6))+alpha*p
    loss=-mtp[0]*torch.log(mixed+1e-9)
    for i in range(1,ys.shape[1]):
        y=ys[:,i];loss=loss-mtp[i]*logp.gather(1,y.clamp_min(0)[:,None])[:,0]*(y>=0)
    return loss-prefix[0]*logp.gather(1,pys.clamp_min(0)[:,None])[:,0]*(pys>=0)


compiled_slab_loss=torch.compile(slab_loss,dynamic=False,fullgraph=True)
