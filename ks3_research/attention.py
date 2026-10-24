"""Same local SDPA, with chunk gradients accumulated without dense slice backwards.

The portable reference makes each query/key/value slice an autograd node. Its
backward repeatedly allocates and adds a full-sequence gradient per small chunk.
This Function recomputes the same attention chunks, but differentiates detached
chunk leaves and accumulates directly into three sequence-sized buffers.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from mi300.portable import _attend, qk_norm_rope, varlen_attention
from mi300.model.attention import CausalSelfAttention, Yarn


def chunks_for(boundaries, window):
    chunks = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        for lo in range(start, end, 256):
            chunks.append((lo, min(lo+256, end), max(start, lo-window)))
    return chunks


class _BufferedSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, seqlens, window, scale):
        ctx.chunks = chunks_for(seqlens.detach().cpu().tolist(), window)
        ctx.window, ctx.scale = window, scale
        ctx.save_for_backward(q, k, v)
        output = [_attend(q[lo:hi], k[kl:hi], v[kl:hi], lo, kl, window, scale)
                  for lo,hi,kl in ctx.chunks]
        return torch.cat(output)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        # Reversed chunk order follows the reference graph's usual reverse
        # construction order. BF16 accumulation may still differ by rounding.
        for lo, hi, kl in reversed(ctx.chunks):
            with torch.enable_grad():
                qc = q[lo:hi].detach().requires_grad_(True)
                kc = k[kl:hi].detach().requires_grad_(True)
                vc = v[kl:hi].detach().requires_grad_(True)
                out = _attend(qc, kc, vc, lo, kl, ctx.window, ctx.scale)
                gq, gk, gv = torch.autograd.grad(out, (qc,kc,vc), grad_output[lo:hi])
            dq[lo:hi].copy_(gq)  # Query chunks do not overlap.
            dk[kl:hi].add_(gk)
            dv[kl:hi].add_(gv)
        return dq, dk, dv, None, None, None


def buffered_attention(q, k, v, seqlens, window, scale):
    # Full evaluation retains the original implementation exactly.
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    return _BufferedSDPA.apply(q,k,v,seqlens,window,scale)


def _pack_group(q, k, v, chunks, window):
    """Pad independent local chunks into one SDPA batch; never join documents."""
    count, heads = len(chunks), q.shape[1]
    qlen = max(hi-lo for lo,hi,kl in chunks)
    klen = max(hi-kl for lo,hi,kl in chunks)
    width = max(q.shape[-1], v.shape[-1])
    qp = q.new_zeros(count,heads,qlen,width)
    kp = k.new_zeros(count,heads,klen,width)
    vp = v.new_zeros(count,heads,klen,width)
    for i,(lo,hi,kl) in enumerate(chunks):
        qp[i,:,:hi-lo,:q.shape[-1]].copy_(q[lo:hi].transpose(0,1))
        kp[i,:,:hi-kl,:k.shape[-1]].copy_(k[kl:hi].transpose(0,1))
        vp[i,:,:hi-kl,:v.shape[-1]].copy_(v[kl:hi].transpose(0,1))
    positions = torch.tensor(chunks,device=q.device)
    lo,hi,kl = (positions[:,i,None,None] for i in range(3))
    qi = lo + torch.arange(qlen,device=q.device)[None,:,None]
    ki = kl + torch.arange(klen,device=q.device)[None,None,:]
    allowed = (ki<=qi) & (ki>=qi-window) & (qi<hi) & (ki<hi)
    return qp,kp,vp,allowed[:,None]


def _gather_group(q,k,v,chunks,window,with_query_layout=False):
    """Batch each Q/K/V gather instead of zeroing/copying every chunk separately."""
    count,heads = len(chunks),q.shape[1]
    qlen=max(hi-lo for lo,hi,kl in chunks)
    klen=max(hi-kl for lo,hi,kl in chunks)
    positions=torch.tensor(chunks,device=q.device)
    lo,hi,kl=(positions[:,i,None] for i in range(3))
    qi=lo+torch.arange(qlen,device=q.device)[None,:]
    ki=kl+torch.arange(klen,device=q.device)[None,:]
    # Padding gathers a valid value from its own document, then is excluded by
    # the mask. This avoids out-of-bounds access and per-chunk zero filling.
    qindices=torch.minimum(qi,hi-1).flatten()
    kindices=torch.minimum(ki,hi-1).flatten()
    width=max(q.shape[-1],v.shape[-1])
    packed=[]
    for source,indices,length in ((q,qindices,qlen),(k,kindices,klen),(v,kindices,klen)):
        value=source.index_select(0,indices).view(count,length,heads,source.shape[-1]).transpose(1,2)
        packed.append(F.pad(value,(0,width-source.shape[-1])))
    allowed=(ki[:,None,:]<=qi[:,:,None]) & (ki[:,None,:]>=qi[:,:,None]-window)
    allowed=allowed & (qi[:,:,None]<hi[:,:,None]) & (ki[:,None,:]<hi[:,:,None])
    if with_query_layout:
        # Query chunks partition this contiguous token span. Map its rows into
        # padded batched outputs without per-chunk copies or a dynamic nonzero.
        rows=torch.arange(chunks[0][0],chunks[-1][1],device=q.device)
        chunk=torch.bucketize(rows,positions[:,1].contiguous(),right=True)
        output_rows=chunk*qlen+rows-positions[chunk,0]
        return *packed,allowed[:,None],qindices,qi<hi,output_rows
    return *packed,allowed[:,None]


class _GroupedSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale,group_size,gather):
        chunks = chunks_for(seqlens.detach().cpu().tolist(),window)
        ctx.groups = [chunks[i:i+group_size] for i in range(0,len(chunks),group_size)]
        ctx.window,ctx.scale = window,scale
        ctx.packer = _gather_group if gather else _pack_group
        ctx.save_for_backward(q,k,v)
        result = v.new_empty(v.shape)
        for group in ctx.groups:
            qp,kp,vp,mask = ctx.packer(q,k,v,group,window)
            output = F.scaled_dot_product_attention(qp,kp,vp,attn_mask=mask,dropout_p=0.,scale=scale)
            for i,(lo,hi,kl) in enumerate(group):
                result[lo:hi].copy_(output[i,:,:hi-lo,:v.shape[-1]].transpose(0,1))
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        q,k,v = ctx.saved_tensors
        dq,dk,dv = torch.zeros_like(q),torch.zeros_like(k),torch.zeros_like(v)
        for group in reversed(ctx.groups):
            qp,kp,vp,mask = ctx.packer(q,k,v,group,ctx.window)
            upstream = torch.zeros_like(qp)
            for i,(lo,hi,kl) in enumerate(group):
                upstream[i,:,:hi-lo,:v.shape[-1]].copy_(grad_output[lo:hi].transpose(0,1))
            with torch.enable_grad():
                qp.requires_grad_(); kp.requires_grad_(); vp.requires_grad_()
                output = F.scaled_dot_product_attention(qp,kp,vp,attn_mask=mask,dropout_p=0.,scale=ctx.scale)
                gq,gk,gv = torch.autograd.grad(output,(qp,kp,vp),upstream)
            for i in reversed(range(len(group))):
                lo,hi,kl = group[i]
                dq[lo:hi].copy_(gq[i,:,:hi-lo,:q.shape[-1]].transpose(0,1))
                dk[kl:hi].add_(gk[i,:,:hi-kl,:k.shape[-1]].transpose(0,1))
                dv[kl:hi].add_(gv[i,:,:hi-kl,:v.shape[-1]].transpose(0,1))
        return dq,dk,dv,None,None,None,None,None


def grouped_attention(q,k,v,seqlens,window,scale,group_size=8):
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    if group_size < 1:
        raise ValueError('Attention chunk group size must be positive')
    return _GroupedSDPA.apply(q,k,v,seqlens,window,scale,group_size,False)


def gathered_attention(q,k,v,seqlens,window,scale,group_size=8):
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    if group_size<1:
        raise ValueError('Attention chunk group size must be positive')
    return _GroupedSDPA.apply(q,k,v,seqlens,window,scale,group_size,True)


def gathered_attention32(q,k,v,seqlens,window,scale):
    return gathered_attention(q,k,v,seqlens,window,scale,group_size=32)


def _unpack_query(values,rows,width):
    # The scalar order is identical to the original per-chunk transpose/copy.
    flat=values.transpose(1,2).reshape(-1,values.shape[1],values.shape[-1])
    return flat.index_select(0,rows)[...,:width]


class _VectorizedGatheredSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        chunks=chunks_for(seqlens.detach().cpu().tolist(),window)
        group_size=getattr(ctx,'group_size',32)
        ctx.groups=[chunks[i:i+group_size] for i in range(0,len(chunks),group_size)]
        ctx.window,ctx.scale=window,scale
        ctx.save_for_backward(q,k,v)
        result=v.new_empty(v.shape)
        for group in ctx.groups:
            qp,kp,vp,mask,_,_,rows=_gather_group(q,k,v,group,window,True)
            output=F.scaled_dot_product_attention(qp,kp,vp,attn_mask=mask,dropout_p=0.,scale=scale)
            result[group[0][0]:group[-1][1]].copy_(_unpack_query(output,rows,v.shape[-1]))
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        q,k,v=ctx.saved_tensors
        dq,dk,dv=torch.zeros_like(q),torch.zeros_like(k),torch.zeros_like(v)
        for group in reversed(ctx.groups):
            qp,kp,vp,mask,indices,valid,rows=_gather_group(q,k,v,group,ctx.window,True)
            count,heads,qlen,width=qp.shape
            upstream=grad_output.index_select(0,indices).view(count,qlen,heads,v.shape[-1]).transpose(1,2)
            upstream=F.pad(upstream,(0,width-v.shape[-1]))
            upstream.masked_fill_(~valid[:,None,:,None],0.)
            with torch.enable_grad():
                qp.requires_grad_();kp.requires_grad_();vp.requires_grad_()
                output=F.scaled_dot_product_attention(qp,kp,vp,attn_mask=mask,dropout_p=0.,scale=ctx.scale)
                gq,gk,gv=torch.autograd.grad(output,(qp,kp,vp),upstream)
            dq[group[0][0]:group[-1][1]].copy_(_unpack_query(gq,rows,q.shape[-1]))
            if getattr(ctx,'fused_accumulation',False):
                from ks3_research.attention_accumulation import ordered_add
                # Limit each ordered scan to32 chunks even when the SDPA batch
                # is larger. Reverse subgroups retain exact reverse add order.
                for start in reversed(range(0,len(group),32)):
                    subset=group[start:start+32]
                    bounds=torch.tensor([(kl,hi) for _,hi,kl in subset],device=q.device)
                    ordered_add(dk,gk[start:start+32],subset,bounds)
                    ordered_add(dv,gv[start:start+32],subset,bounds)
            else:
                # Preserve reverse order and dtype for overlapping K/V spans.
                for i in reversed(range(len(group))):
                    lo,hi,kl=group[i]
                    dk[kl:hi].add_(gk[i,:,:hi-kl,:k.shape[-1]].transpose(0,1))
                    dv[kl:hi].add_(gv[i,:,:hi-kl,:v.shape[-1]].transpose(0,1))
        return dq,dk,dv,None,None,None


def vectorized_gathered_attention(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    return _VectorizedGatheredSDPA.apply(q,k,v,seqlens,window,scale)


class _FusedAccumulationSDPA(_VectorizedGatheredSDPA):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        ctx.fused_accumulation=True
        return _VectorizedGatheredSDPA.forward(ctx,q,k,v,seqlens,window,scale)


def fused_accumulated_attention(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    return _FusedAccumulationSDPA.apply(q,k,v,seqlens,window,scale)


class _Fused64SDPA(_FusedAccumulationSDPA):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        ctx.group_size=64
        return _FusedAccumulationSDPA.forward(ctx,q,k,v,seqlens,window,scale)


def fused64_attention(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    return _Fused64SDPA.apply(q,k,v,seqlens,window,scale)


class BufferedSelfAttention(CausalSelfAttention):
    """KS3-only copy of the portable forward with one attention-call substitution.

    Keeping this adapter here leaves the immutable MLP/control source untouched.
    Wrapper tests compare all projection, auxiliary, gate, XSA and input gradients
    against CausalSelfAttention, including paired heads and unequal QK/V widths.
    """
    attention = staticmethod(buffered_attention)
    paired_seqlens = staticmethod(lambda seqlens: 2*seqlens)

    def forward(self, x, attn_args, qk_w, v_w, o_w, qkv_fp8=None):
        B, T = x.size(0), x.size(1)
        assert B == 1 and T % 16 == 0
        H = self.num_heads
        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        train_max_seq_len, yarn = attn_args.train_max_seq_len, attn_args.yarn
        assert yarn.head_dim == self.qk_dim
        assert not key_offset or self.qk_dim > Yarn.ROTARY_DIM
        qkv_weight = sa_lambdas[0] * torch.cat((qk_w, v_w)).type_as(x)
        qkv = F.linear(x, qkv_weight)
        qk, v = qkv.split((2*H*self.qk_dim, H*self.v_dim), dim=-1)
        qk = qk.view(B,T,2*H,self.qk_dim)
        v = v.view(B,T,H,self.v_dim)
        q, k = qk_norm_rope(qk[0], yarn.factor1[:T], yarn.factor2[:T], H,
                           Yarn.ROTARY_DIM, self.paired, key_offset)
        q, k = q[None], k[None]
        max_len = train_max_seq_len if self.training else self.val_max_seq_len
        if aux_v is not None:
            aux_v = aux_v.view(B,T,H,self.head_dim)[..., :self.v_dim]
        if not self.paired:
            if aux_v is not None:
                v = v + aux_v
        else:
            v = v.reshape(B,T*2,H//2,self.v_dim)
            if aux_v is not None:
                v = v + aux_v.reshape(v.shape)
            seqlens = self.paired_seqlens(seqlens)
            max_len = 2*max_len
        y = self.attention(q[0],k[0],v[0],seqlens,bm_size,yarn.attn_scale)
        y = y.view(B,T,H,self.v_dim)
        if attn_args.xsa_alpha is not None and not self.paired:
            dot = (y*v).sum(-1,keepdim=True)
            denom = v.square().sum(-1,keepdim=True).clamp_min(1e-8)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(B,T,H,1)
            y = y-alpha*(dot/denom)*v
        if attn_gate_w is not None:
            y = y*attn_gate_w.type_as(y).view(B,T,H,1)
        y = y.contiguous().view(B,T,H*self.v_dim)
        o_scale = sa_lambdas[1] if attn_args.o_gain is None else sa_lambdas[1]*attn_args.o_gain
        return F.linear(y,o_scale*o_w.type_as(y))


class GroupedSelfAttention(BufferedSelfAttention):
    attention = staticmethod(grouped_attention)


class GatheredSelfAttention(BufferedSelfAttention):
    attention = staticmethod(gathered_attention)


class Gathered32SelfAttention(BufferedSelfAttention):
    attention = staticmethod(gathered_attention32)


class VectorizedGatheredSelfAttention(BufferedSelfAttention):
    attention = staticmethod(vectorized_gathered_attention)


class FusedAccumulationSelfAttention(BufferedSelfAttention):
    attention = staticmethod(fused_accumulated_attention)


class Fused64SelfAttention(BufferedSelfAttention):
    attention = staticmethod(fused64_attention)
