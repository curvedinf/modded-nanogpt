"""No-GPU architecture, unchanged initialization and ANVIL-policy audit."""
import argparse
import gc
import hashlib
import json
import os
os.environ['HIP_VISIBLE_DEVICES']=''
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from wr_reproduction.backend import install
from wr_reproduction.matched_architecture import audit_model

def fingerprint(model):
    excluded={'mlp_bank','moe_bank','ks3_geometry','ks3_dense_geometry','retained_mlp_bank'}
    return {n:dict(shape=list(p.shape),sha256=hashlib.sha256(p.detach().float().numpy().tobytes()).hexdigest())
            for n,p in model.named_parameters() if n not in excluded}

def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(1);recipe=json.load(open(a.recipe));install('pr393')
    from track_1_short.model.gpt import GPT
    torch.manual_seed(169)
    baseline=GPT(50257,11,6,128,768,128,ngram_dim=768,world_size=8,device=torch.device('cpu'),rank=0)
    before=fingerprint(baseline)
    baseline_mlp=baseline.mlp_bank.detach().clone()
    baseline_router=baseline.moe_router.detach().clone()
    baseline_moe_sha=hashlib.sha256(baseline.moe_bank.detach().numpy().tobytes()).hexdigest()
    del baseline;gc.collect()
    from wr_reproduction.ks3_moe import install as install_moe
    from wr_reproduction.ks3_feedforward import install as install_dense
    cls=install_moe('bf16',recipe['geometry'],charts=recipe['charts'],bf16_kernel_mode='tuned',subspace_rank=recipe['rank'])
    cls=install_dense(cls,recipe)
    torch.manual_seed(169)
    model=cls(50257,11,6,128,768,128,ngram_dim=768,world_size=8,device=torch.device('cpu'),rank=0)
    assert before==fingerprint(model),'Nonfeedforward parameter initialization changed'
    assert torch.equal(model.moe_router,baseline_router)
    features=recipe.get('dense_charts',recipe['charts'])*recipe['rank']
    original_expert_width=recipe['charts']*recipe['rank']==1024
    if original_expert_width:
        assert hashlib.sha256(model.moe_bank.detach().numpy().tobytes()).hexdigest()==baseline_moe_sha,'Routed projection initialization changed'
    for slot in range(5):
        if slot in model.retained_slots:
            index=model.retained_slots.index(slot)
            assert torch.equal(model.retained_mlp_bank[index],baseline_mlp[slot])
        else:assert torch.equal(model.mlp_bank[slot],baseline_mlp[slot,:,:features,:])
    model.cast_matrix_weights_bf16()
    inventory=audit_model(model,recipe['architecture'])
    from track_1_short.optim.anvil import AnvilAndAdam
    optimizer=AnvilAndAdam.__new__(AnvilAndAdam)
    optimizer.rank=0;optimizer.world_size=1;optimizer.param_cfgs={}
    optimizer.param_table={'mlp_bank':dict(optim='anvil',comms='sharded')}
    optimizer.anvil_defaults=dict(lr=.023,momentum=.95,beta2=.9,weight_decay=2.25)
    optimizer._build_param_cfg(model.mlp_bank,'mlp_bank')
    cfg=optimizer.param_cfgs[model.mlp_bank]
    expected=[0. if i//2 in (*model.retained_slots,5,6,7) else 2. if i%2 else 1. for i in range(16)]
    assert cfg.per_matrix_lr_mul==expected
    if model.retained_slots:
        optimizer.param_table['retained_mlp_bank']=dict(optim='anvil',comms='sharded')
        optimizer._build_param_cfg(model.retained_mlp_bank,'retained_mlp_bank')
        cfg=optimizer.param_cfgs[model.retained_mlp_bank]
        assert cfg.label=='retained_mlp_bank'
        assert cfg.per_matrix_lr_mul==[0. if i//2>=len(model.retained_slots) else 2. if i%2 else 1. for i in range(8)]
        assert cfg.lr_mul==(2816/768)**.5
    assert not torch.cuda.is_initialized(),'CPU audit initialized GPU'
    result=dict(passed=True,gpu_initialized=False,architecture=recipe['architecture'],
                nonfeedforward_initialization_identical=True,router_identical=True,
                original_retained_initialization_identical=True,anvil_shape_and_output_lr_and_padding_verified=True,
                routed_projection_initialization_identical=original_expert_width,
                dense_features=features,routed_features=recipe['charts']*recipe['rank'],
                inventory=inventory)
    open(a.out,'w').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'}))

if __name__=='__main__':main()
