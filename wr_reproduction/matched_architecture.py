"""CPU-readable inventory of the immutable PR393 forward's feedforward sites."""
from pathlib import Path
import ast
import hashlib

SOURCE=Path(__file__).parent/'pr393/upstream/track_1_short/model/gpt.py'
DENSE={0:0,2:1,4:2,10:3,8:4}
MOE=(1,3,5,6,8)
RETAIN={'mlp':tuple(range(11)), 'ks3_all':(), 'ks3_first_mlp':(0,),
        'ks3_last_mlp':(10,), 'ks3_both_mlp':(0,10), 'ks3_experts':(0,2,4,8,10)}

def layer_map(architecture,features=1024,dense_features=None):
    dense_features=features if dense_features is None else dense_features
    retained=RETAIN[architecture]
    rows=[]
    for layer in range(11):
        branches=[]
        if layer in DENSE:
            kind='mlp' if layer in retained else 'ks3'
            h=2816 if kind=='mlp' else dense_features
            branches.append(dict(branch='parallel_dense' if layer==8 else 'dense',slot=DENSE[layer],
                                 type=kind,instances=1,projection_shape=[2,h,768],
                                 projection_parameters=2*h*768))
        if layer in MOE:
            kind='mlp' if architecture=='mlp' else 'ks3'
            h=1024 if kind=='mlp' else features
            branches.append(dict(branch='routed_experts',slot=MOE.index(layer),type=kind,
                                 instances=8,topk=1,projection_shape=[2,8,h,768],
                                 projection_parameters=2*8*h*768))
        rows.append(dict(layer=layer,feedforward=branches,removed_feedforward=layer in (7,9)))
    return rows

def audit_model(model,architecture):
    from track_1_short.model import gpt,moe
    assert gpt.NUM_LAYERS==11 and gpt.NO_MLP_LAYERS==(7,9)
    assert gpt.MLP_SLOT_OF_LAYER=={0:0,2:1,4:2,10:3}
    assert (gpt.PARALLEL_MLP_LAYER,gpt.PARALLEL_MLP_SLOT)==(8,4)
    assert moe.MOE_LAYERS==MOE and moe.MOE_EXPERTS==8 and moe.MOE_TOPK==1
    names=dict(model.named_parameters())
    assert tuple(model.moe_router.shape)==(5,8,768)
    if architecture=='mlp':
        assert not any('ks3' in n for n in names)
        assert tuple(model.mlp_bank.shape)==(8,2,2816,768)
        assert tuple(model.moe_bank.shape)==(80,1024,768)
    features=model.moe_bank.shape[1]
    rows=layer_map(architecture,features,model.mlp_bank.shape[2])
    if architecture not in ('mlp','ks3_experts'):
        model.assert_feedforward_inventory(rows)
    branches=[b for row in rows for b in row['feedforward']]
    return dict(architecture=architecture,layers=rows,dense_sites=5,moe_sites=5,
                routed_expert_instances=40,active_conventional_mlp_instances=sum(b['instances'] for b in branches if b['type']=='mlp'),
                total_parameters=sum(p.numel() for p in names.values()),
                parameter_inventory={n:dict(shape=list(p.shape),parameters=p.numel()) for n,p in names.items()},
                upstream_forward_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                forward_transform_sha256=getattr(model,'_matched_forward_sha256',None),
                coverage='Static forward call-site audit; real full-graph capture is required before training',
                excluded=['attention projections','head','embedding','router','MUDD gates'])
