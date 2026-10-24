import json
from pathlib import Path
from ks3_research.planning import stages_for

def load_config(path):
    c=json.loads(Path(path).read_text());stages_for(c)
    assert c['architecture'] in ('pr393-moe','pr393-moe-ks3')
    if c['architecture']=='pr393-moe-ks3':
        if c['target_val_loss']!=3.68:
            raise ValueError('Current MoE KS3 target is 3.68; stale 3.793633 recipes are forbidden')
    else:
        assert c['target_val_loss']==3.793633
    return c

def plan(c,pilot):
    b=c['schedule_base_steps']
    return dict(base_steps=b,total_steps=b+72,termination='full_validation_target_only',
                estimated_validation_seconds=120.)
