"""Audit immutable record sources and derive schedules from their own Python modules."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
VERSIONS=('pr393','record360')

def audit(version):
    if version not in VERSIONS:raise ValueError(version)
    base=ROOT/version
    manifest=json.loads((base/'source_manifest.json').read_text())
    actual={str(p.relative_to(base/'upstream')):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (base/'upstream').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=manifest['files']:
        changed=sorted(k for k in actual.keys()|manifest['files'].keys() if actual.get(k)!=manifest['files'].get(k))
        raise RuntimeError(f'Immutable upstream source changed: {changed}')
    return manifest

def derive(version):
    manifest=audit(version)
    if os.environ.get('NUM_SCHEDULED_ITERATIONS') is not None:
        raise RuntimeError('Schedule override forbidden for exact record reproduction')
    # Run in an independent process: never import two versions under the same module name.
    if any(n=='track_1_short' or n.startswith('track_1_short.') for n in sys.modules):
        raise RuntimeError('Derive each record in a fresh interpreter')
    sys.path.insert(0,str(ROOT/version/'upstream'))
    config=importlib.import_module('track_1_short.config')
    schedule_module=importlib.import_module('track_1_short.schedule')
    hp=config.Hyperparameters()
    schedule=schedule_module.TrainingSchedule(config.TRAINING_STAGES,hp.num_scheduled_iterations,
        hp.num_extension_iterations,'cpu',config.LR_COOLDOWN_FRAC,config.SPLIT_EMBED_STAGE,config.WS_POST_YARN_EXT)
    steps=[]
    for i in range(schedule.total_steps):
        stage,_=schedule.lookup(i)
        if stage.batch_size%8:raise RuntimeError('Record global batch must divide into eight logical ranks')
        steps.append(dict(step=i,global_tokens=stage.batch_size,logical_rank_tokens=stage.batch_size//8,
                          microbatches=8,mtp=schedule.mtp_weights[i].tolist(),prefix=schedule.prefix_weights[i].item()))
    return dict(version=version,commit=manifest['commit'],source_files=len(manifest['files']),
                logical_world_size=8,physical_devices=1,total_steps=schedule.total_steps,
                total_training_tokens=sum(s['global_tokens'] for s in steps),
                val_tokens=hp.val_tokens,val_global_batch_size=hp.val_batch_size,
                boundaries=schedule.boundaries,split_embed_step=schedule.split_step,
                stages=[asdict(s) for s in config.TRAINING_STAGES],steps=steps,
                precision='Preserve original E4M3FN/E5M2 quantization values; native MI300 FNUZ GEMM via register interpretation with exact power-of-two scale correction',
                accumulation='Eight original logical rank batches sequentially; average rank gradients once per original step. Rank-local sampling, auxiliary statistics and FP8 state remain separate.',
                status='configuration_contract_runtime_validation_separate')

def main():
    p=argparse.ArgumentParser();p.add_argument('version',choices=VERSIONS);p.add_argument('--output',type=Path)
    a=p.parse_args();result=derive(a.version)
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','stages')},indent=2))

if __name__=='__main__':main()
