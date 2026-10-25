"""Isolated matched PR393 study contracts; no historical reference import."""
import json
import subprocess
from pathlib import Path
from wr_reproduction import research as base
from wr_reproduction.research import ROOT,atomic,append,sha,digest,read,now,events

STATE=ROOT/'logs/wr-reproduction/matched-study'
RESULTS=ROOT/'research_results/wr_reproduction/matched-study'
FENCE=ROOT/'configs/wr_reproduction/matched_backend.json'
BUDGET=900.0
SHARED=dict(shared_fp8_gemm_tuning='mi300-v1',shared_fp8_amax='atomic',
            shared_fp8_decode='scalar',shared_fp8_head_tuning='legacy',shared_fp8_pipeline_tuning='legacy',
            validation_window_policy='accepted-control',validation_precision='bf16',
            max_train_seconds=BUDGET,validation_setup_reserve_seconds=2.0,
            model_wide_hyperparameters='original-pr393')
STOP_FILES=[STATE/'STOP',RESULTS/'STOP']

def stopped():
    return any(p.exists() for p in STOP_FILES)

def common_paths():
    paths=list((ROOT/'wr_reproduction/pr393/upstream').rglob('*.py'))
    paths+=list((ROOT/'wr_reproduction/record360/upstream').rglob('*.py'))
    excluded=('ks3','check_','bench_','research_controller','matched_','geometry_diagnostics','profile_')
    paths += [p for p in (ROOT/'wr_reproduction').glob('*.py') if not p.name.startswith(excluded)]
    paths += [ROOT/'wr_reproduction/bench_updates.py', ROOT/'wr_reproduction/pr393/contract.json',
              ROOT/'wr_reproduction/source_manifest.json',ROOT/'configs/ks3/screening_policy.json',
              ROOT/'ks3_research/ops/screening.py']
    return sorted(set(p for p in paths if p.exists()))

def freeze():
    from wr_reproduction.contracts import audit
    audit('pr393');audit('record360')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in common_paths()}
    payload=dict(source_hashes=hashes,settings=SHARED,
                 model_wide_config_sha256=sha(ROOT/'wr_reproduction/pr393/upstream/track_1_short/config.py'))
    value=dict(utc=now(),backend_hash=digest(payload),**payload,
               winner_commit='f1366345fcdbde52d06711b33300e481db855c34',
               evidence='Winner mi300-v1 dispatch + original dense MLP autograd verified; all newer shared policies disabled')
    if FENCE.exists():
        assert read(FENCE)['backend_hash']==value['backend_hash'],'Existing fence is immutable'
    else:atomic(FENCE,value)
    return value

def assert_fence(recipe):
    from wr_reproduction.contracts import audit
    audit('pr393');audit('record360')
    fence=read(FENCE)
    if not fence:raise RuntimeError('Backend is not frozen')
    for p,h in fence['source_hashes'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Shared backend fence violation: '+p)
    for k,v in fence['settings'].items():
        if recipe.get(k)!=v:raise RuntimeError('Shared setting fence violation: '+k)
    return fence

class Metadata(base.Metadata):
    def __init__(self,data=None):
        super().__init__(data)
        fence=read(FENCE)
        self.common=dict(matched_backend_hash=fence['backend_hash'],
                         model_wide_hyperparameters=fence['settings']['model_wide_hyperparameters'])
    def training(self,step):return dict(super().training(step),**self.common)
    def validation(self,step):return dict(super().validation(step),**self.common)

def prepare():
    STATE.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    winner=read(ROOT/'configs/wr_reproduction/research_bf16_shared_fp8.json')
    controls=dict(name='matched-mlp-original',architecture='mlp',**SHARED,
                  hypothesis='Measure original PR393 MLP with the identical validated shared mi300-v1 backend, raw full validation and nominal900-second clock.')
    audit=dict(utc=now(),original_commit='426ab710eb0ff4d283d55ecc3391fb5e884e8d61',
               winner_commit='f1366345fcdbde52d06711b33300e481db855c34',
               winner_recipe_sha256=sha(ROOT/'configs/wr_reproduction/research_bf16_shared_fp8.json'),
               model_wide_differences={},collapsed_duplicate_arms=True,
               decision='One frozen original-PR393 matched MLP recipe; independent seeds169,170,171',
               excluded_ks3_settings={k:winner[k] for k in ('precision','charts','rank','geometry','training_gs32','bf16_kernel_mode')},
               evidence='Winner imports immutable original config, schedule, optimizer and data; recipe has no model-wide overrides. KS3 geometry/projection hardware is architecture-specific.',
               old_results_not_architecture_advantage=True)
    atomic(RESULTS/'hyperparameter-audit.json',audit)
    atomic(ROOT/'configs/wr_reproduction/matched_mlp.json',controls)
    for architecture in ('ks3_all','ks3_first_mlp','ks3_last_mlp','ks3_both_mlp'):
        recipe={k:winner[k] for k in ('precision','charts','rank','group_size','nonlinearity','geometry',
                 'projection_accumulation','training_gs32','weight_packing','geometry_backward_rows','bf16_kernel_mode')}
        recipe.update(SHARED,architecture=architecture,name='matched-'+architecture.replace('_','-'),
                      hypothesis='Replace all actual feedforward sites with the verified1024/rank1 KS3 block; retain MLPs only at the explicitly selected transformer edge layers.')
        atomic(ROOT/('configs/wr_reproduction/matched_'+architecture+'.json'),recipe)
    return freeze()
