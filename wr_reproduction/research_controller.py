"""Detached, one-GPU WR trials and persistent immediate three-seed replication.

This controller runs only explicitly supplied ablations and their qualifying
replications. Model turns choose the next meaningful ablation, not polling.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from wr_reproduction.research import (ROOT, STATE, RESULTS, CONTROL_SECONDS, CONTROL_LOSS,
    Metadata, atomic, append, read, sha, digest, now, events, derive_saved_references,
    qualifies, queue_verification, verification_statistics)

PYTHON = ROOT / '.venv-mi300/bin/python'


def validate_recipe(recipe):
    required = dict(group_size=32, nonlinearity='fp16',
                    projection_accumulation='fp32', validation_precision='bf16', max_train_seconds=CONTROL_SECONDS)
    for k, v in required.items():
        if recipe.get(k) != v:
            raise ValueError('Unsupported or mismatched recipe field: ' + k)
    if recipe.get('rank') not in (1,2,4):raise ValueError('Unsupported KS3 rank')
    if recipe['rank']!=1 and (recipe.get('charts',0)*recipe['rank']!=1024 or recipe.get('precision')!='bf16' or recipe.get('bf16_kernel_mode')!='tuned'):
        raise ValueError('Higher rank requires1024 projection features and tuned BF16 training')
    features=recipe.get('charts',0)*recipe['rank']
    if features not in (512,768,1024,1280,1536):raise ValueError('Unsupported projection width')
    if features==1280 and recipe.get('precision')!='bf16':raise ValueError('1280-feature preparation covers BF16 only')
    if recipe.get('shared_fp8_gemm_tuning','legacy') not in ('legacy','mi300-v1','mi300-v2'):
        raise ValueError('Unknown shared FP8 tile policy')
    if recipe.get('fp8_native_gemm_tuning','legacy') not in ('legacy','mi300-v2'):
        raise ValueError('Unknown native FP8 tile policy')
    if recipe.get('fp8_native_gemm_tuning')=='mi300-v2' and (recipe.get('precision')!='fp8' or features!=1024 or recipe['rank']!=1 or not recipe.get('fp8_prepared_native_gather',False)):
        raise ValueError('Native-v2 tiles require1024/rank1 complete prepared FP8')
    if recipe.get('shared_fp8_pipeline_tuning','legacy') not in ('legacy','packed-v1'):
        raise ValueError('Unknown shared FP8 pipeline policy')
    if recipe.get('shared_fp8_pipeline_tuning')=='packed-v1' and (recipe.get('shared_fp8_decode')!='packed4-selected' or recipe.get('shared_fp8_gemm_tuning')!='mi300-v2'):
        raise ValueError('Packed shared pipelines require selected packed decoder and shared-v2 tiles')
    if recipe.get('shared_fp8_head_tuning','legacy') not in ('legacy','packed-v1'):
        raise ValueError('Unknown shared FP8 head policy')
    if recipe.get('shared_fp8_head_tuning')=='packed-v1' and (recipe.get('shared_fp8_decode')!='packed4-selected' or recipe.get('shared_fp8_gemm_tuning')!='mi300-v2'):
        raise ValueError('Packed head pipeline requires selected packed decoder and shared-v2 tiles')
    if recipe.get('shared_fp8_decode','scalar') not in ('scalar','packed4','packed4-selected'):
        raise ValueError('Unknown shared FP8 decode policy')
    if recipe.get('shared_fp8_amax','atomic') not in ('atomic','partial'):
        raise ValueError('Unknown shared FP8 amax policy')
    if recipe.get('validation_window_policy','training-stage') not in ('training-stage','accepted-control'):
        raise ValueError('Unknown validation window policy')
    if recipe.get('diagnostic_shared_fp8_expanded_sweep',False) and not recipe.get('diagnostic_shared_fp8_sweep',False):
        raise ValueError('Expanded FP8 sweep requires diagnostic sweep mode')
    if recipe.get('diagnostic_native_fp8_tile_sweep',False) and (recipe.get('precision')!='fp8' or recipe.get('charts')!=1024 or recipe.get('rank')!=1 or not recipe.get('fp8_prepared_native_gather',False)):
        raise ValueError('Native FP8 sweep requires complete native operand preparation at1024/rank1')
    if recipe.get('diagnostic_bf16_expanded_sweep',False) and (recipe.get('precision')!='bf16' or recipe.get('charts')!=1024 or recipe.get('rank')!=1):
        raise ValueError('Expanded BF16 sweep currently covers1024 rank-one features')
    if recipe.get('precision') not in ('bf16', 'fp8'):
        raise ValueError('Unsupported precision')
    if recipe.get('weight_packing','loop') not in ('loop','batched'):
        raise ValueError('Unknown weight packing kernel')
    if recipe.get('weight_packing','loop')=='batched' and recipe['precision']!='fp8':
        raise ValueError('Batched packing requires FP8 training')
    if not isinstance(recipe.get('fp8_prepared_native_weights',False),bool):
        raise ValueError('fp8_prepared_native_weights must be boolean')
    if recipe.get('fp8_prepared_native_weights',False) and (recipe['precision']!='fp8' or features not in (1024,1536) or recipe['rank']!=1 or recipe.get('weight_packing')!='batched'):
        raise ValueError('Prepared native weights require1024/1536 rank-one FP8 with batched packing')
    if not isinstance(recipe.get('fp8_prepared_native_activations',False),bool):
        raise ValueError('fp8_prepared_native_activations must be boolean')
    if recipe.get('fp8_prepared_native_activations',False) and not recipe.get('fp8_prepared_native_weights',False):
        raise ValueError('Prepared activations require prepared weights')
    if not isinstance(recipe.get('fp8_prepared_native_dual',False),bool):
        raise ValueError('fp8_prepared_native_dual must be boolean')
    if recipe.get('fp8_prepared_native_dual',False) and not recipe.get('fp8_prepared_native_activations',False):
        raise ValueError('Prepared dual layouts require native activations')
    if not isinstance(recipe.get('fp8_prepared_native_gather',False),bool):
        raise ValueError('fp8_prepared_native_gather must be boolean')
    if recipe.get('fp8_prepared_native_gather',False) and not recipe.get('fp8_prepared_native_dual',False):
        raise ValueError('Prepared gather requires native dual layouts')
    if recipe.get('geometry_backward_rows',64) not in (32,64,128):
        raise ValueError('Invalid geometry backward tile')
    if recipe.get('geometry_backward_rows',64)!=64 and recipe['precision']!='fp8':
        raise ValueError('Geometry tile ablation is FP8 training only')
    if recipe.get('bf16_kernel_mode','legacy') not in ('legacy','tuned'):
        raise ValueError('Unknown BF16 kernel mode')
    if recipe.get('bf16_kernel_mode','legacy')=='tuned' and (recipe['precision']!='bf16' or features not in (512,768,1024,1280,1536)):
        raise ValueError('Unsupported tuned BF16 width')
    if recipe.get('bf16_gemm_tuning','mi300-v1') not in ('mi300-v1','mi300-v2'):
        raise ValueError('Unknown BF16 projection tile policy')
    if recipe.get('bf16_gemm_tuning')=='mi300-v2' and (recipe['precision']!='bf16' or features!=1024 or recipe['rank'] not in (1,2) or recipe.get('bf16_kernel_mode')!='tuned'):
        raise ValueError('BF16-v2 projection tiles require tuned BF16 rank1/2 with1024 features')
    if not math.isfinite(recipe.get('projection_lr_multiplier',1.)) or recipe.get('projection_lr_multiplier',1.)<=0:
        raise ValueError('Projection LR multiplier must be finite and positive')
    if not isinstance(recipe.get('training_gs32',True),bool):
        raise ValueError('training_gs32 must be explicit boolean')
    if not recipe.get('training_gs32',True) and recipe['precision']!='fp8':
        raise ValueError('Native-only packing is an FP8 training ablation')
    geometry = recipe['geometry']
    if geometry.get('betas') != [.9, .99] or geometry.get('weight_decay') != 0:
        raise ValueError('Geometry Adam betas/weight decay have not been changed in code')
    if set(geometry) != {'lr_mul', 'betas', 'weight_decay', 'raw_scale', 'raw_shift', 'coefficients'}:
        raise ValueError('Unknown geometry setting')
    if len(geometry['coefficients']) != 6 or not all(math.isfinite(x) for x in
            [geometry['lr_mul'],geometry['raw_scale'],geometry['raw_shift'],*geometry['coefficients']]):
        raise ValueError('Invalid geometry')
    if geometry['lr_mul'] < 0 or not 0 < recipe['validation_setup_reserve_seconds'] < 10:
        raise ValueError('Invalid geometry LR or timer reserve')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = git('ls-files', 'wr_reproduction', 'ks3_research/vendor',
                'ks3_research/ops/screening.py', 'configs/ks3/screening_policy.json').splitlines()
    return {p: sha(ROOT / p) for p in paths if (ROOT / p).is_file()}


def update_status(run, status, **extra):
    row = dict(run_id=run['run_id'], status=status, updated_utc=now(), **extra)
    atomic(Path(run['directory']) / 'status.json', row)
    atomic(STATE / 'latest-run.json', dict(run=run, **row))
    append(STATE / 'registry.jsonl', dict(event='status', **row))


def notes(run, **fields):
    row = dict(run_id=run['run_id'], **fields)
    append(Path(run['directory']) / 'notes.jsonl', row)
    append(STATE / 'notes.jsonl', row)
    append(RESULTS / 'notes.jsonl', row)


def create_run(recipe, seed, kind, unit, metadata, frozen=None):
    validate_recipe(recipe)
    run_id = 'ks3-' + recipe['name'] + '-' + now().replace('-', '').replace(':', '').split('.')[0].replace('+0000', 'Z') + '-' + uuid.uuid4().hex[:6]
    directory = STATE / 'runs' / run_id
    directory.mkdir(parents=True)
    hashes = source_hashes()
    if frozen and hashes != frozen['source_hashes']:
        raise RuntimeError('Frozen verification code changed; restore exact sources before resuming seeds')
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)
    (directory / 'working-tree.patch').write_bytes(patch)
    dirty = bool(git('status', '--porcelain'))
    command = [str(PYTHON), '-m', 'wr_reproduction.train', '--version', 'pr393', '--ks3',
               '--ks3-precision', recipe['precision'], '--max-train-seconds', str(CONTROL_SECONDS),
               '--research-run', str(directory), '--out', str(directory / 'events.jsonl')]
    if kind=='timing':command += ['--timing-updates','100']
    if kind=='profile':command += ['--profile-updates','4']
    if kind=='kernel_check':
        checker='wr_reproduction.check_ks3_fp8_moe' if recipe['precision']=='fp8' else 'wr_reproduction.check_ks3_bf16_tiles'
        if recipe.get('bf16_kernel_mode')=='tuned':checker='wr_reproduction.check_ks3_bf16_block'
        command=[str(PYTHON),'-m',checker,'--out',str(directory/'events.jsonl'),
                 '--charts',str(recipe['charts'])]
        if recipe['rank']!=1:command += ['--rank',str(recipe['rank'])]
        if checker=='wr_reproduction.check_ks3_bf16_block':command += ['--tile-policy',recipe.get('bf16_gemm_tuning','mi300-v1')]
        if not recipe.get('training_gs32',True):command.append('--native-pack')
        if recipe.get('weight_packing','loop')=='batched':command.append('--batched-weight-pack')
        if recipe.get('fp8_prepared_native_weights',False):command.append('--prepared-native-weights')
        if recipe.get('fp8_prepared_native_activations',False):command.append('--prepared-native-activations')
        if recipe.get('fp8_prepared_native_dual',False):command.append('--prepared-native-dual')
        if recipe.get('fp8_prepared_native_gather',False):command.append('--prepared-native-gather')
        if recipe['precision']=='fp8':command += ['--native-tile-policy',recipe.get('fp8_native_gemm_tuning','legacy')]
        command += ['--geometry-backward-rows',str(recipe.get('geometry_backward_rows',64))]
        if recipe.get('diagnostic_geometry_sweep',False):command.append('--sweep-geometry-backward')
        if recipe.get('diagnostic_shared_fp8_sweep',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_shared_fp8_tiles','--out',str(directory/'events.jsonl')]
            if recipe.get('diagnostic_shared_fp8_expanded_sweep',False):command.append('--expanded')
        if recipe.get('diagnostic_shared_fp8_dispatch',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_shared_fp8_dispatch','--out',str(directory/'events.jsonl'),
                     '--tile-policy',recipe.get('shared_fp8_gemm_tuning','mi300-v1')]
        if recipe.get('diagnostic_shared_fp8_amax',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_shared_fp8_amax','--out',str(directory/'events.jsonl'),
                     '--tile-policy',recipe.get('shared_fp8_gemm_tuning','mi300-v2')]
        if recipe.get('diagnostic_packed_shared_dispatch',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_packed_shared_dispatch','--out',str(directory/'events.jsonl')]
        if recipe.get('diagnostic_packed_shared_pipelines',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_packed_shared_pipelines','--out',str(directory/'events.jsonl')]
        if recipe.get('diagnostic_packed_head_dispatch',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_packed_head_dispatch','--out',str(directory/'events.jsonl')]
        if recipe.get('diagnostic_packed_head_tiles',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_packed_head_tiles','--out',str(directory/'events.jsonl')]
        if recipe.get('diagnostic_shared_fp8_decode',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_packed_fp8_decode','--out',str(directory/'events.jsonl')]
        if recipe.get('diagnostic_native_fp8_tile_sweep',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_native_fp8_tiles','--out',str(directory/'events.jsonl')]
        if recipe.get('diagnostic_bf16_expanded_sweep',False):
            command=[str(PYTHON),'-m','wr_reproduction.check_ks3_bf16_expanded','--out',str(directory/'events.jsonl'),
                     '--charts',str(recipe['charts'])]
    run = dict(run_id=run_id, directory=str(directory), created_utc=now(), kind=kind,
               seed=seed, recipe=recipe, continuous_from_initialization=kind in ('discovery','verification'),
               code_commit=frozen['code_commit'] if frozen else git('rev-parse', 'HEAD'),
               launch_head_commit=git('rev-parse', 'HEAD'), working_tree_clean=not dirty,
               reproducibility='frozen-source-hashes' if frozen else ('clean-code-commit' if not dirty else 'code-commit-plus-recorded-patch-and-source-hashes'),
               patch_sha256=sha(directory / 'working-tree.patch'), source_hashes=hashes,
               command=command, unit=unit, validation_data=metadata.data,
               expected_evaluation_signature=metadata.validation(595),
               control=dict(loss=CONTROL_LOSS, train_seconds=CONTROL_SECONDS),
               comparative_pruning_exempt=kind=='verification',
               verification_exception='Explicit user-authorized three fresh seeds; no comparative pruning' if kind=='verification' else None)
    atomic(directory / 'config.json', recipe)
    atomic(directory / 'run.json', run)
    atomic(RESULTS / (run_id + '-config.json'), run)
    append(STATE / 'registry.jsonl', dict(event='registered', run_id=run_id, directory=str(directory),
                                       seed=seed, kind=kind, code_commit=run['code_commit']))
    update_status(run, 'registered')
    notes(run, hypothesis=recipe['hypothesis'], decision='Launch ' + kind,
          code_commit=run['code_commit'], seed=seed)
    return run


def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass


def execute(run):
    if (STATE / 'STOP').exists():
        raise RuntimeError('Explicit study STOP')
    directory = Path(run['directory'])
    env = dict(os.environ, TRAIN_SEED=str(run['seed']), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
    # These would silently alter the immutable recipe or precision.
    for key in ('NUM_SCHEDULED_ITERATIONS', 'MOE_GRAD_FP8', 'MOE_DW_PROMOTE', 'MOE_DX_PROMOTE'):
        if key in env:
            raise RuntimeError('Forbidden inherited configuration override: ' + key)
    with (directory / 'console.log').open('ab', buffering=0) as out:
        process = subprocess.Popen(run['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        active = dict(run_id=run['run_id'], controller_pid=os.getpid(), trainer_pid=process.pid,
                      trainer_pgid=process.pid, unit=run['unit'], updated_utc=now())
        atomic(STATE / 'active.json', active)
        update_status(run, 'running', **{k:v for k,v in active.items() if k not in ('run_id','updated_utc')})
        stopped = False
        try:
            while process.poll() is None:
                if (STATE / 'STOP').exists():
                    stopped = True
                    terminate(process)
                    break
                time.sleep(.5)  # machine watchdog, independent of manager/model turns
        finally:
            terminate(process)
    rows = events(directory / 'events.jsonl')
    vals = [r for r in rows if r.get('event') == 'validation']
    timing = next((r for r in rows if r.get('event') == 'timing_complete'),None)
    kernel_check = next((r for r in rows if r.get('event') == 'kernel_correctness'),None)
    geometry = next((r for r in rows if r.get('event') == 'terminal_geometry'),None)
    profile = next((r for r in rows if r.get('event') == 'profile_complete'),None)
    pruned = next((r for r in rows if r.get('event') in ('screening_abort','research_pruned')), None)
    validation = vals[-1] if vals else None
    status = 'stopped' if stopped else ('error' if process.returncode else ('pruned' if pruned else ('complete' if validation else 'incomplete')))
    if timing and not process.returncode and not stopped:status='timing_complete'
    if profile and not process.returncode and not stopped:status='profile_complete'
    if kernel_check and kernel_check['passed'] and not process.returncode and not stopped:status='correctness_passed'
    if validation and not math.isfinite(validation['loss']):
        validation = dict(validation, loss=None, error='nonfinite_validation_loss')
        status = 'error'
    if geometry and not geometry['all_finite']:
        status = 'error'
    qualifying = status == 'complete' and qualifies(validation, run['expected_evaluation_signature'])
    result = dict(run_id=run['run_id'], status=status, seed=run['seed'], code_commit=run['code_commit'],
                  recipe=run['recipe'], validation=validation, timing=timing,kernel_check=kernel_check,profile=profile,
                  terminal_geometry=geometry,qualifies=qualifying,
                  exit_code=process.returncode, pruning=pruned, completed_utc=now(),
                  events_sha256=sha(directory / 'events.jsonl') if rows else None,
                  previous_result_commit=git('log','-1','--format=%H','--','research_results/wr_reproduction/research'))
    atomic(directory / 'result.json', result)
    atomic(RESULTS / (run['run_id'] + '-result.json'), result)
    update_status(run, status, result=result)
    atomic(STATE / 'active.json', dict(status='idle', previous_run_id=run['run_id'], updated_utc=now()))
    notes(run, result=result, decision='Queue three fresh independent seeds immediately' if qualifying else 'Analyze before next meaningful ablation')
    append(RESULTS / 'milestones.jsonl', dict(milestone='trial_result', **result))
    return result


def verification_work(unit, metadata):
    path = STATE / 'verification.json'
    saved = read(path, dict(groups=[]))
    for group in saved['groups']:
        if group['status'] in ('complete', 'complete_with_errors'):
            continue
        group['status'] = 'running'
        atomic(path, saved)
        for item in group['seeds']:
            if (STATE / 'STOP').exists():
                return
            if item['status'] in ('complete', 'error', 'stopped', 'incomplete'):
                continue
            # Reconcile a completed run if a controller died between result and
            # verification-state persistence. Never silently duplicate a seed.
            if item.get('run_id'):
                prior = read(STATE / 'runs' / item['run_id'] / 'result.json')
                if prior:
                    item.update(status=prior['status'], result=prior)
                    atomic(path, saved)
                    continue
                raise RuntimeError('Interrupted verification needs explicit recovery of ' + item['run_id'])
            frozen = group['frozen_run']
            run = create_run(frozen['recipe'], item['seed'], 'verification', unit, metadata, frozen=frozen)
            item.update(run_id=run['run_id'], status='running')
            atomic(path, saved)
            result = execute(run)
            item.update(status=result['status'], result=result)
            atomic(path, saved)
        completed = [dict(seed=s['seed'], loss=s['result']['validation']['loss'],
                          train_seconds=s['result']['validation']['train_seconds'], run_id=s['run_id'])
                     for s in group['seeds'] if s['status']=='complete']
        if len(completed) == 3:
            group['status'] = 'complete'
            group['statistics'] = verification_statistics(completed)
        else:
            group['status'] = 'complete_with_errors'
            group['statistics'] = verification_statistics(completed) if completed else None
        group['completed_utc'] = now()
        atomic(path, saved)
        atomic(RESULTS / ('verification-' + group['recipe_id'] + '.json'), group)
        append(RESULTS / 'milestones.jsonl', dict(milestone='three_seed_verification', **group))


def work(request):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (STATE / 'STOP').exists():
            raise RuntimeError('Explicit study STOP')
        from wr_reproduction.contracts import audit
        audit('pr393');audit('record360')
        metadata = Metadata()
        derive_saved_references(metadata)
        if not (STATE / 'verification.json').exists():
            atomic(STATE / 'verification.json', dict(groups=[]))
        request = read(request)
        verification_work(request['unit'], metadata)
        if request.get('recipe') is not None and not (STATE / 'STOP').exists():
            run = create_run(request['recipe'], request['seed'], request.get('kind','discovery'), request['unit'], metadata)
            result = execute(run)
            if result['qualifies']:
                queue_verification(STATE, run, result)
                verification_work(request['unit'], metadata)


def launch(recipe, seed, timing=False,kernel_check=False,profile=False):
    if (STATE / 'STOP').exists():
        raise RuntimeError('Explicit study STOP')
    # Check the actual cross-process GPU lease, not merely a stale PID file.
    with (STATE / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if recipe:
        validate_recipe(read(recipe))
        if any(read(recipe).get(k,False) for k in ('diagnostic_shared_fp8_sweep','diagnostic_shared_fp8_dispatch','diagnostic_shared_fp8_amax','diagnostic_shared_fp8_decode','diagnostic_packed_head_tiles','diagnostic_packed_head_dispatch','diagnostic_packed_shared_pipelines','diagnostic_packed_shared_dispatch','diagnostic_bf16_expanded_sweep')) and not kernel_check:
            raise ValueError('Kernel sweep/check recipe requires diagnostic mode')
    unit = 'wr-ks3-trial-' + uuid.uuid4().hex[:12]
    request = STATE / 'requests' / (unit + '.json')
    atomic(request, dict(unit=unit, recipe=read(recipe) if recipe else None, seed=seed,
                        kind='kernel_check' if kernel_check else ('timing' if timing else ('profile' if profile else 'discovery')), utc=now()))
    command = ['systemd-run', '--unit=' + unit, '--working-directory=' + str(ROOT),
               '--property=KillMode=control-group', '--property=TimeoutStopSec=15',
               '--property=PartOf=wr-ks3-research.service',
               str(PYTHON), '-m', 'wr_reproduction.research_controller', 'work', str(request)]
    subprocess.run(command, check=True)
    print(json.dumps(dict(unit=unit, request=str(request))))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action', required=True)
    l = sub.add_parser('launch');l.add_argument('--recipe', type=Path);l.add_argument('--seed', type=int, default=168)
    modes=l.add_mutually_exclusive_group()
    modes.add_argument('--timing',action='store_true',help='100 original updates, skip validation; excluded from reference curves')
    modes.add_argument('--check-kernels',action='store_true',help='Isolated forward/backward correctness; not a training reference')
    modes.add_argument('--profile',action='store_true',help='Four original-prefix updates with GPU tracing, diagnostic only')
    w = sub.add_parser('work');w.add_argument('request', type=Path)
    args = p.parse_args()
    if args.action == 'launch':
        launch(args.recipe, args.seed,args.timing,args.check_kernels,args.profile)
    else:
        try:
            work(args.request)
        except BaseException:
            append(STATE / 'controller-errors.jsonl', dict(error=traceback.format_exc(), request=str(args.request)))
            raise


if __name__ == '__main__':
    main()
