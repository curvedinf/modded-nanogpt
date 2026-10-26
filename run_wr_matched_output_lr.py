"""Deploy audited KS3 output-projection rate policy after the current run and verification."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import time
import traceback

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import ROOT, STATE, RESULTS, read, atomic, append, sha, now, stopped, assert_fence
from run_wr_matched_dense2816 import acquire, require_verification_complete

SOURCES = ('wr_reproduction/ks3_feedforward.py',)
PROTOTYPE = RESULTS / 'output-lr1-prototype.json'
PROGRESS = STATE / 'output-lr1-study.json'
OUTPUT = RESULTS / 'output-lr1-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_output_lr1.json'


def predecessor_ready(value):
    items = value.get('items', [])
    if value.get('status') in ('error', 'incomplete', 'stopped') or any(
            item['status'] in ('error', 'incomplete', 'stopped') for item in items):
        raise RuntimeError('Constant-spine prerequisite needs diagnosis')
    return (value.get('status') == 'complete_choose_next_hypothesis' and len(items) == 1
            and items[0]['status'] in ('complete', 'pruned'))


def audit_policy(model, optimizer):
    if (not model.get('passed') or model.get('gpu_initialized') is not False
            or not model.get('kernel_arithmetic_and_model_class_unchanged')
            or len(model.get('unchanged_ast_sha256', {})) != 2
            or model.get('requested_output_lr_multiplier') != 1.
            or not optimizer.get('passed') or optimizer.get('gpu_initialized') is not False
            or optimizer.get('invalid_output_multipliers_rejected') != 7):
        raise RuntimeError('CPU model/default/validation audit incomplete')
    checks=optimizer.get('checks', [])
    architectures=('ks3_all','ks3_first_mlp','ks3_last_mlp','ks3_both_mlp')
    expected={(a,b) for a in architectures for b in (None,2.,1.)}
    if len(checks)!=12 or {(c['architecture'],c['requested_output_multiplier']) for c in checks}!=expected:
        raise RuntimeError('Output rate audit misses architecture/default cases')
    for c in checks:
        output=2. if c['requested_output_multiplier'] is None else c['requested_output_multiplier']
        keep=dict(ks3_all=(),ks3_first_mlp=(0,),ks3_last_mlp=(3,),ks3_both_mlp=(0,3))[c['architecture']]
        dense=[0. if i//2 in (*keep,5,6,7) else output if i%2 else 1. for i in range(16)]
        routed=[output if i%16>=8 else 1. for i in range(80)]
        if (c['dense_per_matrix_lr_multipliers']!=dense or c['routed_per_matrix_lr_multipliers']!=routed
                or c['effective_output_multiplier']!=output):
            raise RuntimeError('Unexpected per-matrix learning rates')
        if (not c['shared_optimizer_table_unchanged'] or not c['shared_parameter_configs_identical']
                or not c['only_target_matrix_rates_changed'] or not c['target_output_rate_and_padding_verified']
                or not c['original_scalar_staging_verified'] or c['scalar_schedule_points']!=3
                or c['effective_dense_multiplier']!=(1024/768)**.5
                or c['effective_routed_multiplier']!=(1024/768)**.5
                or c['geometry_multiplier']!=.125 or c['dense_geometry_multiplier']!=.125
                or c['retained_original_output_multiplier']!=(2. if keep else None)):
            raise RuntimeError('Output rate audit changes non-target optimizer policy')


def verify_prototype(value, deployed=False):
    changes=value['staged_sources']
    if tuple(c['path'] for c in changes)!=SOURCES:
        raise RuntimeError('Prototype exceeds one KS3 policy file')
    if sha(ROOT/value['patch'])!=value['patch_sha256']:
        raise RuntimeError('Audited patch changed')
    c=changes[0]
    if (sha(ROOT/c['path'])!=c['staged_sha256' if deployed else 'base_sha256']
            or sha(ROOT/c['staged_path'])!=c['staged_sha256']):
        raise RuntimeError('KS3 policy differs from audited source')
    for path, expected in value['unchanged_sources'].items():
        if sha(ROOT/path)!=expected:raise RuntimeError('Unchanged source differs: '+path)
    for key in ('cpu_model_auditor','cpu_optimizer_auditor','cpu_model_audit','cpu_optimizer_audit','baseline_recipe','scheduler'):
        record=value[key]
        if sha(ROOT/record['path'])!=record['sha256']:
            raise RuntimeError('Audited input changed: '+key)
    model=read(ROOT/value['cpu_model_audit']['path']);optimizer=read(ROOT/value['cpu_optimizer_audit']['path'])
    audit_policy(model,optimizer)
    if (model['staged_module_sha256']!=c['staged_sha256'] or optimizer['staged_module_sha256']!=c['staged_sha256']
            or model['base_module_sha256']!=c['base_sha256'] or model['recipe_sha256']!=value['recipe_sha256']
            or optimizer['original_optimizer_sha256']!=sha(ROOT/'wr_reproduction/pr393/upstream/track_1_short/optim/anvil.py')):
        raise RuntimeError('CPU source/recipe provenance differs')
    recipe=value['recipe'];expected=copy.deepcopy(read(ROOT/value['baseline_recipe']['path']))
    expected.update(name='matched-ks3-all-output-lr1',hypothesis=recipe['hypothesis'],ks3_output_projection_lr_multiplier=1.)
    if recipe!=expected or not value.get('kernel_arithmetic_and_architecture_unchanged'):
        raise RuntimeError('Recipe exceeds isolated KS3 output projection rate change')
    if assert_fence(recipe)['backend_hash']!=value['backend_hash']:
        raise RuntimeError('Shared backend changed')
    return recipe


def deploy(value, progress):
    recipe = verify_prototype(value)
    paths = [*SOURCES, str(RECIPE_PATH.relative_to(ROOT))]
    if controller.git('status', '--porcelain', '--', *paths) or RECIPE_PATH.exists():
        raise RuntimeError('Owned deployment paths need reconciliation')
    patch = ROOT / value['patch']
    stat = subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=ROOT, text=True)
    if sorted(line.split('\t')[-1] for line in stat.splitlines()) != sorted(SOURCES):
        raise RuntimeError('Patch writes outside KS3 policy scope')
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
    if stopped():
        return False
    require_verification_complete()
    subprocess.run(['git', 'apply', str(patch)], cwd=ROOT, check=True)
    atomic(RECIPE_PATH, recipe)
    verify_prototype(value, deployed=True)
    if stopped():
        return False
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Expose audited KS3-only output projection rate ablation', '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='ks3_output_lr1_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=value['patch_sha256'], backend_hash=value['backend_hash']))
    return True


def discovery(value, progress, recipe, unit):
    record = progress.get('discovery')
    if record:
        directory = STATE / 'runs' / record['run_id']
        run, result = read(directory / 'run.json'), read(directory / 'result.json')
        if not run or not result:
            raise RuntimeError('Interrupted discovery needs reconciliation')
        return run, result
    run = controller.create_run(recipe, 168, 'discovery', unit)
    run['scheduler'] = progress['scheduler']
    run['ks3_policy_audits'] = {key: value[key] for key in ('cpu_model_audit', 'cpu_optimizer_audit')}
    run['optimizer_policy_metadata'] = value['legacy_logging_note']
    atomic(Path(run['directory']) / 'run.json', run)
    atomic(RESULTS / (run['run_id'] + '-config.json'), run)
    progress['discovery'] = dict(run_id=run['run_id'], status='running')
    atomic(PROGRESS, progress)
    result = controller.execute(run)
    progress['discovery'].update(status=result['status'], result=result)
    atomic(PROGRESS, progress)
    return run, result


def work(unit):
    if stopped():
        return
    value, progress = read(PROTOTYPE), read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    if OUTPUT.exists():
        raise RuntimeError('Results exist; preserve prior evidence')
    verify_prototype(value, progress['deployed'])
    progress.update(status='waiting_for_constant_spine', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)), code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'output-lr1-controller.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not predecessor_ready(read(ROOT / value['prerequisite_manifest'], {})):
            if stopped():
                return
            time.sleep(1)
        with (STATE / 'ablation-queue.lock').open('a') as queue_lock:
            if not acquire(queue_lock):
                return
            with (ROOT / 'logs/wr-reproduction/gpu.lock').open('a') as gpu_lock:
                if not acquire(gpu_lock):
                    return
                require_verification_complete()
                if not progress['deployed'] and not deploy(value, progress):
                    return
                recipe = verify_prototype(value, deployed=True)
                progress.setdefault('source_hashes', controller.source_hashes())
                if progress['source_hashes'] != controller.source_hashes():
                    raise RuntimeError('Source changed after frozen deployment')
                progress['status'] = 'output_lr1_discovery'
                atomic(PROGRESS, progress)
                run, result = discovery(value, progress, recipe, unit)
                if result['status'] not in ('complete', 'pruned'):
                    raise RuntimeError('KS3 output projection rate discovery needs diagnosis')
                if result['qualifies']:
                    controller.verification(run, result, unit)
                    if stopped():
                        return
                    require_verification_complete()
                if stopped():
                    return
                progress.update(status='complete_choose_next_hypothesis', completed_utc=now())
                atomic(PROGRESS, progress)
                atomic(OUTPUT, progress)
                atomic(STATE / 'phase.json', dict(phase='output_lr1_complete_continue_research', utc=now()))


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--unit', required=True);args = parser.parse_args()
    try:
        work(args.unit)
    except BaseException:
        append(STATE / 'controller-errors.jsonl', dict(unit=args.unit, error=traceback.format_exc()))
        raise


if __name__ == '__main__':main()
