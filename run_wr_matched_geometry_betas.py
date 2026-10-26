"""Deploy audited KS3 geometry policy after the current run and verification."""
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

SOURCES = ('wr_reproduction/ks3_optimizer.py', 'wr_reproduction/ks3_moe.py', 'wr_reproduction/ks3_feedforward.py')
PROTOTYPE = RESULTS / 'geometry-beta0999-prototype.json'
PROGRESS = STATE / 'geometry-beta0999-study.json'
OUTPUT = RESULTS / 'geometry-beta0999-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_geometry_beta0999.json'


def predecessor_ready(value):
    items = value.get('items', [])
    if value.get('status') in ('error', 'incomplete', 'stopped') or any(
            item['status'] in ('error', 'incomplete', 'stopped') for item in items):
        raise RuntimeError('Rank-two warp-prior prerequisite needs diagnosis')
    return (value.get('status') == 'complete_choose_next_hypothesis' and len(items) == 1
            and items[0]['status'] in ('complete', 'pruned'))


def audit_policy(model, optimizer):
    if (not model.get('passed') or model.get('gpu_initialized') is not False
            or len(model.get('unchanged_ast_sha256', {})) != 4
            or not optimizer.get('passed') or optimizer.get('gpu_initialized') is not False
            or not optimizer.get('default_parameter_tables_identical')
            or optimizer.get('invalid_geometry_betas_rejected_by_both_overlays') != 11):
        raise RuntimeError('CPU model/default/validation audit incomplete')
    equation = model.get('adam_equation_audit', {})
    if (not equation.get('passed') or len(equation.get('checks', [])) != 2
            or {x['beta2'] for x in equation['checks']} != {.99, .999}
            or any(not x['passed'] or x['updates'] != 12 for x in equation['checks'])):
        raise RuntimeError('Original Adam equation audit incomplete')
    checks = optimizer.get('checks', [])
    expected = {(a, b) for a in ('ks3_all', 'ks3_first_mlp', 'ks3_last_mlp', 'ks3_both_mlp')
                for b in (None, (.9, .99), (.9, .999))}
    actual = {(c['architecture'], None if c['requested_geometry_betas'] is None
               else tuple(c['requested_geometry_betas'])) for c in checks}
    if len(checks) != 12 or actual != expected:
        raise RuntimeError('Geometry beta audit misses architecture/default cases')
    for c in checks:
        betas = c['requested_geometry_betas'] or [.9, .99]
        if (c['effective_dense_geometry_betas'] != betas or c['effective_routed_geometry_betas'] != betas
                or not c['shared_optimizer_table_unchanged'] or not c['output_2x_and_padding_preserved']
                or not c['retained_original_decay_preserved']
                or c['effective_dense_multiplier'] != (1024/768)**.5
                or c['effective_routed_multiplier'] != (1024/768)**.5
                or c['geometry_multiplier'] != .125 or c['dense_geometry_multiplier'] != .125
                or c['effective_dense_decay_multiplier'] != 1. or c['effective_routed_decay_multiplier'] != 1.):
            raise RuntimeError('Geometry beta audit changes non-target optimizer policy')


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if tuple(c['path'] for c in changes) != SOURCES:
        raise RuntimeError('Prototype exceeds three KS3 policy files')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Audited patch changed')
    for c in changes:
        if (sha(ROOT / c['path']) != c['staged_sha256' if deployed else 'base_sha256']
                or sha(ROOT / c['staged_path']) != c['staged_sha256']):
            raise RuntimeError('KS3 policy differs from audited source')
    for path, expected in value['unchanged_sources'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Unchanged source differs: ' + path)
    for key in ('cpu_model_auditor', 'cpu_optimizer_auditor', 'cpu_model_audit', 'cpu_optimizer_audit', 'baseline_recipe', 'scheduler'):
        record = value[key]
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('Audited input changed: ' + key)
    model = read(ROOT / value['cpu_model_audit']['path'])
    optimizer = read(ROOT / value['cpu_optimizer_audit']['path'])
    audit_policy(model, optimizer)
    staged = {Path(c['path']).name: c['staged_sha256'] for c in changes}
    base = {Path(c['path']).name: c['base_sha256'] for c in changes}
    if (model['staged_source_hashes'] != staged or optimizer['staged_source_hashes'] != staged
            or model['base_source_hashes'] != base or model['recipe_sha256'] != value['recipe_sha256']):
        raise RuntimeError('CPU source/recipe provenance differs')
    recipe = value['recipe']
    expected = copy.deepcopy(read(ROOT / value['baseline_recipe']['path']))
    expected.update(name='matched-ks3-all-geometry-beta0999', hypothesis=recipe['hypothesis'])
    expected['geometry']['betas'] = [.9, .999]
    if recipe != expected or not value.get('kernel_arithmetic_and_architecture_unchanged'):
        raise RuntimeError('Recipe exceeds isolated KS3 geometry beta2 change')
    if assert_fence(recipe)['backend_hash'] != value['backend_hash']:
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
    subprocess.run(['git', 'commit', '--only', '-m', 'Expose audited KS3-only geometry Adam beta2 ablation', '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='ks3_geometry_beta0999_deployed',
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
    progress.update(status='waiting_for_rank2_warp_prior', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)), code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'geometry-beta0999-controller.lock').open('a') as own_lock:
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
                progress['status'] = 'geometry_beta0999_discovery'
                atomic(PROGRESS, progress)
                run, result = discovery(value, progress, recipe, unit)
                if result['status'] not in ('complete', 'pruned'):
                    raise RuntimeError('KS3 geometry beta2 discovery needs diagnosis')
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
                atomic(STATE / 'phase.json', dict(phase='geometry_beta0999_complete_continue_research', utc=now()))


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--unit', required=True);args = parser.parse_args()
    try:
        work(args.unit)
    except BaseException:
        append(STATE / 'controller-errors.jsonl', dict(unit=args.unit, error=traceback.format_exc()))
        raise


if __name__ == '__main__':main()
