"""Deploy audited KS3 output initialization after its predecessor and verification."""
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
PROTOTYPE = RESULTS / 'output-init-gain01-prototype.json'
PROGRESS = STATE / 'output-init-gain01-study.json'
OUTPUT = RESULTS / 'output-init-gain01-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_output_init_gain01.json'


def predecessor_ready(value):
    if value.get('status') in ('error', 'incomplete', 'stopped'):
        raise RuntimeError('Rank-two width768 prerequisite needs diagnosis')
    result = value.get('discovery', {}).get('result', {})
    if result.get('status') in ('error', 'incomplete', 'stopped'):
        raise RuntimeError('Rank-two width768 discovery needs diagnosis')
    return (value.get('status') == 'complete_choose_next_hypothesis'
            and result.get('status') in ('complete', 'pruned'))


def audit_policy(model, mathematical):
    expected = {(a, .1) for a in ('ks3_all', 'ks3_first_mlp', 'ks3_last_mlp', 'ks3_both_mlp')} | {('ks3_all', 0.)}
    checks = model.get('checks', [])
    if (not model.get('passed') or model.get('gpu_initialized') is not False or len(checks) != 5
            or {(c['architecture'], c['gain']) for c in checks} != expected):
        raise RuntimeError('CPU initialization audit misses architecture/default cases')
    for c in checks:
        if (not c.get('passed') or c.get('gpu_initialized') is not False
                or not all(c.get(key) for key in ('shared_initialization_identical', 'global_rng_preserved',
                          'input_projections_identical', 'retained_mlps_and_padding_unchanged',
                          'exact_isolated_normal_draws_verified', 'original_bank_labels_shapes_and_bf16_cast_preserved',
                          'geometry_initialization_unchanged'))
                or c.get('invalid_gains_rejected') != 7
                or not c.get('scope', {}).get('only_initialization_validation_and_reporting_changed')):
            raise RuntimeError('Non-target initialization or source scope changed')
        policy = c['initialization_policy']
        if c['gain'] == 0.:
            if policy != dict(distribution='original-zero', gain=0., seed=None):
                raise RuntimeError('Default initialization regressed')
        else:
            keep = dict(ks3_all=(), ks3_first_mlp=(0,), ks3_last_mlp=(3,), ks3_both_mlp=(0,3))[c['architecture']]
            if (policy.get('gain') != .1 or policy.get('distribution') != 'normal'
                    or policy.get('seed') != 169 or policy.get('routed_output_matrices') != 40
                    or policy.get('dense_output_slots') != [n for n in range(5) if n not in keep]
                    or policy.get('routed_fan_in') != 1024 or policy.get('dense_fan_in') != 1024
                    or policy.get('routed_std') != .1/32 or policy.get('dense_std') != .1/32):
                raise RuntimeError('Candidate initialization policy differs from the requested distribution')
    rows = mathematical.get('checks', [])
    if (not mathematical.get('passed') or mathematical.get('gpu_initialized') is not False
            or len(rows) != 2 or {x['gain'] for x in rows} != {0., .1}):
        raise RuntimeError('Independent mathematical gradient audit incomplete')
    for row in rows:
        if (not row.get('analytical_gradients_passed') or not row.get('initial_warp_gradients_zero')
                or not row.get('output_projection_gradient_norm', 0.) > 0.):
            raise RuntimeError('Independent mathematical gradient audit incomplete')
        for key in ('input_gradient_norm', 'input_projection_gradient_norm', 'spine_coefficient_gradient_norm'):
            if not (row[key] == 0. if row['gain'] == 0. else row[key] > 0.):
                raise RuntimeError('First-update gradient-flow check failed')


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if tuple(c['path'] for c in changes) != SOURCES:
        raise RuntimeError('Prototype exceeds the single KS3 initialization file')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Audited patch changed')
    for c in changes:
        if (sha(ROOT / c['path']) != c['staged_sha256' if deployed else 'base_sha256']
                or sha(ROOT / c['staged_path']) != c['staged_sha256']):
            raise RuntimeError('KS3 initialization differs from audited source')
    for path, expected in value['unchanged_sources'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Unchanged source differs: ' + path)
    for key in ('cpu_auditor', 'cpu_model_audit', 'cpu_math_audit', 'baseline_recipe', 'scheduler', 'control_set'):
        record = value[key]
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('Audited input changed: ' + key)
    model = read(ROOT / value['cpu_model_audit']['path'])
    mathematical = read(ROOT / value['cpu_math_audit']['path'])
    audit_policy(model, mathematical)
    for record in model['case_reports']:
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('CPU initialization case report changed')
    for check in model['checks']:
        scope = check['scope']
        if (scope['base_source_sha256'] != changes[0]['base_sha256']
                or scope['staged_source_sha256'] != changes[0]['staged_sha256']
                or check['recipe_sha256'] != value['recipe_sha256']
                or check['cpu_auditor_sha256'] != value['cpu_auditor']['sha256']):
            raise RuntimeError('CPU source/recipe provenance differs')
    recipe = value['recipe']
    expected = copy.deepcopy(read(ROOT / value['baseline_recipe']['path']))
    expected.update(name='matched-ks3-all-output-init-gain01', hypothesis=recipe['hypothesis'],
                    ks3_output_projection_init_gain=.1)
    if recipe != expected or not value.get('kernel_arithmetic_and_optimizer_unchanged'):
        raise RuntimeError('Recipe exceeds isolated KS3 output initialization')
    gates = value['existing_numerical_gates']
    if {g['branch'] for g in gates} != {'dense', 'routed'}:
        raise RuntimeError('Both unchanged KS3 branches need numerical evidence')
    for record in gates:
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('Established numerical evidence changed')
        previous = read(ROOT / record['path'])
        if previous.get('status') != 'correctness_passed' or not previous.get('correctness', {}).get('passed'):
            raise RuntimeError('Unchanged KS3 operators lack their established numerical gate')
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
        raise RuntimeError('Patch writes outside KS3 initialization scope')
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
    subprocess.run(['git', 'commit', '--only', '-m', 'Add audited KS3-only nonzero output initialization', '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='ks3_output_init_gain01_deployed',
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
    run['ks3_policy_audits'] = {key: value[key] for key in ('cpu_model_audit', 'cpu_math_audit')}
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
    progress.update(status='waiting_for_rank2_width768', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)), code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'output-init-gain01-controller.lock').open('a') as own_lock:
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
                progress['status'] = 'output_init_gain01_discovery'
                atomic(PROGRESS, progress)
                run, result = discovery(value, progress, recipe, unit)
                if result['status'] not in ('complete', 'pruned'):
                    raise RuntimeError('KS3 output initialization discovery needs diagnosis')
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
                atomic(STATE / 'phase.json', dict(phase='output_init_gain01_complete_continue_research', utc=now()))


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--unit', required=True);args = parser.parse_args()
    try:
        work(args.unit)
    except BaseException:
        append(STATE / 'controller-errors.jsonl', dict(unit=args.unit, error=traceback.format_exc()))
        raise


if __name__ == '__main__':main()
