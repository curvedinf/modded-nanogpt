"""Deploy a KS3-only parameter-policy ablation after the current study queue."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import time
import traceback

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import (
    ROOT, STATE, RESULTS, read, atomic, append, sha, now, stopped, assert_fence,
)
from run_wr_matched_dense2816 import acquire, require_verification_complete

SOURCE = 'wr_reproduction/ks3_feedforward.py'
PROTOTYPE = RESULTS / 'ks3-projection-decay-prototype.json'
PROGRESS = STATE / 'ks3-projection-decay-study.json'
OUTPUT = RESULTS / 'ks3-projection-decay-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_projection_wd050.json'


def predecessor_ready(value):
    items = value.get('items', [])
    if value.get('status') in ('error', 'incomplete', 'stopped') or any(
            item['status'] in ('error', 'incomplete', 'stopped') for item in items):
        raise RuntimeError('Dense768 prerequisite needs diagnosis')
    return (value.get('status') == 'complete_choose_next_hypothesis' and len(items) == 1
            and items[0]['status'] in ('complete', 'pruned'))


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if len(changes) != 1 or changes[0]['path'] != SOURCE:
        raise RuntimeError('Decay override exceeds KS3 parameter-policy scope')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Audited patch changed')
    if sha(ROOT / SOURCE) != changes[0]['staged_sha256' if deployed else 'base_sha256']:
        raise RuntimeError('KS3 source differs from audited prototype')
    for path, expected in value['unchanged_sources'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Unchanged dependency differs: ' + path)
    for key in ('cpu_auditor', 'cpu_optimizer_audit', 'cpu_model_audit', 'baseline_recipe'):
        record = value[key]
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('Audited input changed: ' + key)
    audit = read(ROOT / value['cpu_optimizer_audit']['path'])
    model = read(ROOT / value['cpu_model_audit']['path'])
    if (not audit.get('passed') or audit.get('gpu_initialized') is not False
            or not model.get('passed') or model.get('gpu_initialized') is not False
            or len(audit.get('checks', [])) != 12
            or audit.get('invalid_decay_multipliers_rejected') != 3
            or len(model.get('unchanged_ast_sha256', {})) != 4
            or model.get('base_source_sha256') != changes[0]['base_sha256']
            or model.get('staged_source_sha256') != changes[0]['staged_sha256']):
        raise RuntimeError('CPU architecture or optimizer audit incomplete')
    if {c['architecture'] for c in audit['checks']} != {'ks3_all', 'ks3_first_mlp', 'ks3_last_mlp', 'ks3_both_mlp'}:
        raise RuntimeError('Missing hybrid optimizer audit')
    for check in audit['checks']:
        expected = check['requested_projection_decay_multiplier']
        expected = 1. if expected is None else expected
        if (check['effective_dense_decay_multiplier'] != expected
                or check['effective_routed_decay_multiplier'] != expected
                or not check['retained_original_decay_preserved']
                or not check['shared_optimizer_table_unchanged']):
            raise RuntimeError('Decay audit changed non-KS3 parameter policy')
    baseline = read(ROOT / value['baseline_recipe']['path'])
    recipe = value['recipe']
    without = lambda r: {k: v for k, v in r.items()
                          if k not in ('name', 'hypothesis', 'ks3_projection_weight_decay_multiplier')}
    if without(recipe) != without(baseline) or recipe.get('ks3_projection_weight_decay_multiplier') != .5:
        raise RuntimeError('Recipe may change only KS3 projection decay to0.5')
    if not value.get('kernel_arithmetic_and_architecture_unchanged'):
        raise RuntimeError('Numerical oracle reuse requires unchanged model arithmetic')
    if assert_fence(recipe)['backend_hash'] != value['backend_hash']:
        raise RuntimeError('Shared backend changed')
    return recipe


def deploy(value, progress):
    recipe = verify_prototype(value)
    paths = [SOURCE, str(RECIPE_PATH.relative_to(ROOT))]
    if controller.git('status', '--porcelain', '--', *paths) or RECIPE_PATH.exists():
        raise RuntimeError('Owned deployment paths need reconciliation')
    patch = ROOT / value['patch']
    stat = subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=ROOT, text=True)
    if [line.split('\t')[-1] for line in stat.splitlines()] != [SOURCE]:
        raise RuntimeError('Patch writes outside KS3 scope')
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
    subprocess.run(['git', 'commit', '--only', '-m', 'Add KS3-specific projection weight decay ablation', '--', *paths],
                   cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='ks3_projection_decay_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=value['patch_sha256'],
           backend_hash=value['backend_hash'], optimizer_audit_sha256=value['cpu_optimizer_audit']['sha256']))
    return True


def work(unit):
    if stopped():
        return
    value = read(PROTOTYPE)
    progress = read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    if OUTPUT.exists():
        raise RuntimeError('Decay results already exist; preserve evidence')
    verify_prototype(value, progress['deployed'])
    progress.update(status='waiting_for_dense768', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)),
                                   code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'ks3-projection-decay-controller.lock').open('a') as own_lock:
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
                    raise RuntimeError('Source changed during KS3 decay trial')
                progress['status'] = 'ks3_projection_decay_discovery'
                atomic(PROGRESS, progress)
                if progress.get('run_id'):
                    directory = STATE / 'runs' / progress['run_id']
                    run, result = read(directory / 'run.json'), read(directory / 'result.json')
                    if not run or not result:
                        raise RuntimeError('Interrupted decay discovery needs reconciliation')
                else:
                    run = controller.create_run(recipe, 168, 'discovery', unit)
                    run['scheduler'] = progress['scheduler']
                    atomic(Path(run['directory']) / 'run.json', run)
                    atomic(RESULTS / (run['run_id'] + '-config.json'), run)
                    progress['run_id'] = run['run_id']
                    atomic(PROGRESS, progress)
                    result = controller.execute(run)
                progress['result'] = result
                atomic(PROGRESS, progress)
                if result['status'] not in ('complete', 'pruned'):
                    raise RuntimeError('KS3 decay discovery needs diagnosis')
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
                atomic(STATE / 'phase.json', dict(phase='ks3_projection_decay_complete_continue_research', utc=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    try:
        work(args.unit)
    except BaseException:
        append(STATE / 'controller-errors.jsonl', dict(unit=args.unit, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
