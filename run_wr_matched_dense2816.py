"""Deploy the audited original-width dense KS3 candidate after its predecessors.

Only the two recorded KS3 width allowlists may change. The exclusive GPU lease
covers deployment, the independent output/gradient oracle, and discovery plus
any required frozen verification. Shared sources and controls remain fenced.
"""
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

ALLOWED = frozenset(('wr_reproduction/ks3_feedforward.py',
                     'wr_reproduction/check_ks3_feedforward.py'))
PROTOTYPE = RESULTS / 'dense-width2816-prototype.json'
PROGRESS = STATE / 'dense-width2816-study.json'
OUTPUT = RESULTS / 'dense-width2816-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_dense2816.json'


def predecessor_ready(value):
    for item in value.get('items', []):
        if item['status'] in ('error', 'incomplete', 'stopped'):
            raise RuntimeError('Predecessor needs diagnosis: ' + item.get('run_id', item['recipe']))
    return (value.get('status') == 'complete_choose_next_hypothesis'
            and len(value.get('items', [])) == 2
            and all(i['status'] in ('complete', 'pruned') for i in value['items']))


def require_verification_complete():
    groups = read(STATE / 'verification.json', dict(groups=[]))['groups']
    if any(g['status'] != 'complete' for g in groups):
        raise RuntimeError('Frozen three-seed verification is unfinished; keep sources unchanged')


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if len(changes) != len(ALLOWED) or {c['path'] for c in changes} != ALLOWED:
        raise RuntimeError('Prototype may change only the two KS3 width allowlists')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Prototype patch changed')
    key = 'staged_sha256' if deployed else 'base_sha256'
    for item in changes:
        if sha(ROOT / item['path']) != item[key]:
            raise RuntimeError('KS3 source differs from audited prototype: ' + item['path'])
    for key in ('cpu_optimizer_audit', 'cpu_model_audit'):
        record = value[key]
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('CPU audit changed: ' + key)
        audit = read(ROOT / record['path'])
        if not audit.get('passed') or audit.get('gpu_initialized') is not False:
            raise RuntimeError('CPU audit did not pass: ' + key)
    capacity = value['projection_capacity_comparison']
    if not (capacity['all_projection_shapes_and_counts_identical']
            and capacity['projection_initialization_identical']
            and value['kernel_arithmetic_unchanged']):
        raise RuntimeError('Original-width projection audit is incomplete')
    recipe = value['recipe']
    if (recipe['architecture'], recipe['charts'], recipe['dense_charts'], recipe['rank'],
        recipe.get('projection_lr_multiplier', 1.),
        recipe.get('dense_projection_lr_multiplier', 1.)) != ('ks3_all', 1024, 2816, 1, 1., 1.):
        raise RuntimeError('Unexpected original-width KS3 recipe')
    if assert_fence(recipe)['backend_hash'] != value['backend_hash']:
        raise RuntimeError('Prototype backend differs from the frozen controls')
    return recipe


def deploy(value, progress):
    recipe = verify_prototype(value)
    paths = sorted(ALLOWED) + [str(RECIPE_PATH.relative_to(ROOT))]
    if controller.git('status', '--porcelain', '--', *paths):
        raise RuntimeError('Owned deployment paths have uncommitted changes')
    if RECIPE_PATH.exists():
        raise RuntimeError('Original-width recipe already exists; reconcile deployment')
    patch = ROOT / value['patch']
    stat = subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=ROOT, text=True)
    destinations = [line.split('\t')[-1] for line in stat.splitlines()]
    if len(destinations) != len(ALLOWED) or set(destinations) != ALLOWED:
        raise RuntimeError('Patch writes outside the audited KS3 sources')
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
    subprocess.run(['git', 'commit', '--only', '-m',
                    'Enable original-width dense KS3 with an independent oracle gate',
                    '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'),
                    deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='dense_width2816_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=value['patch_sha256'],
           backend_hash=value['backend_hash'],
           cpu_model_audit_sha256=value['cpu_model_audit']['sha256'],
           cpu_optimizer_audit_sha256=value['cpu_optimizer_audit']['sha256'],
           decision='Independent GPU output/gradient oracle required before discovery'))
    return True


def execute_once(progress, label, recipe, kind, seed, unit):
    item = progress.setdefault(label, {})
    if item.get('run_id'):
        directory = STATE / 'runs' / item['run_id']
        run, result = read(directory / 'run.json'), read(directory / 'result.json')
        if not run or not result:
            raise RuntimeError('Interrupted original-width run needs reconciliation: ' + item['run_id'])
    else:
        run = controller.create_run(recipe, seed, kind, unit)
        run['scheduler'] = progress['scheduler']
        atomic(Path(run['directory']) / 'run.json', run)
        atomic(RESULTS / (run['run_id'] + '-config.json'), run)
        item.update(run_id=run['run_id'], status='running')
        atomic(PROGRESS, progress)
        result = controller.execute(run)
    item.update(status=result['status'], result=result)
    atomic(PROGRESS, progress)
    if run['recipe'] != recipe or run['source_hashes'] != controller.source_hashes():
        raise RuntimeError('Original-width run differs from frozen recipe/source')
    return run, result


def require_oracle(result):
    if result.get('status') != 'correctness_passed' or not result.get('correctness', {}).get('passed'):
        raise RuntimeError('Independent dense2816 GPU oracle did not pass')


def acquire(lock):
    while not stopped():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            time.sleep(.5)
    return False


def work(unit):
    if stopped():
        return
    value = read(PROTOTYPE)
    progress = read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    if OUTPUT.exists():
        raise RuntimeError('Original-width results already exist; preserve evidence')
    verify_prototype(value, progress['deployed'])
    progress.update(status='waiting_for_dense_rate_capacity_queue', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)),
                                   code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'dense-width2816-controller.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not predecessor_ready(read(Path(value['prerequisite_manifest']), {})):
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
                progress['status'] = 'gpu_oracle_then_original_width_trial'
                atomic(PROGRESS, progress)
                _, result = execute_once(progress, 'correctness', recipe, 'correctness', 172, unit)
                require_oracle(result)
                if stopped():
                    return
                run, result = execute_once(progress, 'discovery', recipe, 'discovery', 168, unit)
                if result['status'] not in ('complete', 'pruned'):
                    raise RuntimeError('Original-width discovery needs diagnosis')
                if result['qualifies']:
                    controller.verification(run, result, unit)
                    if stopped():
                        return
                    require_verification_complete()
                progress.update(status='complete_choose_next_hypothesis', completed_utc=now())
                atomic(PROGRESS, progress)
                atomic(OUTPUT, progress)
                atomic(STATE / 'phase.json', dict(phase='original_width_complete_continue_research', utc=now()))


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
