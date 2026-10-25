"""Deploy the reviewed KS3-only dense-width prototype after learning trials.

The exclusive GPU lease covers deployment, correctness and the candidate.
No active discovery/verification source is changed. Common source is checked
against the immutable matched-control fence before and after deployment.
"""
import argparse
import fcntl
import importlib
from pathlib import Path
import subprocess
import time
import traceback

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import (
    ROOT, STATE, RESULTS, read, atomic, append, sha, now, stopped, assert_fence,
)

ALLOWED = frozenset((
    'wr_reproduction/ks3_feedforward.py',
    'wr_reproduction/matched_architecture.py',
    'wr_reproduction/check_ks3_feedforward_cpu.py',
    'wr_reproduction/check_ks3_feedforward.py',
    'wr_reproduction/matched_controller.py',
    'test_wr_matched_study.py',
))
PROTOTYPE = RESULTS / 'dense-width1536-prototype.json'
PROGRESS = STATE / 'capacity-study.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_dense1536.json'


def learning_ready(value):
    for item in value.get('items', []):
        if item['status'] in ('error', 'incomplete', 'stopped'):
            raise RuntimeError('Learning ablation needs diagnosis: ' + item.get('run_id', item['recipe']))
    return (value.get('status') == 'complete_choose_next_hypothesis'
            and bool(value.get('items'))
            and all(item['status'] in ('complete', 'pruned') for item in value['items']))


def verify_prototype(prototype, deployed=False):
    changes = prototype['staged_sources']
    if {item['path'] for item in changes} != ALLOWED or len(changes) != len(ALLOWED):
        raise RuntimeError('Prototype contains an unapproved source path')
    patch = ROOT / prototype['patch']
    if sha(patch) != prototype['patch_sha256']:
        raise RuntimeError('Prototype patch changed')
    key = 'staged_sha256' if deployed else 'base_sha256'
    for item in changes:
        if sha(ROOT / item['path']) != item[key]:
            raise RuntimeError('Prototype source mismatch: ' + item['path'])
    if not prototype['cpu_audit']['passed'] or prototype['cpu_audit']['gpu_initialized']:
        raise RuntimeError('CPU prototype audit is missing or invalid')
    recipe = prototype['recipe']
    if (recipe['architecture'], recipe['charts'], recipe.get('dense_charts'), recipe['rank']) != ('ks3_all', 1024, 1536, 1):
        raise RuntimeError('Unexpected capacity recipe')
    assert_fence(recipe)
    return recipe


def deploy(prototype, progress):
    recipe = verify_prototype(prototype)
    paths = sorted(ALLOWED)
    if controller.git('status', '--porcelain', '--', *paths):
        raise RuntimeError('Prototype source paths have uncommitted changes')
    if RECIPE_PATH.exists():
        raise RuntimeError('Capacity recipe already exists; reconcile deployment first')
    patch = ROOT / prototype['patch']
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
    if stopped():
        return False
    subprocess.run(['git', 'apply', str(patch)], cwd=ROOT, check=True)
    atomic(RECIPE_PATH, recipe)
    verify_prototype(prototype, deployed=True)
    subprocess.run([str(controller.PYTHON), '-m', 'unittest', 'test_wr_matched_study', '-v'], cwd=ROOT, check=True)
    if stopped():
        return False
    paths.append(str(RECIPE_PATH.relative_to(ROOT)))
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    # --only preserves any unrelated staged work in the shared checkout.
    subprocess.run(['git', 'commit', '--only', '-m', 'Enable independent dense KS3 width and its GPU oracle', '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='dense_capacity_prototype_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=prototype['patch_sha256'],
           backend_hash=prototype['backend_hash'], decision='GPU output/gradient oracle must pass before capacity training'))
    importlib.reload(controller)
    return True


def execute_once(progress, label, recipe, kind, seed, unit):
    item = progress.setdefault(label, {})
    if item.get('run_id'):
        directory = STATE / 'runs' / item['run_id']
        run = read(directory / 'run.json')
        result = read(directory / 'result.json')
        if not result:
            raise RuntimeError('Interrupted capacity work needs reconciliation: ' + item['run_id'])
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
    return run, result


def work(unit):
    if stopped():
        return
    prototype = read(PROTOTYPE)
    progress = read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    progress.update(status='waiting_for_learning_queue', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)),
                                   code_commit=controller.git('rev-parse', 'HEAD')))
    verify_prototype(prototype, progress['deployed'])
    atomic(PROGRESS, progress)
    with (STATE / 'capacity-controller.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not learning_ready(read(STATE / 'pending-ablations.json', {})):
            if stopped():
                return
            time.sleep(1)
        with (ROOT / 'logs/wr-reproduction/gpu.lock').open('a') as gpu_lock:
            while True:
                if stopped():
                    return
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.5)
            groups = read(STATE / 'verification.json', dict(groups=[]))['groups']
            if any(g['status'] != 'complete' for g in groups):
                raise RuntimeError('Frozen verification is unfinished; keep discovery sources unchanged')
            if not progress['deployed'] and not deploy(prototype, progress):
                return
            recipe = verify_prototype(prototype, deployed=True)
            progress['status'] = 'gpu_oracle_then_dense_capacity_trial'
            atomic(PROGRESS, progress)
            _, result = execute_once(progress, 'correctness', recipe, 'correctness', 172, unit)
            if result['status'] != 'correctness_passed':
                raise RuntimeError('Dense1536 GPU oracle did not pass')
            if stopped():
                return
            run, result = execute_once(progress, 'discovery', recipe, 'discovery', 168, unit)
            if result['status'] not in ('complete', 'pruned'):
                raise RuntimeError('Capacity trial needs diagnosis: ' + result['run_id'])
            if result['qualifies']:
                controller.verification(run, result, unit)
                if stopped():
                    return
                if any(g['status'] != 'complete' for g in read(STATE / 'verification.json')['groups']):
                    raise RuntimeError('Capacity verification needs diagnosis before further research')
            progress.update(status='complete_choose_next_hypothesis', completed_utc=now())
            atomic(PROGRESS, progress)
            atomic(RESULTS / 'dense-capacity-study-results.json', progress)
            atomic(STATE / 'phase.json', dict(phase='dense_capacity_complete_continue_research', utc=now()))


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
