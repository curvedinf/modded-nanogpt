"""Serialize prepared KS3 ablations after the required architecture sequence.

This is a CPU scheduler. Training and verification use the existing fenced
matched controller; this entry point never imports GPU kernels or alters them.
"""
import argparse
import fcntl
from pathlib import Path
import time
import traceback

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import (
    ROOT, STATE, RESULTS, read, atomic, append, sha, now, stopped, assert_fence,
)

ARCHITECTURES = ('ks3_all', 'ks3_first_mlp', 'ks3_last_mlp', 'ks3_both_mlp')
SUCCESS = ('complete', 'pruned')


def initial_ready(state):
    arms = {item['architecture']: item for item in state.get('arms', [])}
    for item in arms.values():
        if item['status'] in ('error', 'incomplete', 'stopped'):
            raise RuntimeError('Initial architecture needs diagnosis: ' + item['run_id'])
    return all(arms.get(name, {}).get('status') in SUCCESS for name in ARCHITECTURES)


def checked_recipe(item):
    path = ROOT / item['recipe']
    if sha(path) != item['recipe_sha256']:
        raise RuntimeError('Prepared recipe changed: ' + str(path))
    recipe = read(path)
    assert_fence(recipe)
    if recipe['architecture'] == 'mlp':
        raise RuntimeError('This queue is restricted to KS3 ablations')
    return recipe


def dependency_ready(state):
    study_name = state.get('wait_for_study')
    if study_name:
        if Path(study_name).name != study_name or not study_name.endswith('.json'):
            raise RuntimeError('Study prerequisite requires a local JSON filename')
        study = read(STATE / study_name, {})
        if study.get('status') in ('error', 'incomplete', 'stopped'):
            raise RuntimeError('Study prerequisite needs diagnosis: ' + study_name)
        if study.get('status') != 'complete_choose_next_hypothesis':
            return False
        require_verified(study.get('discovery', {}).get('result'))
    run_id = state.get('wait_for_run')
    if not run_id:
        return True
    result = read(STATE / 'runs' / run_id / 'result.json')
    if not result:
        return False
    if result['status'] not in SUCCESS:
        raise RuntimeError('Dependency needs diagnosis: ' + run_id)
    return True


def require_verified(result):
    if not result or not result['qualifies']:
        return
    groups = read(STATE / 'verification.json', dict(groups=[]))['groups']
    group = next((g for g in groups if g['discovery_run']['run_id'] == result['run_id']), None)
    if not group or group['status'] != 'complete':
        raise RuntimeError('Frozen verification is unfinished: ' + result['run_id'])


def result_path(state):
    name = state.get('results_name', 'ks3-learning-ablation-results.json')
    if Path(name).name != name or not name.endswith('.json'):
        raise RuntimeError('Queue results require a local JSON filename')
    path = RESULTS / name
    if path.exists():
        raise RuntimeError('Queue results already exist; preserve prior evidence: ' + str(path))
    return path


def ensure_correctness(item, recipe, state, manifest, unit, provenance):
    """Gate a new geometry/shape on its independent oracle under the GPU lease."""
    if not item.get('correctness_required'):
        return
    audit = item['cpu_audit']
    path = ROOT / audit['path']
    if sha(path) != audit['sha256']:
        raise RuntimeError('CPU architecture audit changed')
    checked = read(path)
    if not checked.get('passed') or checked.get('gpu_initialized') is not False:
        raise RuntimeError('CPU architecture audit did not pass')
    if item.get('correctness_run_id'):
        directory = STATE / 'runs' / item['correctness_run_id']
        run = read(directory / 'run.json')
        result = read(directory / 'result.json')
        if not run or not result:
            raise RuntimeError('Interrupted correctness run needs reconciliation')
    else:
        run = controller.create_run(recipe, 172, 'correctness', unit)
        run['scheduler'] = provenance
        atomic(Path(run['directory']) / 'run.json', run)
        atomic(RESULTS / (run['run_id'] + '-config.json'), run)
        item['correctness_run_id'] = run['run_id']
        atomic(manifest, state)
        result = controller.execute(run)
    item['correctness_result'] = result
    atomic(manifest, state)
    if result.get('status') != 'correctness_passed' or not result.get('correctness', {}).get('passed'):
        raise RuntimeError('Independent KS3 oracle did not pass: ' + run['run_id'])
    if run['recipe'] != recipe or run['source_hashes'] != controller.source_hashes():
        raise RuntimeError('Independent KS3 oracle differs from the candidate source/recipe')


def run_queue(manifest, unit):
    if stopped():
        return
    state = read(manifest)
    if state and state.get('status') == 'complete_choose_next_hypothesis':
        return
    if not state or not read(RESULTS / 'control_set.json'):
        raise RuntimeError('Prepared queue and measured controls are required')
    output = result_path(state)
    for item in state['items']:
        checked_recipe(item)
    provenance = dict(path=str(Path(__file__).resolve().relative_to(ROOT)),
                      sha256=sha(Path(__file__)), code_commit=controller.git('rev-parse', 'HEAD'))
    # A prerequisite may itself need this lease to deploy or recover. Waiting
    # outside the lease prevents a downstream queue from blocking that work.
    state.update(unit=unit, scheduler=provenance, status='waiting_for_prerequisites')
    atomic(manifest, state)
    while not initial_ready(read(STATE / 'initial-research.json', {})) or not dependency_ready(state):
        if stopped():
            return
        time.sleep(1)
    with (STATE / 'ablation-queue.lock').open('a') as queue_lock:
        while True:
            if stopped():
                return
            try:
                fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.5)
        if not initial_ready(read(STATE / 'initial-research.json', {})) or not dependency_ready(state):
            raise RuntimeError('Prerequisite changed before ablation lease acquisition')
        with (ROOT / 'logs/wr-reproduction/gpu.lock').open('a') as gpu_lock:
            while True:
                if stopped():
                    return
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.5)
            if state.get('wait_for_run'):
                require_verified(read(STATE / 'runs' / state['wait_for_run'] / 'result.json'))
            if state.get('wait_for_study') and not dependency_ready(state):
                raise RuntimeError('Study prerequisite changed before GPU acquisition')
            state['status'] = 'running'
            atomic(manifest, state)
            for item in state['items']:
                if stopped():
                    return
                recipe = checked_recipe(item)
                ensure_correctness(item, recipe, state, manifest, unit, provenance)
                if stopped():
                    return
                if item.get('run_id'):
                    directory = STATE / 'runs' / item['run_id']
                    run = read(directory / 'run.json')
                    result = read(directory / 'result.json')
                    if not result:
                        raise RuntimeError('Interrupted ablation needs reconciliation: ' + item['run_id'])
                else:
                    run = controller.create_run(recipe, item['seed'], 'discovery', unit)
                    run['scheduler'] = provenance
                    atomic(Path(run['directory']) / 'run.json', run)
                    atomic(RESULTS / (run['run_id'] + '-config.json'), run)
                    item.update(run_id=run['run_id'], status='running')
                    atomic(manifest, state)
                    atomic(STATE / 'phase.json', dict(phase='ks3_config_ablations',
                           current_run_id=run['run_id'], manifest=str(manifest), utc=now()))
                    result = controller.execute(run)
                item.update(status=result['status'], result=result)
                atomic(manifest, state)
                if result['status'] not in SUCCESS:
                    raise RuntimeError('Ablation needs diagnosis: ' + result['run_id'])
                if result['qualifies']:
                    controller.verification(run, result, unit)
                    if stopped():
                        return
                    require_verified(result)
            state.update(status='complete_choose_next_hypothesis', completed_utc=now())
            atomic(manifest, state)
            atomic(output, state)
            atomic(STATE / 'phase.json', dict(phase='learning_ablations_complete_continue_research', utc=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=STATE / 'pending-ablations.json')
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    try:
        run_queue(args.manifest, args.unit)
    except BaseException:
        append(STATE / 'controller-errors.jsonl', dict(unit=args.unit, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
