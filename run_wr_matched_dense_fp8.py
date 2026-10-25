"""Gate dense-only KS3 FP8 on correctness and a paired whole-model timing check."""
import argparse
import fcntl
import importlib
import math
from pathlib import Path
import subprocess
import time
import traceback

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import (
    ROOT, STATE, RESULTS, read, atomic, append, sha, now, stopped, assert_fence, events,
)
from run_wr_matched_dense2816 import acquire, predecessor_ready, require_verification_complete

ALLOWED = frozenset('wr_reproduction/' + name for name in (
    'ks3_feedforward.py', 'ks3_feedforward_source.py', 'ks3_dense_fp8.py',
    'check_ks3_feedforward_fp8.py', 'matched_controller.py'))
PROTOTYPE = RESULTS / 'dense-fp8-prototype.json'
PROGRESS = STATE / 'dense-fp8-study.json'
OUTPUT = RESULTS / 'dense-fp8-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_dense2816_fp8.json'


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if len(changes) != len(ALLOWED) or {c['path'] for c in changes} != ALLOWED:
        raise RuntimeError('Prototype exceeds dense KS3 and oracle/controller scope')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Prototype patch changed')
    key = 'staged_sha256' if deployed else 'base_sha256'
    for item in changes:
        path = ROOT / item['path']
        actual = sha(path) if path.exists() else None
        if actual != item[key]:
            raise RuntimeError('Prototype source mismatch: ' + item['path'])
    if len(value['cpu_audits']) != 4:
        raise RuntimeError('All architecture CPU audits are required')
    for record in value['cpu_audits']:
        path = ROOT / record['path']
        if sha(path) != record['sha256']:
            raise RuntimeError('CPU audit changed')
        audit = read(path)
        if not audit.get('passed') or audit.get('gpu_initialized') is not False:
            raise RuntimeError('CPU architecture/state audit did not pass')
        for name, expected in audit['staged_source_hashes'].items():
            entry = next(c for c in changes if c['path'] == 'wr_reproduction/' + name)
            if entry['staged_sha256'] != expected:
                raise RuntimeError('Audited source differs from the prototype')
    recipe = value['recipe']
    if (recipe['architecture'], recipe['precision'], recipe['dense_precision'],
        recipe['rank'], recipe['charts'], recipe['dense_charts']) != ('ks3_all', 'bf16', 'fp8', 1, 1024, 2816):
        raise RuntimeError('Unexpected dense-only precision recipe')
    if assert_fence(recipe)['backend_hash'] != value['backend_hash']:
        raise RuntimeError('Prototype backend differs from matched controls')
    return recipe


def deploy(value, progress):
    recipe = verify_prototype(value)
    paths = sorted(ALLOWED) + [str(RECIPE_PATH.relative_to(ROOT))]
    if controller.git('status', '--porcelain', '--', *paths) or RECIPE_PATH.exists():
        raise RuntimeError('Owned deployment paths need reconciliation')
    patch = ROOT / value['patch']
    stat = subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=ROOT, text=True)
    destinations = [line.split('\t')[-1] for line in stat.splitlines()]
    if len(destinations) != len(ALLOWED) or set(destinations) != ALLOWED:
        raise RuntimeError('Patch writes outside the audited KS3 scope')
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
    subprocess.run(['git', 'commit', '--only', '-m', 'Add gated FP8 products for dense KS3 feedforward sites', '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='dense_ks3_fp8_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=value['patch_sha256'],
           backend_hash=value['backend_hash'], decision='Independent oracle and paired timing required before convergence'))
    return True


def execute_once(progress, label, recipe, kind, seed, unit):
    if controller.source_hashes() != progress['source_hashes']:
        raise RuntimeError('Source changed during dense FP8 gates')
    item = progress.setdefault(label, {})
    if item.get('run_id'):
        directory = STATE / 'runs' / item['run_id']
        run, result = read(directory / 'run.json'), read(directory / 'result.json')
        if not run or not result:
            raise RuntimeError('Interrupted precision run needs reconciliation')
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
    if run['recipe'] != recipe or run['kind'] != kind or run['source_hashes'] != progress['source_hashes']:
        raise RuntimeError('Recorded precision run differs from its frozen inputs')
    return run, result


def timing_measurement(result, rows):
    if result['status'] != 'timing_complete' or result.get('validation'):
        raise RuntimeError('Paired timing did not complete without validation')
    timing = result['timing']
    if timing['steps'] != 100:
        raise RuntimeError('Exactly100 timing updates are required')
    observed = {r['step']: r['train_seconds'] for r in rows if r.get('event') == 'train'}
    if not all(math.isfinite(v) for v in (observed[25], observed[100], timing['train_seconds'])):
        raise RuntimeError('Timing observations must be finite')
    if not (0 < observed[25] < observed[100] <= timing['train_seconds']):
        raise RuntimeError('Timing observations are inconsistent')
    return dict(first100_seconds=timing['train_seconds'], updates25_to100_seconds=observed[100] - observed[25])


def timing_decision(baseline, candidate):
    keys = ('first100_seconds', 'updates25_to100_seconds')
    faster = all(candidate[k] < baseline[k] for k in keys)
    return dict(passed=faster, baseline=baseline, candidate=candidate,
                candidate_over_baseline={k: candidate[k] / baseline[k] for k in keys},
                validation=False, objective_losses_not_compared=True,
                protocol='Both use the same existing100-update timing loop, seed168 and source, outside the900s discovery/reference set',
                decision='Proceed to fenced900s discovery' if faster else 'Skip slower precision candidate; no convergence result')


def work(unit):
    if stopped():
        return
    value = read(PROTOTYPE)
    progress = read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    if OUTPUT.exists():
        raise RuntimeError('Precision results already exist; preserve evidence')
    verify_prototype(value, progress['deployed'])
    progress.update(status='waiting_for_geometry_learning_queue', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)),
                                   guard_sha256=sha(ROOT / 'run_wr_matched_dense2816.py'),
                                   code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'dense-fp8-controller.lock').open('a') as own_lock:
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
                importlib.reload(controller)
                recipe = verify_prototype(value, deployed=True)
                progress.setdefault('source_hashes', controller.source_hashes())
                progress['status'] = 'oracle_then_paired_timing'
                atomic(PROGRESS, progress)
                _, result = execute_once(progress, 'correctness', recipe, 'correctness', 172, unit)
                if result['status'] != 'correctness_passed' or not result['correctness']['passed']:
                    raise RuntimeError('Dense FP8 oracle did not pass')
                baseline = read(ROOT / value['baseline_recipe']['path'])
                if sha(ROOT / value['baseline_recipe']['path']) != value['baseline_recipe']['sha256']:
                    raise RuntimeError('BF16 timing counterpart changed')
                exempt = {'name', 'hypothesis', 'dense_precision', 'fp8_native_gemm_tuning'}
                if {k: v for k, v in baseline.items() if k not in exempt} != {k: v for k, v in recipe.items() if k not in exempt}:
                    raise RuntimeError('Timing counterpart changes more than KS3 precision')
                measurements = {}
                for label, config in [('bf16_timing', baseline), ('fp8_timing', recipe)]:
                    if stopped():
                        return
                    run, result = execute_once(progress, label, config, 'timing', 168, unit)
                    measurements[label] = timing_measurement(result, events(Path(run['directory']) / 'events.jsonl'))
                progress['timing_gate'] = timing_decision(measurements['bf16_timing'], measurements['fp8_timing'])
                atomic(PROGRESS, progress)
                if progress['timing_gate']['passed'] and not stopped():
                    progress['status'] = 'dense_fp8_discovery'
                    atomic(PROGRESS, progress)
                    run, result = execute_once(progress, 'discovery', recipe, 'discovery', 168, unit)
                    if result['status'] not in ('complete', 'pruned'):
                        raise RuntimeError('Dense FP8 discovery needs diagnosis')
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
                append(RESULTS / 'notes.jsonl', dict(hypothesis=recipe['hypothesis'],
                       result=progress['timing_gate'], decision='Record precision result and continue meaningful KS3 research'))
                atomic(STATE / 'phase.json', dict(phase='dense_fp8_complete_continue_research', utc=now()))


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
