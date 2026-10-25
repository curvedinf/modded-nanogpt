"""Gate direct BF16 encoding inside dense KS3 after the prepared trial queue.

No shared kernel changes. The GPU lease covers deployment, independent direct
and original-mode oracles, paired timings, discovery, and frozen verification.
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
    ROOT, STATE, RESULTS, read, atomic, append, sha, now, stopped, assert_fence, events,
)
from run_wr_matched_dense2816 import acquire, require_verification_complete
from run_wr_matched_dense_fp8 import timing_measurement, timing_decision

ALLOWED = frozenset('wr_reproduction/' + name for name in (
    'ks3_feedforward.py', 'ks3_moe.py', 'check_ks3_feedforward_bf16_direct.py', 'matched_controller.py'))
PROTOTYPE = RESULTS / 'dense-direct-bf16-prototype-v2.json'
PROGRESS = STATE / 'dense-direct-bf16-study.json'
OUTPUT = RESULTS / 'dense-direct-bf16-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_dense_direct_bf16.json'


def predecessor_ready(value):
    items = value.get('items', [])
    if value.get('status') in ('error', 'incomplete', 'stopped') or any(
            i['status'] in ('error', 'incomplete', 'stopped') for i in items):
        raise RuntimeError('Prepared width1280 queue needs diagnosis')
    return (value.get('status') == 'complete_choose_next_hypothesis' and len(items) == 1
            and items[0]['status'] in ('complete', 'pruned'))


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if len(changes) != len(ALLOWED) or {c['path'] for c in changes} != ALLOWED:
        raise RuntimeError('Direct encoding exceeds dense KS3 and oracle/controller scope')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Prototype patch changed')
    key = 'staged_sha256' if deployed else 'base_sha256'
    for item in changes:
        path = ROOT / item['path']
        actual = sha(path) if path.exists() else None
        if actual != item[key]:
            raise RuntimeError('Prototype source mismatch: ' + item['path'])
    for name, expected in value['unchanged_sources'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Unchanged KS3 dependency differs: ' + name)
    if sha(ROOT / value['cpu_auditor']['path']) != value['cpu_auditor']['sha256']:
        raise RuntimeError('CPU audit implementation changed')
    if len(value['cpu_audits']) != 4:
        raise RuntimeError('All architecture CPU audits are required')
    audited_architectures = set()
    for record in value['cpu_audits']:
        path = ROOT / record['path']
        if sha(path) != record['sha256']:
            raise RuntimeError('CPU audit changed')
        audit = read(path)
        if not audit.get('passed') or audit.get('gpu_initialized') is not False:
            raise RuntimeError('CPU architecture/state audit did not pass')
        if not audit.get('oracle_projection_shape_check'):
            raise RuntimeError('CPU oracle projection shape audit is missing')
        audited_architectures.add(audit['architecture'])
        for name, expected in audit['staged_source_hashes'].items():
            source = 'wr_reproduction/' + name
            change = next((c for c in changes if c['path'] == source), None)
            if (change['staged_sha256'] if change else value['unchanged_sources'][source]) != expected:
                raise RuntimeError('Audited source differs from prototype')
        checks = audit.get('cache_policy_checks', [])
        if len(checks) != 8 or len(audit.get('autograd_arity_checks', [])) != 2:
            raise RuntimeError('CPU cache or autograd audit incomplete')
        for check in checks:
            expected = check['precision'] == 'fp8' or not check['training'] or check['dense_training_gs32']
            if check['bf16_cache_gs32'] is not expected or not check['fp8_refresh_preserved']:
                raise RuntimeError('CPU cache policy differs from the intended training/evaluation precision')
    if audited_architectures != {'ks3_all', 'ks3_first_mlp', 'ks3_last_mlp', 'ks3_both_mlp'}:
        raise RuntimeError('Architecture CPU audits are incomplete')
    baseline = read(ROOT / value['baseline_recipe']['path'])
    if sha(ROOT / value['baseline_recipe']['path']) != value['baseline_recipe']['sha256']:
        raise RuntimeError('Paired baseline recipe changed')
    recipe = value['recipe']
    experiment = {k: v for k, v in recipe.items() if k not in ('name', 'hypothesis', 'dense_training_gs32')}
    original = {k: v for k, v in baseline.items() if k not in ('name', 'hypothesis')}
    if experiment != original or recipe.get('dense_training_gs32') is not False:
        raise RuntimeError('Direct encoding must change only the dense quantization flag')
    if assert_fence(recipe)['backend_hash'] != value['backend_hash']:
        raise RuntimeError('Prototype backend differs from matched controls')
    return recipe, baseline


def deploy(value, progress):
    recipe, _ = verify_prototype(value)
    paths = sorted(ALLOWED) + [str(RECIPE_PATH.relative_to(ROOT))]
    if controller.git('status', '--porcelain', '--', *paths) or RECIPE_PATH.exists():
        raise RuntimeError('Owned deployment paths need reconciliation')
    patch = ROOT / value['patch']
    stat = subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=ROOT, text=True)
    destinations = [line.split('\t')[-1] for line in stat.splitlines()]
    if len(destinations) != len(ALLOWED) or set(destinations) != ALLOWED:
        raise RuntimeError('Patch writes outside audited KS3 scope')
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
    subprocess.run(['git', 'commit', '--only', '-m', 'Gate direct BF16 encoding for dense KS3 training',
                    '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='dense_ks3_direct_bf16_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=value['patch_sha256'],
           backend_hash=value['backend_hash'], decision='Direct and original-mode oracles plus paired timing required'))
    return True


def execute_once(progress, label, recipe, kind, seed, unit):
    if controller.source_hashes() != progress['source_hashes']:
        raise RuntimeError('Source changed during direct BF16 gates')
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
        raise RuntimeError('Recorded precision run differs from frozen inputs')
    return run, result


def require_oracle(result, expected_gs32):
    check = result.get('correctness') or {}
    if (result['status'] != 'correctness_passed' or not check.get('passed')
            or check.get('dense_training_projection_gs32', True) is not expected_gs32):
        raise RuntimeError('Independent precision oracle failed or tested a different encoding')


def work(unit):
    if stopped():
        return
    value = read(PROTOTYPE)
    progress = read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    if OUTPUT.exists():
        raise RuntimeError('Direct BF16 results already exist; preserve evidence')
    verify_prototype(value, progress['deployed'])
    progress.update(status='waiting_for_width1280_queue', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)),
                                   timing_guard_sha256=sha(ROOT / 'run_wr_matched_dense_fp8.py'),
                                   lease_guard_sha256=sha(ROOT / 'run_wr_matched_dense2816.py'),
                                   code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'dense-direct-bf16-controller.lock').open('a') as own_lock:
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
                importlib.reload(controller)
                recipe, baseline = verify_prototype(value, deployed=True)
                progress.setdefault('source_hashes', controller.source_hashes())
                progress['status'] = 'direct_and_original_mode_oracles'
                atomic(PROGRESS, progress)
                for label, config, expected in [('direct_correctness', recipe, False),
                                                 ('original_correctness', baseline, True)]:
                    if stopped():
                        return
                    _, result = execute_once(progress, label, config, 'correctness', 172, unit)
                    require_oracle(result, expected)
                measurements = {}
                for label, config in [('gs32_timing', baseline), ('direct_timing', recipe)]:
                    if stopped():
                        return
                    run, result = execute_once(progress, label, config, 'timing', 168, unit)
                    measurements[label] = timing_measurement(result, events(Path(run['directory']) / 'events.jsonl'))
                progress['timing_gate'] = timing_decision(measurements['gs32_timing'], measurements['direct_timing'])
                atomic(PROGRESS, progress)
                if progress['timing_gate']['passed'] and not stopped():
                    progress['status'] = 'direct_dense_bf16_discovery'
                    atomic(PROGRESS, progress)
                    run, result = execute_once(progress, 'discovery', recipe, 'discovery', 168, unit)
                    if result['status'] not in ('complete', 'pruned'):
                        raise RuntimeError('Direct dense BF16 discovery needs diagnosis')
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
                atomic(STATE / 'phase.json', dict(phase='dense_direct_bf16_complete_continue_research', utc=now()))


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
