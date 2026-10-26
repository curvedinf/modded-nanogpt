"""Deploy one audited KS3 shape admission after its predecessor and verification."""
import argparse
import copy
import fcntl
import math
from pathlib import Path
import subprocess
import time
import traceback

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import ROOT, STATE, RESULTS, read, atomic, append, sha, now, stopped, assert_fence
from run_wr_matched_dense2816 import acquire, require_verification_complete

SOURCE = 'wr_reproduction/ks3_moe.py'
PROTOTYPE = RESULTS / 'rank2-width768-prototype.json'
PROGRESS = STATE / 'rank2-width768-study.json'
OUTPUT = RESULTS / 'rank2-width768-study-results.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_rank2_width768_normlr.json'
ORACLE = ROOT / 'check_wr_ks3_rank2_width768.py'


def predecessor_ready(value):
    items = value.get('items', [])
    if value.get('status') in ('error', 'incomplete', 'stopped') or any(
            item['status'] in ('error', 'incomplete', 'stopped') for item in items):
        raise RuntimeError('Output-rate prerequisite needs diagnosis')
    return (value.get('status') == 'complete_choose_next_hypothesis' and len(items) == 1
            and items[0]['status'] in ('complete', 'pruned'))


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if len(changes) != 1 or changes[0]['path'] != SOURCE:
        raise RuntimeError('Prototype exceeds the single KS3 admission guard')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Audited patch changed')
    if sha(ROOT / SOURCE) != changes[0]['staged_sha256' if deployed else 'base_sha256']:
        raise RuntimeError('KS3 source differs from audited prototype')
    for path, expected in value['unchanged_sources'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Unchanged source differs: ' + path)
    for key in ('cpu_model_auditor', 'cpu_optimizer_auditor', 'cpu_model_audit', 'cpu_optimizer_audit', 'baseline_recipe', 'control_set'):
        record = value[key]
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('Audited input changed: ' + key)
    for path, expected in value['oracle_source_hashes'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Numerical oracle changed: ' + path)
    model = read(ROOT / value['cpu_model_audit']['path'])
    optimizer = read(ROOT / value['cpu_optimizer_audit']['path'])
    for audit in (model, optimizer):
        if not audit.get('passed') or audit.get('gpu_initialized') is not False or audit.get('rank') != 2:
            raise RuntimeError('CPU audit incomplete')
    if (model.get('guard_cases') != 60 or not model.get('only_admission_guard_changed')
            or model.get('newly_admitted') != [dict(precision='bf16', mode='tuned', rank=2, features=768)]
            or model.get('base_source_sha256') != changes[0]['base_sha256']
            or model.get('staged_source_sha256') != changes[0]['staged_sha256']
            or optimizer.get('staged_core_sha256') != changes[0]['staged_sha256']):
        raise RuntimeError('Admission or staged-source audit mismatch')
    if len(optimizer.get('checks', [])) != 4 or any(
            not c['shared_optimizer_table_unchanged'] or not c['output_2x_and_padding_preserved']
            or c['dense_features'] != 768 or c['routed_features'] != 768
            for c in optimizer['checks']):
        raise RuntimeError('Original optimizer policy audit incomplete')
    recipe = value['recipe']
    expected = copy.deepcopy(read(ROOT / value['baseline_recipe']['path']))
    expected.update(name='matched-ks3-all-rank2-width768-normlr', hypothesis=recipe['hypothesis'],
                    charts=384, dense_charts=384, projection_lr_multiplier=(1024/768)**.5)
    if recipe != expected or model.get('recipe_sha256') != value['recipe_sha256']:
        raise RuntimeError('Recipe differs from audited rank-two capacity ablation')
    if not value.get('kernel_arithmetic_unchanged') or assert_fence(recipe)['backend_hash'] != value['backend_hash']:
        raise RuntimeError('Shared backend or kernel arithmetic changed')
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
    subprocess.run(['git', 'commit', '--only', '-m', 'Admit audited rank-two768 KS3 capacity ablation', '--', *paths],
                   cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='rank2_width768_admission_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=value['patch_sha256'], backend_hash=value['backend_hash']))
    return True


def execute_kind(value, progress, recipe, kind, unit):
    record = progress.get(kind)
    if record:
        directory = STATE / 'runs' / record['run_id']
        run, result = read(directory / 'run.json'), read(directory / 'result.json')
        if not run or not result:
            raise RuntimeError('Interrupted ' + kind + ' needs reconciliation')
        return run, result
    run = controller.create_run(recipe, 172 if kind == 'correctness' else 168, kind, unit)
    run['scheduler'] = progress['scheduler']
    if kind == 'correctness':
        run['command'] = [str(controller.PYTHON), str(ORACLE), '--out', str(Path(run['directory']) / 'events.jsonl')]
        run['auxiliary_source_hashes'] = value['oracle_source_hashes']
    atomic(Path(run['directory']) / 'run.json', run)
    atomic(RESULTS / (run['run_id'] + '-config.json'), run)
    progress[kind] = dict(run_id=run['run_id'], status='running')
    atomic(PROGRESS, progress)
    result = controller.execute(run)
    progress[kind].update(status=result['status'], result=result)
    atomic(PROGRESS, progress)
    return run, result


def correctness_ok(result):
    check = result.get('correctness') or {}
    rows = check.get('checks', [])
    caches = check.get('cache_checks', [])
    expected = {(branch, tokens) for branch in ('dense', 'routed') for tokens in (1, 257, 4096)}
    return (result['status'] == 'correctness_passed' and check.get('passed') is True
            and check.get('rank') == 2 and check.get('features') == 768 and check.get('charts') == 384
            and check.get('relative_rms_tolerance') == .015 and len(rows) == 6
            and {(row.get('branch'), row.get('tokens')) for row in rows} == expected
            and all(row.get('relative_rms') and all(math.isfinite(v) and 0 <= v < .015
                    for v in row['relative_rms'].values()) for row in rows)
            and len(caches) == 5 and all(c.get('cache_bytes_exact') is True for c in caches))


def work(unit):
    if stopped():
        return
    value, progress = read(PROTOTYPE), read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    if OUTPUT.exists():
        raise RuntimeError('Study output exists; preserve prior evidence')
    verify_prototype(value, progress['deployed'])
    progress.update(status='waiting_for_output_rate', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)),
                                   code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'rank2-width768-controller.lock').open('a') as own_lock:
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
                    raise RuntimeError('Source changed during rank-two capacity trial')
                progress['status'] = 'correctness_gate'
                atomic(PROGRESS, progress)
                _, gate = execute_kind(value, progress, recipe, 'correctness', unit)
                if not correctness_ok(gate):
                    raise RuntimeError('Dense and routed numerical gates did not pass')
                if stopped():
                    return
                if progress['source_hashes'] != controller.source_hashes():
                    raise RuntimeError('Source changed after numerical gate')
                progress['status'] = 'rank2_width768_discovery'
                atomic(PROGRESS, progress)
                run, result = execute_kind(value, progress, recipe, 'discovery', unit)
                if result['status'] not in ('complete', 'pruned'):
                    raise RuntimeError('Rank-two capacity discovery needs diagnosis')
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
                atomic(STATE / 'phase.json', dict(phase='rank2_width768_complete_continue_research', utc=now()))


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
