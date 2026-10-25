"""Deploy the audited dense-only KS3 LR after the existing queue finishes."""
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

SOURCE = 'wr_reproduction/ks3_feedforward.py'
PROTOTYPE = RESULTS / 'dense-projection-lr150-prototype.json'
PROGRESS = STATE / 'dense-projection-lr-study.json'
RECIPE_PATH = ROOT / 'configs/wr_reproduction/matched_ks3_all_denselr150.json'


def predecessor_ready(value):
    for item in value.get('items', []):
        if item['status'] in ('error', 'incomplete', 'stopped'):
            raise RuntimeError('Predecessor needs diagnosis: ' + item.get('run_id', item['recipe']))
    return (value.get('status') == 'complete_choose_next_hypothesis'
            and len(value.get('items', [])) == 2
            and all(i['status'] in ('complete', 'pruned') for i in value['items']))


def verify_prototype(value, deployed=False):
    changes = value['staged_sources']
    if len(changes) != 1 or changes[0]['path'] != SOURCE:
        raise RuntimeError('Prototype may change only the KS3 feedforward overlay')
    if sha(ROOT / value['patch']) != value['patch_sha256']:
        raise RuntimeError('Prototype patch changed')
    expected = changes[0]['staged_sha256' if deployed else 'base_sha256']
    if sha(ROOT / SOURCE) != expected:
        raise RuntimeError('KS3 source differs from audited prototype')
    for key in ('cpu_optimizer_audit', 'cpu_model_audit'):
        record = value[key]
        if sha(ROOT / record['path']) != record['sha256']:
            raise RuntimeError('CPU audit changed: ' + key)
        audit = read(ROOT / record['path'])
        if not audit.get('passed') or audit.get('gpu_initialized') is not False:
            raise RuntimeError('CPU audit did not pass: ' + key)
    recipe = value['recipe']
    if (recipe['architecture'], recipe['charts'], recipe['rank'],
        recipe.get('dense_projection_lr_multiplier'), recipe.get('projection_lr_multiplier', 1.)) != ('ks3_all', 1024, 1, 1.5, 1.):
        raise RuntimeError('Unexpected dense-only learning-rate recipe')
    assert_fence(recipe)
    return recipe


def deploy(value, progress):
    recipe = verify_prototype(value)
    if controller.git('status', '--porcelain', '--', SOURCE, str(RECIPE_PATH.relative_to(ROOT))):
        raise RuntimeError('Owned deployment paths have uncommitted changes')
    if RECIPE_PATH.exists():
        raise RuntimeError('Dense-only recipe already exists; reconcile deployment')
    patch = ROOT / value['patch']
    stat = subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=ROOT, text=True)
    if [line.split('\t')[-1] for line in stat.splitlines()] != [SOURCE]:
        raise RuntimeError('Patch writes outside the audited KS3 source')
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
    if stopped():
        return False
    subprocess.run(['git', 'apply', str(patch)], cwd=ROOT, check=True)
    atomic(RECIPE_PATH, recipe)
    verify_prototype(value, deployed=True)
    audit = RESULTS / 'dense-projection-lr-deployed-cpu-audit.json'
    if audit.exists():
        raise RuntimeError('Deployed audit already exists; preserve evidence')
    subprocess.run([str(controller.PYTHON), str(ROOT / 'check_wr_dense_projection_lr.py'),
                    '--module', str(ROOT / SOURCE), '--out', str(audit)], cwd=ROOT, check=True)
    if stopped():
        return False
    paths = [SOURCE, str(RECIPE_PATH.relative_to(ROOT)), str(audit.relative_to(ROOT))]
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Enable independently tuned dense KS3 projection learning rate', '--', *paths], cwd=ROOT, check=True)
    progress.update(deployed=True, deployment_commit=controller.git('rev-parse', 'HEAD'), deployment_utc=now())
    atomic(PROGRESS, progress)
    append(RESULTS / 'milestones.jsonl', dict(milestone='dense_projection_lr_deployed',
           code_commit=progress['deployment_commit'], patch_sha256=value['patch_sha256'],
           backend_hash=value['backend_hash'], deployed_cpu_audit_sha256=sha(audit),
           kernel_arithmetic_unchanged=True))
    return True


def work(unit):
    if stopped():
        return
    value = read(PROTOTYPE)
    progress = read(PROGRESS, dict(deployed=False))
    if progress.get('status') == 'complete_choose_next_hypothesis':
        return
    verify_prototype(value, progress['deployed'])
    progress.update(status='waiting_for_warpinit_dense1280_queue', unit=unit,
                    scheduler=dict(path=Path(__file__).name, sha256=sha(Path(__file__)),
                                   code_commit=controller.git('rev-parse', 'HEAD')))
    atomic(PROGRESS, progress)
    with (STATE / 'dense-lr-controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not predecessor_ready(read(Path(value['prerequisite_manifest']), {})):
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
            if any(g['status'] != 'complete' for g in read(STATE / 'verification.json', dict(groups=[]))['groups']):
                raise RuntimeError('Frozen three-seed verification is unfinished; keep sources unchanged')
            if not progress['deployed'] and not deploy(value, progress):
                return
            recipe = verify_prototype(value, deployed=True)
            progress['status'] = 'dense_only_projection_lr_discovery'
            atomic(PROGRESS, progress)
            if progress.get('run_id'):
                directory = STATE / 'runs' / progress['run_id']
                run, result = read(directory / 'run.json'), read(directory / 'result.json')
                if not run or not result:
                    raise RuntimeError('Interrupted dense LR discovery needs reconciliation')
            else:
                run = controller.create_run(recipe, 168, 'discovery', unit)
                run['scheduler'] = progress['scheduler']
                atomic(Path(run['directory']) / 'run.json', run)
                atomic(RESULTS / (run['run_id'] + '-config.json'), run)
                progress['run_id'] = run['run_id']
                atomic(PROGRESS, progress)
                atomic(STATE / 'phase.json', dict(phase='dense_only_projection_lr_discovery', run_id=run['run_id'], utc=now()))
                result = controller.execute(run)
            progress['result'] = result
            atomic(PROGRESS, progress)
            if result['status'] not in ('complete', 'pruned'):
                raise RuntimeError('Dense-only LR discovery needs diagnosis')
            if result['qualifies']:
                controller.verification(run, result, unit)
                if stopped():
                    return
                if any(g['status'] != 'complete' for g in read(STATE / 'verification.json')['groups']):
                    raise RuntimeError('Dense-only LR verification needs diagnosis')
            progress.update(status='complete_choose_next_hypothesis', completed_utc=now())
            atomic(PROGRESS, progress)
            atomic(RESULTS / 'dense-projection-lr-study-results.json', progress)
            atomic(STATE / 'phase.json', dict(phase='dense_only_projection_lr_complete_continue_research', utc=now()))


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
