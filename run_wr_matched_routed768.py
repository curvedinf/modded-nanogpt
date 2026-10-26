"""Use the registered routed-width oracle before a single serialized ablation."""
from pathlib import Path
import traceback

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import ROOT, STATE, RESULTS, read, sha, atomic, append
from run_wr_matched_ablations import run_queue

ORACLE = ROOT / 'check_wr_ks3_routed768.py'
MANIFEST = STATE / 'pending-routed768-corrected-oracle.json'
UNIT = 'wr-matched-routed768-corrected-gate'
ORIGINAL_CREATE = controller.create_run


def create_run(recipe, seed, kind, unit, frozen=None):
    if (recipe['name'] != 'matched-ks3-all-routed768-normlr'
            or recipe['architecture'] != 'ks3_all' or recipe['charts'] != 768
            or recipe.get('dense_charts') != 1024 or recipe['rank'] != 1
            or recipe['precision'] != 'bf16'
            or recipe.get('routed_training_gs32', True) is not True):
        raise RuntimeError('Routed-width oracle wrapper is restricted to the prepared recipe')
    run = ORIGINAL_CREATE(recipe, seed, kind, unit, frozen)
    if kind == 'correctness':
        run['command'] = [str(controller.PYTHON), str(ORACLE), '--features', '768',
                          '--out', str(Path(run['directory']) / 'events.jsonl')]
        run['auxiliary_source_hashes'] = {str(ORACLE.relative_to(ROOT)): sha(ORACLE)}
        run['oracle_scope'] = 'Eight routed experts, including router/post/projection/input/geometry gradients, empty experts, cache bytes and graph replay; GS32/BF16 width768 only.'
    run['entrypoint'] = dict(path=Path(__file__).name, sha256=sha(Path(__file__)))
    atomic(Path(run['directory']) / 'run.json', run)
    atomic(RESULTS / (run['run_id'] + '-config.json'), run)
    return run


def main():
    manifest = read(MANIFEST)
    if sha(ORACLE) != manifest['oracle_sha256']:
        raise RuntimeError('Prepared routed oracle changed')
    controller.create_run = create_run
    try:
        run_queue(MANIFEST, UNIT)
    except BaseException:
        append(STATE / 'controller-errors.jsonl', dict(unit=UNIT, error=traceback.format_exc()))
        raise
    finally:
        controller.create_run = ORIGINAL_CREATE


if __name__ == '__main__':
    main()
