"""Run one isolated numerical diagnosis; never launch convergence training."""
from pathlib import Path
import fcntl

from wr_reproduction import matched_controller as controller
from wr_reproduction.matched_study import ROOT, STATE, RESULTS, read, atomic, append, sha, stopped
from run_wr_matched_dense2816 import acquire, require_verification_complete


def main():
    if stopped():
        return
    with (STATE / 'ablation-queue.lock').open('a') as queue:
        if not acquire(queue):
            return
        with (ROOT / 'logs/wr-reproduction/gpu.lock').open('a') as gpu:
            if not acquire(gpu):
                return
            require_verification_complete()
            recipe = read(ROOT / 'configs/wr_reproduction/matched_ks3_all_routed768_normlr.json')
            run = controller.create_run(recipe, 172, 'correctness', 'wr-matched-routed768-post-diagnostic')
            diagnostic = ROOT / 'diagnose_wr_routed768_post.py'
            run['command'] = [str(controller.PYTHON), str(diagnostic), '--features', '768',
                              '--out', str(Path(run['directory']) / 'events.jsonl')]
            run['diagnostic_only'] = True
            run['auxiliary_source_hashes'] = {str(diagnostic.relative_to(ROOT)): sha(diagnostic),
                                            Path(__file__).name: sha(Path(__file__))}
            atomic(Path(run['directory']) / 'run.json', run)
            atomic(RESULTS / (run['run_id'] + '-config.json'), run)
            atomic(STATE / 'routed768-post-diagnostic.json', dict(run_id=run['run_id'], status='running'))
            result = controller.execute(run)
            atomic(STATE / 'routed768-post-diagnostic.json', dict(run_id=run['run_id'], result=result))
            append(RESULTS / 'notes.jsonl', dict(run_id=run['run_id'],
                   decision='Inspect scalar derivative diagnosis. A diagnostic cannot satisfy the required correctness gate or start training.'))


if __name__ == '__main__':
    main()
