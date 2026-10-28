"""Prepare an isolated full-WR KS3 run; execute the printed launch.sh afterward."""
from pathlib import Path
import datetime
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen-updates', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--study', default='h100-ks3')
    parser.add_argument('--scheduled-steps', type=int, default=981)
    parser.add_argument('--drop-dense-layer', type=int, choices=(0,2,4))
    parser.add_argument('--drop-parallel-mlp', action='store_true')
    parser.add_argument('--dense-width', type=int, default=2816)
    parser.add_argument('--first-batch-units', type=int, choices=(4,8,12,16), default=8)
    parser.add_argument('--final-window-blocks',type=int,default=20)
    parser.add_argument('--eval-probe-windows',default='')
    args = parser.parse_args()
    if args.screen_updates < 0:
        parser.error('screen updates must be nonnegative')
    if args.scheduled_steps <= 0:
        parser.error('scheduled steps must be positive')
    if args.dense_width <= 0 or args.dense_width % 256:
        parser.error('dense width must be a positive multiple of 256')
    probe_windows=[int(v) for v in args.eval_probe_windows.split(',') if v]
    if not all(1<=w<=64 for w in [args.final_window_blocks,*probe_windows]):
        parser.error('window blocks must be in1..64')
    upstream = ROOT / 'wr_reproduction/pr393/upstream'
    manifest = json.loads((upstream.parent / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((upstream / name).read_bytes()).hexdigest() == digest, name
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'logs/wr-reproduction' / args.study / stamp
    run.mkdir(parents=True)
    shutil.copytree(upstream, run / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    model_path=run/'source/track_1_short/model/gpt.py'
    model_source=model_path.read_text()
    if args.drop_dense_layer is not None:
        assert model_source.count('NO_MLP_LAYERS = (7, 9)')==1
        model_source=model_source.replace('NO_MLP_LAYERS = (7, 9)',f'NO_MLP_LAYERS = {tuple(sorted((7,9,args.drop_dense_layer)))}')
    if args.drop_parallel_mlp:
        before='PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = 8, len(_MLP_LIVE_LAYERS)'
        assert model_source.count(before)==1
        model_source=model_source.replace(before,'PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = -1, len(_MLP_LIVE_LAYERS)')
        before='NUM_USED_MLP_SLOTS = PARALLEL_MLP_SLOT + 1'
        assert model_source.count(before)==1
        model_source=model_source.replace(before,'NUM_USED_MLP_SLOTS = PARALLEL_MLP_SLOT')
    if args.dense_width != 2816:
        assert model_source.count('MLP_HIDDEN_DIM = 2816')==1
        model_source=model_source.replace('MLP_HIDDEN_DIM = 2816',f'MLP_HIDDEN_DIM = {args.dense_width}')
    model_path.write_text(model_source)
    if args.first_batch_units != 8:
        config_path=run/'source/track_1_short/config.py'
        config_source=config_path.read_text()
        before='TrainingStage(duration=STAGE_DURATIONS[0], train_max_seq_len=896, batch_size=8 * 2048 * 8'
        assert config_source.count(before)==1
        config_path.write_text(config_source.replace(before,before.replace('batch_size=8 ',f'batch_size={args.first_batch_units} ')))
    if args.final_window_blocks!=20:
        config_path=run/'source/track_1_short/config.py'
        config_source=config_path.read_text();before='WS_POST_YARN_EXT = 20'
        assert config_source.count(before)==1
        config_path.write_text(config_source.replace(before,f'WS_POST_YARN_EXT = {args.final_window_blocks}'))
    path = run / 'source/train_gpt.py'
    source = path.read_text()
    marker = '    model: nn.Module = GPT('
    assert source.count(marker) == 1
    source = source.replace(marker, '    from experiments.h100_ks3.port import install as install_h100_ks3\n    KS3GPT = install_h100_ks3()\n    model: nn.Module = KS3GPT(')
    # The upstream script imported its display constant before the overlay.
    source = source.replace('{MOE_HIDDEN}', '{model.ks3_up.shape[-2]}')
    marker = '            uncompiled_model.complete_yarn_tables()'
    assert source.count(marker) == 1
    source = source.replace(marker, '            # Refresh KS3 from final shipped weights, on the training clock.\n            uncompiled_model.prepare_ks3_eval()\n' + marker)
    marker = '            # CPLM diagnostics (token-averaged over all ranks)'
    assert source.count(marker) == 1
    source = source.replace(marker, '            print0(f"[wr-result] steps:{step} loss:{val_loss:.9f} seconds:{training_time_ms/1000:.9f}", console=True)\n' + marker)
    if probe_windows:
        marker='            del val_batches, val_pulls'
        assert source.count(marker)==1
        source=source.replace(marker,'            if not last_step: del val_batches, val_pulls')
        marker="            # The clock is stopped: flush the log and collect the loop's garbage here."
        assert source.count(marker)==1
        source=source.replace(marker,f'''            if last_step:
                from experiments.h100_ks3.eval_probe import run as probe_eval_windows
                probe_eval_windows(uncompiled_model,training_manager,ngram_table,val_batches,val_pulls,{probe_windows!r},print0)
                del val_batches,val_pulls
'''+marker)
    if args.screen_updates:
        marker = '        last_step = (step == training_schedule.total_steps)'
        assert source.count(marker) == 1
        replacement = f"""        if step == {args.screen_updates}:
            deferred_gathers.flush()
            torch.cuda.synchronize()
            elapsed_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f"[timing-screen] updates:{{step}} elapsed_ms:{{elapsed_ms:.3f}} validation:skipped", console=True)
            flush_log()
            dist.destroy_process_group()
            return
""" + marker
        source = source.replace(marker, replacement)
    if args.profile:
        source = source.replace('    barrier_start = time.perf_counter()', '    from experiments.h100_ks3 import profile_full\n    profile_full.start(env.rank)\n    barrier_start = time.perf_counter()')
        source = source.replace('            model.eval()\n', '            profile_full.validation(env.rank)\n            model.eval()\n')
        source = source.replace('    print0(f"peak memory allocated:', '    profile_full.stop(env.rank)\n    print0(f"peak memory allocated:')
    path.write_text(source)
    for folder in ['experiments/h100_ks3', 'wr_reproduction', 'ks3_research']:
        shutil.copytree(ROOT / folder, run / 'overlay' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', 'upstream'))
    (run / 'execution_manifest.json').write_text(json.dumps({
        str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['source', 'overlay'] for p in (run / folder).rglob('*.py')
    }, indent=2) + '\n')
    script = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/source"
export PATH=/root/venvs/pr393-h100/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export LOCAL_KERNELS=devenpzak/flash-attn3-12864=/root/venvs/pr393-fa3-pinned
export DATA_PATH=/root/modded-nanogpt
export PYTHONPATH="$(pwd)/../overlay"
export NUM_SCHEDULED_ITERATIONS=__SCHEDULED_STEPS__
nvidia-smi > ../gpu-environment.txt
uv pip freeze --python /root/venvs/pr393-h100/bin/python > ../packages.txt
set +e
bash run.sh > ../console.log 2>&1
result=$?
printf '%s\\n' "$result" > ../exit_code.txt
exit "$result"
'''
    (run / 'launch.sh').write_text(script.replace('__SCHEDULED_STEPS__',str(args.scheduled_steps)))
    (run / 'launch.sh').chmod(0o755)
    print(run / 'launch.sh')

if __name__ == '__main__':
    main()
