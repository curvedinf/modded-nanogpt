"""CPU preflight of the exact BOS-aligned target batches through sampled stages.

Uses the unchanged Shard.next_batch selection, mmap host storage and the original
CandidateBuilder. It neither initializes a GPU nor changes training data/state.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from mi300.config import Hyperparameters
from mi300.data import HEADER_BYTES, Shard
from mi300.sampled_softmax import CandidateBuilder
from ks3_research.planning import load_config, schedule_for
from ks3_research.sampling import scheduled_counts


def check(config):
    torch.set_num_threads(8)
    schedule = schedule_for(config, config['schedule_base_steps'], torch.device('cpu'))
    counts = scheduled_counts(schedule, config['sampled_candidates'])
    files = iter(sorted(glob.glob(Hyperparameters().train_files)))
    shard = None
    builder_type = CandidateBuilder
    if config.get('architecture') in ('pr393-moe', 'pr393-moe-ks3'):
        from control393.sampling import CandidateBuilder as builder_type
    builder = builder_type(50304, max(counts), max(config['batch_tokens']))
    builder.reset(0, 1)
    records = []
    for step in range(max(i for i, p in enumerate(counts) if p) + 1):
        stage, _ = schedule.lookup(step)
        while True:
            if shard is None:
                path = next(files)
                header = np.fromfile(path, dtype=np.int32, count=256)
                assert tuple(header[:2]) == (20240520, 1)
                tokens = np.memmap(path, dtype=np.uint16, mode='c', offset=HEADER_BYTES,
                                   shape=(int(header[2]),))
                shard = Shard(torch.from_numpy(tokens))
                shard._loader_thread.join()
            try:
                starts, ends = shard.next_batch(stage.batch_size, stage.train_max_seq_len)
                break
            except StopIteration:
                shard = None
        targets = np.concatenate([tokens[a:b] for a, b in zip(starts[0], ends[0])])[1:].astype(np.int64)
        distinct = np.count_nonzero(np.bincount(targets, minlength=50304))
        if counts[step]:
            candidates, pos, _ = builder.build(counts[step], targets, None)
            np.testing.assert_array_equal(candidates[pos], targets)
        records.append(dict(step=step, batch_tokens=stage.batch_size,
                            candidates=counts[step], distinct_targets=int(distinct)))
    return dict(cpu_only=True, measured_training=False, steps=len(records),
                maximum_distinct_targets=max(r['distinct_targets'] for r in records),
                records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    if raw.get('architecture') in ('pr393-moe', 'pr393-moe-ks3'):
        from control393.launch_config import load_config
    report = check(load_config(args.config))
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}))
