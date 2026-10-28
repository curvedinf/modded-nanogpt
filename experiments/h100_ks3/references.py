"""Resolved timing references, keeping completed calibration distinct from a single run."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / 'research_results/wr_reproduction/h100-control-calibration/summary.json'


def local_control_reference(path=CALIBRATION):
    if Path(path).exists():
        report = json.loads(Path(path).read_text())
        if report['completed_fresh_runs'] == 11:
            rows = report['fresh_runs']
            assert len(rows) == len({row['run_id'] for row in rows}) == 11
            assert len({row['seed'] for row in rows}) == 11
            assert all(row['status'] == 'complete' and row['source_hashes_unchanged'] for row in rows)
            times = [row['official_seconds'] for row in rows]
            assert all(math.isfinite(value) and value > 0 for value in times)
            return dict(kind='eleven_fresh_control_mean', n=11,
                        seconds=statistics.mean(times), sample_std_seconds=statistics.stdev(times),
                        loss=statistics.mean(row['validation_loss'] for row in rows),
                        run_ids=[row['run_id'] for row in rows], source=str(Path(path)))
    return dict(kind='historical_single_control', n=1, seconds=31.099, loss=3.2765,
                run_ids=['20261028T180502Z'])
