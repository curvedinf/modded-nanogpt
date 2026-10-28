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
        frozen = report.get('reference_frozen_by_user', False)
        if report['completed_fresh_runs'] == 11 or frozen:
            rows = report['fresh_runs']
            count = report['accepted_fresh_runs'] if frozen else 11
            assert count > 0 and report['completed_fresh_runs'] == count
            assert len(rows) == len({row['run_id'] for row in rows}) == count
            assert len({row['seed'] for row in rows}) == count
            assert all(row['status'] == 'complete' and row['source_hashes_unchanged'] for row in rows)
            # Only completed measurements augment the already collected control.
            population = [report['original'], *rows]
            assert len({row['run_id'] for row in population}) == count + 1
            times = [row['official_seconds'] for row in population]
            assert all(math.isfinite(value) and value > 0 for value in times)
            return dict(kind='original_plus_completed_control_mean', n=count + 1,
                        seconds=statistics.mean(times), sample_std_seconds=statistics.stdev(times),
                        loss=statistics.mean(row['validation_loss'] for row in population),
                        run_ids=[row['run_id'] for row in population], source=str(Path(path)))
    return dict(kind='historical_single_control', n=1, seconds=31.099, loss=3.2765,
                run_ids=['20261028T180502Z'])
