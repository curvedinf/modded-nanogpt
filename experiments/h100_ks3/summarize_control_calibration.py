"""Validate and report the eleven fresh, exact-PR393 control measurements."""

import csv
import hashlib
import json
import math
import statistics

from experiments.h100_ks3.control_calibration import ORIGINAL, OUT, utc
from experiments.h100_ks3.verification_summary import student_cdf


def describe(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if student_cdf(mid, n - 1) < 0.975:
            lo = mid
        else:
            hi = mid
    radius = (lo + hi) / 2 * se
    return dict(n=n, mean=mean, sample_std=sd, standard_error=se,
                mean_ci95=[mean - radius, mean + radius],
                minimum=min(values), maximum=max(values))


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    rows = sorted(summary['fresh_runs'], key=lambda row: row['control_index'])
    assert summary['completed_fresh_runs'] == len(rows) == 11
    assert [row['control_index'] for row in rows] == list(range(1, 12))
    assert [row['seed'] for row in rows] == list(range(601, 612))
    assert len({row['run_id'] for row in rows}) == 11
    manifest = json.loads((ORIGINAL / 'source_manifest.json').read_text())
    expected_hashes = {'source/' + name: sha for name, sha in manifest['files'].items()}
    first_folder = OUT / 'runs' / rows[0]['run_id']
    packages = (first_folder / 'packages.txt').read_bytes()
    launcher = (first_folder / 'launch.sh').read_bytes()
    for row in rows:
        folder = OUT / 'runs' / row['run_id']
        assert json.loads((folder / 'result.json').read_text()) == row
        assert row['status'] == 'complete' and row['exit_code'] == 0
        assert row['source_hashes_unchanged']
        assert row['steps'] == 1053 and row['validation_tokens'] == 10485760
        assert row['upstream_commit'] == manifest['commit']
        assert json.loads((folder / 'execution_manifest.json').read_text()) == expected_hashes
        settings = json.loads((folder / 'settings.json').read_text())
        assert settings['source_changes'] == [] and not settings['comparative_pruning']
        assert settings['seed'] == row['seed']
        assert settings['environment'] == {'TRAIN_SEED': str(row['seed'])}
        assert f"; seed {row['seed']}; " in (folder / 'console.log').read_text()
        assert (folder / 'packages.txt').read_bytes() == packages
        assert (folder / 'launch.sh').read_bytes() == launcher
        assert (OUT / 'archives' / (row['run_id'] + '.tar.gz')).is_file()

    report = dict(utc=utc(), upstream_commit=manifest['commit'],
                  validation_tokens=10485760, timed_training_updates=1053,
                  fresh_runs=rows, original=summary['original'],
                  primary_population='Preserved original plus eleven fresh independent seeds, 601 through 611; fresh-only statistics are also reported.',
                  spread='Sample standard deviation; CI95 is a Student-t interval for the mean.',
                  measurement_precision='Unmodified upstream logs round each time to 0.001 second and loss to 0.0001.',
                  packages_sha256=hashlib.sha256(packages).hexdigest(),
                  launcher_sha256=hashlib.sha256(launcher).hexdigest(),
                  source_changes=[], comparative_pruning=False)
    lines = [
        '# PR393 local control calibration', '',
        f"Frozen upstream commit: `{manifest['commit']}`.", '',
        'Eleven fresh independent seeds, with unchanged model/training sources, '
        '1,053 updates per run and full 10,485,760-token final WR validation. '
        'The final local reference combines these eleven with the preserved original, for twelve measurements. '
        'The eleven fresh runs are also summarized separately.', '',
        '| Population | Mean time (s) | Time SD (s) | Mean validation loss | Loss SD |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name, label, group in [
        ('fresh_only', '11 fresh runs', rows),
        ('combined_with_original', 'Original + 11 fresh runs', [summary['original'], *rows]),
    ]:
        times = describe([row['official_seconds'] for row in group])
        losses = describe([row['validation_loss'] for row in group])
        if losses['standard_error']:
            loss_t = (losses['mean'] - 3.28) / losses['standard_error']
            loss_p = student_cdf(loss_t, losses['n'] - 1)
        else:
            loss_t = None
            loss_p = 0.0 if losses['mean'] < 3.28 else 0.5 if losses['mean'] == 3.28 else 1.0
        report[name] = dict(seconds=times, loss=losses,
                            runs_at_or_below_target=sum(row['validation_loss'] <= 3.28 for row in group),
                            loss_target=3.28, one_sided_loss_t=loss_t, one_sided_loss_p=loss_p,
                            loss_target_met_with_p_lt_01=losses['mean'] <= 3.28 and loss_p < 0.01)
        for metric, computed in [('seconds', times), ('loss', losses)]:
            assert math.isclose(computed['mean'], summary[name][metric]['mean'], abs_tol=1e-12)
            assert math.isclose(computed['sample_std'], summary[name][metric]['sample_std'], abs_tol=1e-12)
        lines.append(f"| {label} | {times['mean']:.6f} | {times['sample_std']:.6f} | "
                     f"{losses['mean']:.7f} | {losses['sample_std']:.7f} |")
    lines += ['', 'SD is the sample standard deviation across runs. Original logs round time to '
              'milliseconds and loss to four decimal places; averaging does not restore the discarded precision.', '',
              '| Run | Seed | Time (s) | Validation loss |', '| --- | ---: | ---: | ---: |']
    original = summary['original']
    lines.append(f"| Original `{original['run_id']}` | Random default | {original['official_seconds']:.3f} | {original['validation_loss']:.4f} |")
    for row in rows:
        lines.append(f"| [{row['control_index']}](runs/{row['run_id']}/result.json) | {row['seed']} | "
                     f"{row['official_seconds']:.3f} | {row['validation_loss']:.4f} |")
    lines += ['', 'Every fresh result has a source manifest, full console log, environment capture and '
              'replay archive in this directory. No run was selected or pruned based on its loss or timing.', '']
    (OUT / 'averages.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'averages.md').write_text('\n'.join(lines))
    with (OUT / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['control_index', 'run_id', 'seed', 'official_seconds', 'validation_loss'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({name: report[name] for name in ('fresh_only', 'combined_with_original')}, indent=2))


if __name__ == '__main__':
    main()
