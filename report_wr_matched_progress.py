"""Refresh derived matched-study summaries without changing run evidence."""
from collections import Counter

from wr_reproduction.matched_study import STATE, RESULTS, atomic, read, sha, now


def main():
    control = read(RESULTS / 'control_set.json')
    discoveries = []
    for path in (STATE / 'runs').glob('*/run.json'):
        run = read(path)
        if run['kind'] != 'discovery':
            continue
        recipe = run['recipe']
        result = read(path.parent / 'result.json', {})
        status = read(path.parent / 'status.json', {})
        pruning = result.get('pruning') or {}
        discoveries.append(dict(
            run_id=run['run_id'], created_utc=run['created_utc'],
            recipe_name=recipe['name'], architecture=recipe['architecture'],
            seed=run['seed'], code_commit=run['code_commit'],
            status=result.get('status', status.get('status', 'registered')),
            charts=recipe['charts'], dense_charts=recipe.get('dense_charts', recipe['charts']),
            routed_precision=recipe['precision'], dense_precision=recipe.get('dense_precision', 'bf16'),
            dense_training_projection_gs32=recipe.get('dense_training_gs32', True),
            rank=recipe['rank'], projection_lr_multiplier=recipe.get('projection_lr_multiplier', 1.0),
            dense_projection_lr_multiplier=recipe.get('dense_projection_lr_multiplier', recipe.get('projection_lr_multiplier', 1.0)),
            geometry_lr_multiplier=recipe['geometry']['lr_mul'],
            dense_geometry_lr_multiplier=recipe.get('dense_geometry_lr_multiplier', recipe['geometry']['lr_mul']),
            initial_raw_scale=recipe['geometry']['raw_scale'],
            proxy_prune_seconds=pruning.get('train_seconds'),
            training_proxy_gap=pruning.get('gap'),
            reference_run_id=pruning.get('reference_run_id'),
            reference_seconds=pruning.get('reference_wall_seconds'),
            validation=result.get('validation'), qualifies=result.get('qualifies', False)))
    discoveries.sort(key=lambda item: item['created_utc'])
    groups = read(STATE / 'verification.json', dict(groups=[]))['groups']
    latest = read(STATE / 'latest-run.json', {})
    latest_run = latest.get('run', {})
    latest_recipe = latest_run.get('recipe', {})
    current = dict(run_id=latest.get('run_id'), status=latest.get('status'),
                   kind=latest_run.get('kind'), recipe=latest_recipe.get('name'),
                   routed_precision=latest_recipe.get('precision'),
                   dense_precision=latest_recipe.get('dense_precision', 'bf16'),
                   dense_training_projection_gs32=latest_recipe.get('dense_training_gs32', True),
                   timing_only=latest_run.get('kind') == 'timing')
    value = dict(
        utc=now(), derived_summary=True, backend_hash=control['backend_hash'],
        control_set_sha256=sha(RESULTS / 'control_set.json'),
        nominal_training_budget_seconds=control['nominal_training_seconds'],
        matched_controls=control['controls'], matched_mlp_best=control['best'],
        matched_mlp_mean=control['mean'], matched_mlp_sample_std=control['sample_std'],
        discovery_status_counts=dict(Counter(item['status'] for item in discoveries)),
        ks3_full_validation_count=sum(item['validation'] is not None for item in discoveries),
        qualifying_discoveries=sum(item['qualifies'] for item in discoveries),
        discoveries=discoveries,
        latest_run=current,
        verification=[dict(recipe_id=g['recipe_id'], status=g['status'],
                           discovery_run_id=g['discovery_run']['run_id'],
                           statistics=g.get('statistics')) for g in groups])
    atomic(RESULTS / 'study-progress.json', value)
    controls = ', '.join(f"seed {r['seed']}: {r['loss']:.8f} at {r['train_seconds']:.2f}s"
                         for r in control['controls'])
    lines = ['# Matched KS3 study progress', '', f"Snapshot: {value['utc']}", '',
             f'Matched MLP full-validation losses: {controls}.',
             f"Mean {control['mean']:.8f}; sample SD {control['sample_std']:.8f}. "
             f"Nominal budget: {control['nominal_training_seconds']:g} timed training seconds.", '',
             f"KS3 discovery full-validation results: {value['ks3_full_validation_count']}. "
             f"Qualifying discoveries: {value['qualifying_discoveries']}.", '',
             f"Latest run: {current['run_id']} ({current['kind']}, {current['status']}). "
             'Timing diagnostics are excluded from convergence and pruning references.', '',
             '| Recipe | Status | Proxy-prune elapsed (s) | Proxy gap to compatible reference | Full validation |',
             '|---|---|---:|---:|---:|']
    for item in discoveries:
        seconds = '—' if item['proxy_prune_seconds'] is None else f"{item['proxy_prune_seconds']:.2f}"
        gap = '—' if item['training_proxy_gap'] is None else f"{item['training_proxy_gap']:+.5f}"
        val = item['validation']
        loss = 'Not measured' if val is None else str(val['loss'])
        lines.append(f"| {item['recipe_name']} | {item['status']} | {seconds} | {gap} | {loss} |")
    lines += ['', 'Proxy gaps use each run’s compatible, nonfuture reference within 15 seconds. '
              'They are not validation-progress measurements and do not rank full-validation architecture quality. '
              'The earlier first-layer adapter failure remains preserved; its corrected retry is listed separately.', '',
              f"Frozen shared backend: `{control['backend_hash']}`. All three controls use full "
              '10,485,760-token validation, raw learned weights and accepted 640/1408 evaluation state.', '']
    target = RESULTS / 'study-progress.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(target)
    print({key: value[key] for key in ('utc', 'discovery_status_counts', 'ks3_full_validation_count', 'qualifying_discoveries')})


if __name__ == '__main__':
    main()
