"""Report every fresh frozen-config seed and the WR mean-loss significance."""
import argparse,json,math,statistics
from pathlib import Path
import mpmath
from experiments.h100_ks3.research import RESULTS,STATE,utc


def student_cdf(t,df):
    if t==0:return .5
    x=df/(df+t*t)
    half=float(mpmath.betainc(df/2,.5,0,x,regularized=True))/2
    return half if t<0 else 1-half


def describe(values):
    return dict(n=len(values),mean=statistics.mean(values),sample_std=statistics.stdev(values),
                minimum=min(values),maximum=max(values))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('discovery');a=ap.parse_args()
    original=json.loads((STATE/a.discovery/'result.json').read_text())
    expected=original['settings']['environment'].copy();expected.pop('TRAIN_SEED')
    runs=[]
    for p in sorted((RESULTS/'runs').glob('*/result.json')):
        row=json.loads(p.read_text());settings=row.get('settings',{})
        if settings.get('verification_of')==a.discovery:runs.append(row)
    assert len(runs)>=3,'Need at least three completed fresh independent seeds'
    seeds=[r['settings']['seed'] for r in runs]
    assert len(set(seeds))==len(seeds) and original['settings']['seed'] not in seeds
    for row in runs:
        assert row['status']=='complete' and row['source_hashes_unchanged']
        assert row['manifest_sha256']==original['manifest_sha256']
        assert row['code_commit']==original['code_commit']
        env=row['settings']['environment'].copy();env.pop('TRAIN_SEED')
        assert env==expected
        for key in ('scheduled_steps','growth_steps','extension_steps','validation_protocol','model_dim','dense_width','ngram_multiplier'):
            assert row['settings'].get(key)==original['settings'].get(key),key
    losses=describe([r['validation_loss'] for r in runs]);times=describe([r['official_seconds'] for r in runs])
    if losses['sample_std']:
        t=(losses['mean']-3.28)/(losses['sample_std']/math.sqrt(losses['n']))
        p=student_cdf(t,losses['n']-1)
    else:
        t=None;p=0. if losses['mean']<3.28 else .5 if losses['mean']==3.28 else 1.
    report=dict(utc=utc(),discovery=a.discovery,discovery_excluded_from_statistics=True,
                code_commit=original['code_commit'],manifest_sha256=original['manifest_sha256'],
                runs=[dict(run_id=r['run_id'],seed=r['settings']['seed'],loss=r['validation_loss'],seconds=r['official_seconds']) for r in runs],
                loss=losses,seconds=times,one_sided_student_t=t,one_sided_p=p,
                reference_seconds=31.099,loss_threshold=3.28,significance_threshold=.01,
                local_control_win_with_significance=losses['mean']<=3.28 and p<.01 and times['mean']<31.099,
                scope='Fresh-seed result against the immutable local PR393 baseline; no claim of maintainer acceptance or current global record.')
    out=RESULTS/'verification'/f'{a.discovery}.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
