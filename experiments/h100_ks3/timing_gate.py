"""Continue a precision trial only if its saved 100-update timing gate passes."""
import argparse,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from experiments.h100_ks3.research import ROOT,RESULTS,append


def prefix(text):
    rows={int(step):float(ms) for step,ms in re.findall(r'step:(\d+)/\d+ train_time:([\d.]+)ms',text)}
    if not {25,100}.issubset(rows):return None
    assert 0<rows[25]<rows[100]
    return dict(first100_ms=rows[100],updates26_to100_ms=rows[100]-rows[25])


def compare(candidate,reference):
    return all(candidate[key]>reference[key] for key in reference)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('config',type=Path);ap.add_argument('reference',type=Path)
    ap.add_argument('--allow-growth-extension',action='store_true',help='Compare the unchanged first100 updates when only the late growth duration is extended.')
    a=ap.parse_args()
    c=json.loads(a.config.read_text());ref=prefix((a.reference/'console.log').read_text())
    assert ref is not None
    previous=json.loads((a.reference/'settings.json').read_text())
    for key in ('model_dim','dense_width','ngram_multiplier'):
        assert c[key]==previous[key],key
    if a.allow_growth_extension:
        assert c['growth_steps']>=previous['growth_steps']
        assert c.get('scheduled_steps',981)==previous['scheduled_steps'] and previous['scheduled_steps']>100
    else:assert c['growth_steps']==previous['growth_steps']
    assert c.get('extension_steps',20)==previous.get('extension_steps',20)
    seed=int(c['environment'].get('TRAIN_SEED',str(previous['seed'])))
    assert seed==previous['seed'],'Paired timing gate requires the same seed'
    command=[sys.executable,'-m','experiments.h100_ks3.research','--variant','fp8-native','--seed',str(seed),'--hypothesis',c['hypothesis']]
    for key in ('model_dim','dense_width','growth_steps','ngram_multiplier'):
        command.extend(['--'+key.replace('_','-'),str(c[key])])
    command.extend(['--extension-steps',str(c.get('extension_steps',20))])
    for k,v in c['environment'].items():command.extend(['--env',k+'='+v])
    env={k:v for k,v in os.environ.items() if not k.startswith('KS3_')}
    p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,text=True)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda sig,frame:p.send_signal(sig))
    first=p.stdout.readline();print(first,end='',flush=True);run=Path(first.strip())
    assert run.is_dir(),first
    gate=None
    while p.poll() is None:
        log=run/'console.log'
        candidate=prefix(log.read_text()) if log.exists() else None
        if candidate is not None:
            reject=compare(candidate,ref)
            gate=dict(run_id=run.name,reference_run_id=a.reference.name,**candidate,reference=ref,reject=reject,
                      late_growth_extension_only=a.allow_growth_extension,
                      rule='Stop if both first 100 updates and updates 26-100 are slower than the saved same-architecture reference; otherwise continue this run.')
            (run/'timing-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
            append('micro-registry.jsonl',dict(kind='whole_update_timing_gate',**gate))
            print(json.dumps(gate),flush=True)
            if reject:p.terminate()
            break
        time.sleep(.2)
    output=p.communicate()[0]
    if output:print(output,end='',flush=True)
    if gate:
        out=RESULTS/'runs'/run.name;out.mkdir(parents=True,exist_ok=True)
        (out/'timing-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    raise SystemExit(p.returncode)


if __name__=='__main__':main()
