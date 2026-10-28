"""Preserve source snapshots, complete logs and machine metadata off the rental."""
import argparse,gzip,hashlib,io,json,shlex,tarfile
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();run=a.run.resolve();files={}
    for folder in ('source','overlay'):
        if not (run/folder).exists():continue
        for p in (run/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
                files[str(p.relative_to(run))]=p.read_bytes()
    for name in ('launch.sh','settings.json','result.json','execution_manifest.json','source_manifest.json',
                 'run_metadata.json','console.log','gpu-environment.txt','packages.txt','exit_code.txt'):
        if (run/name).is_file():files[name]=(run/name).read_bytes()
    if 'execution_manifest.json' in files:
        expected=json.loads(files['execution_manifest.json'])
        assert all(hashlib.sha256(files[p]).hexdigest()==h for p,h in expected.items())
    custom=json.loads(files.get('settings.json',b'{}')).get('environment',{})
    files['replay.sh']=('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'+
        ''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in custom.items())+'exec bash launch.sh\n').encode()
    files['ARCHIVE-MANIFEST.json']=(json.dumps({p:hashlib.sha256(b).hexdigest() for p,b in sorted(files.items())},indent=2)+'\n').encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for p,b in sorted(files.items()):
            info=tarfile.TarInfo(run.name+'/'+p);info.size=len(b);info.mode=0o755 if p.endswith('.sh') else 0o644
            tar.addfile(info,io.BytesIO(b))
    # Read back every archived byte, including the full run log and source.
    with tarfile.open(a.output) as tar:
        for p,b in files.items():assert tar.extractfile(run.name+'/'+p).read()==b
    print(json.dumps(dict(run_id=run.name,archive=str(a.output),bytes=a.output.stat().st_size,
                         files=len(files),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
