"""CPU-only process-group STOP regression, including a resistant descendant."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from wr_reproduction.probe_controller import terminate

def main():
    with tempfile.TemporaryDirectory() as tmp:
        pidfile=Path(tmp)/'child.pid'
        code="""import os,signal,time,sys
pid=os.fork()
if pid==0:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 open(sys.argv[1],'w').write(str(os.getpid()))
 while True:time.sleep(.1)
while True:time.sleep(.1)
"""
        p=subprocess.Popen([sys.executable,'-c',code,str(pidfile)],start_new_session=True)
        try:
            for _ in range(100):
                if pidfile.exists():break
                time.sleep(.01)
            child=int(pidfile.read_text());terminate(p)
            for _ in range(100):
                path=Path(f'/proc/{child}/stat')
                if not path.exists() or path.read_text().split()[2]=='Z':break
                time.sleep(.01)
            else:raise AssertionError('Trainer descendant survived termination')
        finally:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
    print('PASS: trainer and SIGTERM-resistant descendant stop as one process group')
if __name__=='__main__':main()
