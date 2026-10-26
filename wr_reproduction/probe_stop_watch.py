"""Independent STOP watcher: survives manager exit and owns all probe-study units."""
import subprocess
import time
from wr_reproduction.probe_study import stopped,STOP_FILES,STATE,atomic,now

def main():
    while not stopped():time.sleep(.5)
    atomic(STATE/'stop-observed.json',dict(utc=now(),markers=[str(p) for p in STOP_FILES if p.exists()]))
    # systemd owns complete cgroups, including compiler/data-loader descendants.
    subprocess.run(['systemctl','stop','wr-probe-*','wr-ks3-probe-research.service'],check=True)
if __name__=='__main__':main()
