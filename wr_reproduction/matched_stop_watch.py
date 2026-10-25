"""Terminate the detached manager and dependent jobs on either new-study STOP."""
import subprocess
import time
from wr_reproduction.matched_study import ROOT,STOP_FILES,stopped,atomic,now

def main():
    while not stopped():time.sleep(1)
    manager=ROOT/'logs/wr-reproduction/rebaseline-manager'
    (manager/'STOP').touch()
    atomic(manager/'matched-stop.json',dict(utc=now(),reason='explicit_new_study_STOP',
                                          markers=[str(p) for p in STOP_FILES if p.exists()]))
    subprocess.run(['systemctl','stop','wr-ks3-rebaseline.service'],check=True)

if __name__=='__main__':main()
