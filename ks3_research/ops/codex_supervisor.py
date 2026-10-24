"""One saved prompt, persistent Astra/max exec session, independent of SSH."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
STATE=ROOT/'logs/ks3-autoresearch'
PROMPT=ROOT/'configs/ks3/autonomous_prompt.txt'
CODEX='/root/.local/bin/codex'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    temp=STATE/'supervisor.json.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(STATE/'supervisor.json')


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    (STATE/'codex').mkdir(exist_ok=True)
    previous=json.loads((STATE/'supervisor.json').read_text()) if (STATE/'supervisor.json').exists() else {}
    session=previous.get('session_id')
    attempt=previous.get('attempt',0)
    failures=0
    while not (STATE/'STOP').exists():
        attempt+=1
        events_path=STATE/'codex'/f'attempt-{attempt:06d}.jsonl'
        error_path=STATE/'codex'/f'attempt-{attempt:06d}.stderr.log'
        final_path=STATE/'codex'/f'attempt-{attempt:06d}.last-message.txt'
        command=[CODEX,'exec','--model','gpt-6-astra','-c','model_reasoning_effort="max"',
                 '--dangerously-bypass-approvals-and-sandbox','--json','--color','never',
                 '--output-last-message',str(final_path)]
        command+=['resume',session,'-'] if session else ['-']
        state=dict(supervisor_pid=os.getpid(),state='starting',model='gpt-6-astra',reasoning='max',
                   yolo=True,attempt=attempt,session_id=session,prompt=str(PROMPT),
                   events_path=str(events_path),stderr_path=str(error_path),command=command,
                   started_utc=now(),updated_utc=now())
        save(state)
        with PROMPT.open('rb') as prompt, error_path.open('ab') as error, events_path.open('ab',buffering=0) as output:
            process=subprocess.Popen(command,cwd=ROOT,stdin=prompt,stdout=subprocess.PIPE,
                                     stderr=error,start_new_session=True)
            state.update(codex_pid=process.pid,process_group_id=process.pid,state='running')
            save(state)
            selector=selectors.DefaultSelector()
            selector.register(process.stdout,selectors.EVENT_READ)
            buffer=b''
            while selector.get_map():
                for key,_ in selector.select(timeout=10):
                    chunk=os.read(key.fd,65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output.write(chunk)
                    buffer+=chunk
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1)
                        try:
                            event=json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get('type')=='thread.started':
                            session=event.get('thread_id',session)
                            state['session_id']=session
                        state['last_event_type']=event.get('type')
                        state['last_event_utc']=now()
                        if event.get('usage'):
                            state['last_usage']=event['usage']
                state['updated_utc']=now()
                save(state)
            code=process.wait()
            selector.close()
        failures=failures+1 if code else 0
        delay=min(900,120*max(1,2**min(failures,3)))
        state.update(state='waiting_to_resume',last_exit_code=code,retry_after_seconds=delay,updated_utc=now())
        save(state)
        # A completed exec turn is not the end of the requested indefinite study.
        # Resume the same session with the same single saved prompt; never duplicate live jobs.
        deadline=time.monotonic()+delay
        while time.monotonic()<deadline and not (STATE/'STOP').exists():
            time.sleep(min(10,max(0,deadline-time.monotonic())))
            state['updated_utc']=now()
            save(state)
    state=locals().get('state',previous)
    state.update(state='stopped',updated_utc=now())
    save(state)


if __name__=='__main__':
    main()
