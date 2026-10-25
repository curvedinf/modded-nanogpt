"""Small pinned D2H observations; event waits happen off the training thread."""
import queue
import threading
import time

from wr_reproduction.research import STATE, Screen


class ResearchStopped(RuntimeError):
    pass


class Observations:
    def __init__(self, run, log, clock, state=STATE):
        self.screen = Screen(run, state)
        self.log, self.clock = log, clock
        self.pending = queue.Queue(maxsize=4)
        self.failure = None
        self.abort = None
        self.durations = {}
        self.last_wall = 0.
        self.thread = threading.Thread(target=self._work, daemon=True, name='WR-observations')
        self.thread.start()

    def check(self):
        if (self.screen.state / 'STOP').exists():
            raise ResearchStopped('explicit_study_stop')
        if self.failure is not None:
            raise RuntimeError('Research observation worker failed') from self.failure
        if self.abort:
            raise ResearchStopped(self.abort['reason'])

    def submit(self, loss, step, signature):
        import torch
        self.check()
        # copy_ enqueues one scalar into pinned memory on the training stream.
        # The worker waits for this event only; no device-wide synchronize or
        # loss.item() is issued by the training thread for ordinary observations.
        host = torch.empty((), dtype=loss.dtype, pin_memory=True)
        host.copy_(loss, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending.put((host, event, step, signature, time.perf_counter()))

    def _work(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                host, event, step, signature, submitted = item
                event.synchronize()
                completed = time.perf_counter()
                wall = self.clock(completed)
                row = dict(event='train', step=step, objective=host.item(), train_seconds=wall,
                           objective_signature=signature, observation_method='pinned-scalar-event-wait-worker',
                           submitted_to_observed_seconds=completed-submitted)
                if self.last_wall:
                    history = self.durations.setdefault(signature['stage'], [])
                    history.append(wall-self.last_wall)
                    del history[:-8]
                self.last_wall = wall
                self.log(**row)
                if self.abort is None:
                    result = self.screen.observe(row)
                    if result['abort']:
                        self.abort = result
                        self.log(event='screening_abort', **result)
            except BaseException as error:
                self.failure = error
            finally:
                self.pending.task_done()

    def drain(self):
        self.pending.join()
        self.check()

    def close(self):
        self.pending.put(None)
        self.thread.join()

    def predicted_update_seconds(self, stage):
        history = self.durations.get(stage, [])
        # Conservative until actual timing exists; stage 0 starts far from cap.
        return max(history[-4:], default=2.5) * 1.05
