"""A training run's life, in one place: its states, the commands each state allows, which run a command acts on, and
the refusals, worded the same wherever they come from (the command line, the page, an agent's MCP tools), with the
files that carry the protocol: the pause request, the lock around starting and resuming, and the backup that puts a
run back as it was when a resume's launch fails.

  training     its trainer is running                                  pause, stop
  pausing      running, asked to pause once its iteration is saved     stop
  paused       its trainer saved and exited as asked                   resume, stop
  interrupted  its trainer ended without a pause or a stop             resume, stop
  finished     it trained every iteration                              -
  stopped      it was stopped                                          -

Starting a new run needs every run at rest (finished or stopped): one run trains at a time. A run in its final
evaluation or shutting down its workers (its phase) can only be stopped.
"""
import contextlib
import json
import os
import time
from enum import StrEnum
from pathlib import Path

from tools import ppo_neural

LOCK = '.trainer.lock'
STALE_LOCK_SECONDS = 60


class State(StrEnum):
    TRAINING = 'training'
    PAUSING = 'pausing'
    PAUSED = 'paused'
    INTERRUPTED = 'interrupted'
    FINISHED = 'finished'
    STOPPED = 'stopped'


class Command(StrEnum):
    START = 'start'
    PAUSE = 'pause'
    RESUME = 'resume'
    STOP = 'stop'


RUNNING = (State.TRAINING, State.PAUSING)          # its trainer is running
WAITING = (State.PAUSED, State.INTERRUPTED)        # it can carry on
ALLOWED = {State.TRAINING: (Command.PAUSE, Command.STOP), State.PAUSING: (Command.STOP,),
           State.PAUSED: (Command.RESUME, Command.STOP), State.INTERRUPTED: (Command.RESUME, Command.STOP),
           State.FINISHED: (), State.STOPPED: ()}
CLOSING_PHASES = ('evaluating', 'finished')        # a running trainer past its last iteration: only stop applies


class TrainerError(Exception):
    """A request that cannot be carried out; the message is for people."""


Refused = TrainerError      # a command the runs' states do not allow


# --- a run's state ---
def state_of(path, pid):
    """The state of the run in `path`, whose trainer runs as `pid` (None: not running)."""
    path = Path(path)
    if pid is not None:
        return State.PAUSING if pause_requested(path) else State.TRAINING
    try:
        recorded = json.loads((path / ppo_neural.STATE_FILE).read_text()).get('state')
    except (OSError, ValueError, AttributeError):
        recorded = None
    if recorded in (State.PAUSED, State.FINISHED, State.STOPPED):
        return State(recorded)
    return State.INTERRUPTED if (path / ppo_neural.RESUME_FILE).exists() else State.STOPPED


def active(runs):
    """The run commands act on: the one whose trainer is running, else the newest that can carry on; None."""
    running = [r for r in runs if r.state in RUNNING]
    waiting = [r for r in runs if r.state in WAITING]
    return running[0] if running else (waiting[-1] if waiting else None)


def commands(run, phase=None):
    """The commands `run` allows now (none for no run), in the order the page offers them."""
    if run is None:
        return ()
    if run.state in RUNNING and phase in CLOSING_PHASES:
        return (Command.STOP,)
    return ALLOWED[State(run.state)]


# --- whether a command may act, and on which run ---
def may_start(runs):
    """Refuse a new run while any run is training, pausing, paused or interrupted."""
    for run in runs:
        if run.state in RUNNING:
            raise Refused(f'{run.name} is training; stop or pause it first')
        if run.state == State.PAUSED:
            raise Refused(f'{run.name} is paused; resume or stop it first')
        if run.state == State.INTERRUPTED:
            raise Refused(f'{run.name} was interrupted; resume or stop it first')


def to_pause(runs):
    """The run a pause acts on: the one training."""
    run = next((r for r in runs if r.state in RUNNING), None)
    if run is None:
        raise Refused('nothing is training')
    if Command.PAUSE not in ALLOWED[State(run.state)]:
        raise Refused(f'{run.name} is already pausing')
    return run


def to_stop(runs):
    """The run a stop acts on: the one training, else the newest paused or interrupted one."""
    run = active(runs)
    if run is None:
        raise Refused('nothing is training')
    return run


def to_resume(runs, name=None):
    """The run a resume acts on: `name`, else the newest paused or interrupted one; nothing may be running."""
    for other in runs:
        if other.state in RUNNING:
            raise Refused(f'{other.name} is training; stop or pause it first')
    if name is None:
        waiting = [r for r in runs if Command.RESUME in ALLOWED[State(r.state)]]
        if not waiting:
            raise Refused('nothing to resume')
        return waiting[-1]
    run = next((r for r in runs if r.name == name), None)
    if run is None:
        raise Refused(f'there is no {name}')
    if Command.RESUME not in ALLOWED[State(run.state)]:
        raise Refused(f'{run.name} is {run.state}; only a paused or interrupted run can resume')
    return run


# --- the protocol's files ---
def request_pause(path):
    """Ask the run's trainer to stop once its current iteration is saved."""
    (Path(path) / ppo_neural.PAUSE_FILE).touch()


def pause_requested(path):
    return (Path(path) / ppo_neural.PAUSE_FILE).exists()


def clear_pause(path):
    (Path(path) / ppo_neural.PAUSE_FILE).unlink(missing_ok=True)


def mark_stopped(path, iteration):
    """The run is over: record it stopped (after `iteration`) and drop any pause request."""
    ppo_neural.write_state(path, State.STOPPED.value, iteration)
    clear_pause(path)


@contextlib.contextmanager
def lock(root):
    """Only one start or resume at a time, whoever asks (the page, the command line, an agent)."""
    path = Path(root) / LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > STALE_LOCK_SECONDS:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            raise Refused('a run is already starting; try again in a moment') from None
    else:
        raise Refused('a run is already starting; try again in a moment')
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _atomic(path, data):
    temporary = path.with_name(path.name + '.resume-settings.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


@contextlib.contextmanager
def resuming(path, files=None):
    """Around a resume's launch: write `files` ({path: bytes}, a settings edit) into the run in `path` after keeping a
    copy of its config, and clear its pause request. If the launch raises, every file, and the pause request, go
    back as they were."""
    path, files = Path(path), files or {}
    backups = {file: file.read_bytes() for file in files}
    had_pause = pause_requested(path)
    if files:
        _atomic(path / f'config-before-resume-{time.time_ns()}.json', (path / 'config.json').read_bytes())
    try:
        for file, data in files.items():
            _atomic(file, data)
        clear_pause(path)
        yield
    except BaseException:
        for file, data in backups.items():
            _atomic(file, data)
        if had_pause:
            request_pause(path)
        raise
