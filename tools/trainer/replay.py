"""Replaying training episodes: an iteration's episodes, how each went, and watching one in the game window."""
import sys
from pathlib import Path

from tools.trainer.control import TrainerError, launch_process
from tools.trainer.runs import RL_ROOT

SPEEDS = (1.0, 0.5, 0.25)          # normal speed and slower


def _run_dir(root, run, iteration):
    """The run's folder, if `iteration` is one it has finished (only those have their checkpoint to fly from)."""
    path = Path(root) / str(run)
    if not str(run).startswith('run') or not path.is_dir():
        raise TrainerError(f'there is no run {run}')
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0 \
            or not (path / f'iter{iteration:04d}.npz').exists():
        raise TrainerError(f'{run} has not finished iteration {iteration}')
    return path


def _task(path):
    """The registered task the run in `path` trained (ValueError for one this trainer does not know)."""
    from learning.tasks import UNTAGGED_TASK, task
    from tools.trainer.settings import config_of
    return task((config_of(path) or {}).get('task', UNTAGGED_TASK))


def _replay_mode(path):
    """How the run's episodes replay (see learning.tasks.Capabilities): 'hooks' (its task's replay module, checked
    before watching) or 'builtin' (the Hunter's episode jobs). Refuses a task whose mode is None."""
    mode = _task(path).capabilities.replay_mode
    if mode is None:
        raise TrainerError(f'{path.name} cannot be replayed: its task does not replay episodes')
    return mode


def replayable(path):
    """Whether the run recorded enough to fly its episodes again: its settings (runs from before run51 did not)."""
    from tools.trainer.settings import config_of
    config = config_of(Path(path)) or {}
    try:
        if _task(Path(path)).capabilities.replay_mode is None:
            return False
    except ValueError:            # an unknown task: the episodes request names it
        pass
    from tools import run_record
    return run_record.has_settings(config)


def _jobs(path, iteration):
    from tools.watch_episode import episode_jobs
    try:
        return episode_jobs(path, iteration)[0]
    except (SystemExit, KeyError, ValueError) as e:
        raise TrainerError(f'{path.name} cannot be replayed: {e}') from None


def episodes(run, iteration, root=RL_ROOT):
    """The iteration's episodes as training flew them, each with its level: quick, nothing is flown."""
    path = _run_dir(root, run, iteration)
    if _replay_mode(path) == 'hooks':
        return _task(path).replay().episodes(path, iteration)
    return [{'episode': i, 'level': level, 'seed': seed} for i, (level, seed, *_) in enumerate(_jobs(path, iteration))]


def fly(run, iteration, root=RL_ROOT, workers=8):
    """The iteration's episodes flown again, with how each went. Takes a while, and slows any training (the page
    never asks for it: a Campaign Hunter iteration is 96 episodes of up to 180 s each)."""
    from tools.watch_episode import episode_rows
    path = _run_dir(root, run, iteration)
    if _replay_mode(path) == 'hooks':
        return _fly_apart(path, iteration)
    _jobs(path, iteration)
    return episode_rows(path, iteration, workers)


def _fly_apart(path, iteration):
    """A creature task's episodes flown again in a process of their own. Flying starts pygame, which in the server
    would open its window on a request thread (macOS refuses) and take over SIGTERM and ctrl-c for good."""
    import json
    import os
    import subprocess
    from tools.trainer.processes import REPO
    env = dict(os.environ, SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', PYGAME_HIDE_SUPPORT_PROMPT='1')
    flown = subprocess.run([sys.executable, '-m', 'tools.watch_episode', str(path), str(iteration)], cwd=REPO,
                           env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if flown.returncode:
        reason = (flown.stderr or flown.stdout).strip().splitlines()
        raise TrainerError(f'{path.name} iteration {iteration} cannot be flown again: '
                           f'{reason[-1] if reason else f"exit status {flown.returncode}"}')
    return json.loads(flown.stdout)


def watch(run, iteration, episode, speed=1.0, root=RL_ROOT, launch=None):
    """Open the episode in the game window, at `speed` (one of SPEEDS)."""
    path = _run_dir(root, run, iteration)
    count = len(episodes(run, iteration, root))
    if not isinstance(episode, int) or isinstance(episode, bool) or not 0 <= episode < count:
        raise TrainerError(f'iteration {iteration} has episodes 0 to {count - 1}')
    if speed not in SPEEDS:
        raise TrainerError(f'the speed must be one of {", ".join(f"{s:g}" for s in SPEEDS)}')
    command = [sys.executable, '-m', 'tools.watch_episode', str(path), str(iteration), str(episode),
               '--speed', f'{speed:g}']
    if _replay_mode(path) == 'hooks':    # check the recorded rules still replay before opening a window
        import subprocess
        try:
            checked=subprocess.run(command+['--check'],capture_output=True,text=True,timeout=20)
        except subprocess.TimeoutExpired:
            raise TrainerError('Replay validation timed out; try again shortly.') from None
        if checked.returncode:
            reason=(checked.stderr or checked.stdout).strip().splitlines()
            raise TrainerError(f'{run} cannot be watched with the current rules: {reason[-1] if reason else "validation failed"}')
    (launch or launch_process)(command, Path(root) / 'replay.log')
    return command
