"""The runs in the RL folder: what each is, how far it got, and what is training now."""
import datetime
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from tools import ppo_neural
from tools.trainer.processes import REPO as ROOT, list_processes, training
from tools.trainer.settings import config_of

RL_ROOT = ppo_neural.DEFAULT_OUT
NUMBER = re.compile(r'-?\d+(\.\d+)?(e-?\d+)?')


@dataclass
class Run:
    name: str
    number: int
    path: Path
    state: str                 # training | pausing | paused | interrupted | finished | stopped
    iterations_done: int
    total: int | None
    started_from: str | None
    checkpoints: list = field(default_factory=list)
    pid: int | None = None


def run_folders(root):
    root = Path(root)
    folders = [p for p in root.glob('run*') if p.is_dir() and re.fullmatch(r'run\d+', p.name)] if root.is_dir() else []
    return sorted(folders, key=lambda p: int(p.name[3:]))


def next_run_number(root):
    folders = run_folders(root)
    return int(folders[-1].name[3:]) + 1 if folders else 1


_LOG_CACHE = {}               # train.log path -> ((mtime_ns, size), rows): a finished run's log never changes
_LOG_CACHE_MAX = 1000
_LOG_LOCK = threading.Lock()


def log_rows(run_dir):
    """The run's complete train.log lines, as numbers (and True/False). Parsed again only when the file changed."""
    path = Path(run_dir) / 'train.log'
    try:
        info = path.stat()
    except OSError:
        return []
    key = (info.st_mtime_ns, info.st_size)
    with _LOG_LOCK:
        hit = _LOG_CACHE.get(path)
    if hit is None or hit[0] != key:           # stat came first, so the rows are at least as new as the key
        hit = (key, _parse_log(path))
        with _LOG_LOCK:
            if len(_LOG_CACHE) >= _LOG_CACHE_MAX and path not in _LOG_CACHE:
                _LOG_CACHE.clear()
            _LOG_CACHE[path] = hit
    return list(hit[1])                        # the row dicts are shared: callers only read them


def _parse_log(path):
    try:
        text = path.read_text()
    except OSError:
        return []
    lines = text.split('\n')[:-1]          # the last piece has no newline yet: a line still being written
    rows = []
    for line in lines:
        if 'iteration=' not in line or 'update_s=' not in line:   # a line cut short and then written over
            continue
        try:
            pairs = [part.split('=') for part in line.split()]
            rows.append({k: (float(v) if NUMBER.fullmatch(v) else v == 'True') for k, v in pairs})
        except ValueError:
            continue
    return rows


def _started_from(config, root):
    start = (config or {}).get('start')
    if not start:
        return None
    path = Path(start)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path.parent.parent == Path(root).resolve():
        return f'{path.parent.name}/{path.stem}'
    return str(start)


def _state(path, pid):
    if pid is not None:
        return 'pausing' if (path / ppo_neural.PAUSE_FILE).exists() else 'training'
    try:
        recorded = json.loads((path / ppo_neural.STATE_FILE).read_text()).get('state')
    except (OSError, ValueError, AttributeError):
        recorded = None
    if recorded in ('paused', 'finished', 'stopped'):
        return recorded
    return 'interrupted' if (path / ppo_neural.RESUME_FILE).exists() else 'stopped'


def read_run(path, root, trained):
    config = config_of(path)
    total = ((config or {}).get('args') or {}).get('iterations')
    pid = trained.get(path.name)
    rows = log_rows(path)
    return Run(name=path.name, number=int(path.name[3:]), path=path, state=_state(path, pid),
               iterations_done=len(rows), total=total, started_from=_started_from(config, root),
               checkpoints=sorted(p.stem for p in path.glob('iter[0-9]*.npz')), pid=pid)


def list_runs(root=RL_ROOT, processes=None):
    """Every run, oldest first."""
    trained = training(list_processes() if processes is None else processes, root)
    return [read_run(path, root, trained) for path in run_folders(root)]


def status(root=RL_ROOT, processes=None):
    """What training is doing now: the active run (training, pausing, paused or interrupted), else the newest."""
    runs = list_runs(root, processes)
    active = [r for r in runs if r.state in ('training', 'pausing')] or \
             [r for r in runs if r.state in ('paused', 'interrupted')][-1:]
    run = active[0] if active else (runs[-1] if runs else None)
    if run is None:
        return {'state': 'idle', 'phase': None, 'run': None, 'iteration': None, 'iterations': None,
                'seconds_per_iteration': None, 'started': None, 'latest': {}}
    rows = log_rows(run.path)
    last = rows[-1] if rows else {}
    try:
        started = datetime.datetime.fromtimestamp((run.path / 'config.json').stat().st_mtime).isoformat(timespec='seconds')
    except OSError:
        started = None
    try:
        phase = json.loads((run.path / ppo_neural.STATE_FILE).read_text()).get('state') if run.pid else None
    except (OSError, ValueError, AttributeError):
        phase = None
    return {'state': run.state if active else 'idle', 'phase': phase, 'run': run.name,
            'iteration': int(last['iteration']) if 'iteration' in last else None, 'iterations': run.total,
            'seconds_per_iteration': (last['rollout_s'] + last['update_s']) if 'rollout_s' in last else None,
            'started': started, 'latest': last, 'pid': run.pid,
            'workers': ((config_of(run.path) or {}).get('args') or {}).get('workers')}
