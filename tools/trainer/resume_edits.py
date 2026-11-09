"""Changing a paused run's settings as it resumes, for every task, in one place.

A run's settings are its recorded arguments. Everything else the run records about them (a hunter run's rewards rows,
episode block and shaping; a creature run's manifest, shaping and episode block) is recomputed from the edited
arguments by the code that wrote it at the start, never edited in step. This module owns the whole edit: checking the
changes, refusing a setting the run's task does not take, recomputing the records, changing the learning rate in the
saved optimiser, logging the edits in the run's config, and writing the files (after a backup, atomically, and back
as they were if the launch fails).
"""
import contextlib
import copy
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from tools import ppo_neural
from tools.trainer.settings import apply_changes, changed, config_of

LEARNING = ('lr', 'workers', 'iterations', 'budget_seconds')     # any run may change these on resume
# Settings a run records nowhere but in its arguments: changing only these recomputes nothing.
ARGUMENTS_ONLY = ('workers', 'iterations', 'budget_seconds')


class EditRefused(ValueError):
    """An edit a paused run cannot take; the message is for people."""


@dataclass
class Edit:
    """The run's settings with the changes applied (tools.trainer.settings Values), and the files to write for them."""
    values: dict
    files: dict = field(default_factory=dict)     # path -> bytes
    changes: list = field(default_factory=list)   # (setting, old, new)


def prepare(path, base, changes):
    """The edit `changes` ({setting: value}) makes to the paused run in `path`, whose settings are `base`. Nothing is
    written: see applied()."""
    path = Path(path)
    if not isinstance(changes, dict):
        raise EditRefused('resume changes must be a settings object')
    try:
        values = apply_changes(base, changes)
    except ValueError as exc:
        raise EditRefused(str(exc)) from None
    edits = changed(base, values)
    if not edits:
        return Edit(values)
    config = copy.deepcopy(config_of(path))
    unsupported = [key for key, _, _ in edits if key not in editable(config)]
    if unsupported:
        raise EditRefused('These settings require a new run: ' + ', '.join(unsupported))
    before = dict(config['args'])
    args = config['args']
    for key, _, value in edits:
        args[key] = value
    _check_learning(args)
    files = {}
    if any(key not in ARGUMENTS_ONLY for key, _, _ in edits):
        try:
            config = recorded(config, before, args, path)
        except ValueError as exc:
            raise EditRefused(str(exc)) from None
    if any(key == 'lr' for key, _, _ in edits):
        files[path / 'resume.pt'] = _with_learning_rate(path / 'resume.pt', args['lr'])
    _log(path, config, edits)
    files[path / 'config.json'] = json.dumps(config, indent=2).encode()
    return Edit(values, files, edits)


def editable(config):
    """The settings a paused run may change on resume: the learning ones for any run, and every setting its own task
    takes (its rewards, episodes and so on)."""
    from learning.tasks import UNTAGGED_TASK
    from tools.trainer import settings
    key = config.get('task', UNTAGGED_TASK)
    return set(LEARNING) | {dest for dest in settings.BY_DEST if dest != 'task' and key in settings.applies_to(dest)}


def _check_learning(args):
    lr, workers = args.get('lr'), args.get('workers', 1)
    iterations, budget = args.get('iterations', 0), args.get('budget_seconds', 0.)
    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise EditRefused('learning rate must be finite and positive')
    if workers < 1 or iterations < 0 or not math.isfinite(budget) or budget < 0:
        raise EditRefused('workers must be positive; iterations and budget must be nonnegative')


def arguments(args):
    """A run's recorded arguments as the trainer's parser would give them: options it did not record yet (added since
    it started) at their defaults."""
    from learning.tasks import UNTAGGED_TASK
    parsed = ppo_neural.build_parser().parse_args(['--task', args.get('task', UNTAGGED_TASK)])
    return SimpleNamespace(**{**vars(parsed), **args})


def recorded(config, before, after, path):
    """`config` with what it records about its arguments recomputed from `after` (the edited arguments; `before`, the
    ones it recorded). Raises ValueError for arguments the task refuses."""
    from learning.tasks import UNTAGGED_TASK
    key = config.get('task', UNTAGGED_TASK)
    if key in ppo_neural.PPO_TASKS:
        return _ppo_recorded(config, after)
    return _scenario_recorded(config, before, after, Path(path))


def _ppo_recorded(config, args):
    """A hunter-loop run (tools.ppo_neural): its rewards rows, episode block and shaping, from run_config(), except the
    shaping it keeps for life (ppo_neural.RUN_RULES), which stays as recorded (or absent, for a run older than it)."""
    fresh = ppo_neural.run_config(arguments(args))
    prior = config.get('shaping') or {}
    shaping = {k: v for k, v in fresh['shaping'].items() if k not in ppo_neural.RUN_RULES}
    shaping.update({k: prior[k] for k in ppo_neural.RUN_RULES if k in prior})
    episode = dict(config.get('episode') or {})
    episode.update(fresh['episode'])
    return dict(config, rewards=fresh['rewards'], episode=episode, shaping=shaping)


def _scenario_recorded(config, before, after, path):
    """A creature run (learning.run): its manifest, as learning.run.expected_manifest() works it out on resume, and the
    shaping and episode block it writes beside it. An edit that leaves the manifest alone leaves the run's records
    alone; one that changes it is refused if the run's rules have changed since it started (it could not resume)."""
    from learning.run import expected_manifest
    old, new = expected_manifest(_scenario_args(before, path)), expected_manifest(_scenario_args(after, path))
    if old == new:
        return config
    if config.get('manifest') != old:
        raise EditRefused('the rules this run trains with have changed since it started, so it cannot resume with '
                          'edited settings')
    episode = dict(config.get('episode') or {}, horizon_seconds=after['horizon'])
    return dict(config, manifest=new, shaping=new['rewards'], episode=episode)


def _scenario_args(args, path):
    """The arguments learning.run resumes with: a run that flies against an opponent uses its own copy of it."""
    from learning.tasks import task
    namespace = arguments(args)
    if getattr(task(namespace.task).adapter(), 'USES_OPPONENT', True):
        namespace.opponent = str((path / 'opponent.npz').resolve())
    return namespace


def _with_learning_rate(resume_path, lr):
    """The saved resume state with every optimiser group at `lr`."""
    import torch
    saved = torch.load(resume_path, map_location='cpu', weights_only=False)
    for group in saved['optimiser']['param_groups']:
        group['lr'] = lr
        if 'initial_lr' in group:
            group['initial_lr'] = lr
    output = io.BytesIO()
    torch.save(saved, output)
    return output.getvalue()


def _log(path, config, edits):
    """Note the edits in the run's config, with the iteration they come after."""
    state_path = path / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    config.setdefault('resume_settings_changes', []).append(dict(
        time=time.time(), after_iteration=state.get('iteration'),
        changes=[dict(setting=key, previous=old, value=new) for key, old, new in edits]))


def _atomic(path, data):
    temporary = path.with_name(path.name + '.resume-settings.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


@contextlib.contextmanager
def applied(edit, path):
    """Write the edit's files into the run in `path` for the body (the launch), keeping a copy of its config first. If
    the body raises, every file goes back as it was."""
    path = Path(path)
    if not edit.files:
        yield
        return
    backups = {file: file.read_bytes() for file in edit.files}
    _atomic(path / f'config-before-resume-{time.time_ns()}.json', (path / 'config.json').read_bytes())
    try:
        for file, data in edit.files.items():
            _atomic(file, data)
        yield
    except BaseException:
        for file, data in backups.items():
            _atomic(file, data)
        raise
