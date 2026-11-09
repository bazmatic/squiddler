"""Starting, stopping, pausing and resuming training runs. One run trains at a time: which command each run's state
allows, and the files that carry it out, are tools.trainer.lifecycle's."""
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tools.trainer import lifecycle, resume_edits
from tools.trainer.lifecycle import TrainerError  # noqa: F401 (the callers' error, raised here and by lifecycle)
from tools.trainer.processes import REPO, list_processes, terminate_tree
from tools.trainer.runs import RL_ROOT, list_runs, log_rows, next_run_number
from tools.trainer.settings import NoSettings, apply_changes, changed, config_of, read_settings, to_argv


@dataclass
class Started:
    run: str
    started_from: str
    changes: list = field(default_factory=list)
    command: list = field(default_factory=list)


def launch_process(command, log_path):
    """Run `command` in the background, outliving whoever asked, with its output appended to `log_path`."""
    with open(log_path, 'a') as log:
        return subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True).pid


def _runs(root, processes):
    return list_runs(root, list_processes() if processes is None else processes)


def _source(from_, runs):
    """(run, checkpoint name) to carry on from."""
    by_name = {run.name: run for run in runs}
    if from_ is None:
        candidates = [run for run in runs if run.checkpoints]
        if not candidates:
            raise TrainerError('there is no run with a checkpoint to carry on from')
        return candidates[-1], candidates[-1].checkpoints[-1]
    name, _, checkpoint = str(from_).strip().strip('/').partition('/')
    if name not in by_name:
        raise TrainerError(f'there is no {name}')
    run = by_name[name]
    if not run.checkpoints:
        raise TrainerError(f'{name} has no checkpoints to carry on from')
    if not checkpoint:
        return run, run.checkpoints[-1]
    if checkpoint not in run.checkpoints:
        raise TrainerError(f'{name} has no checkpoint {checkpoint}')
    return run, checkpoint


def _actor_damping(key):
    from learning.tasks import task
    try:
        return task(key).capabilities.actor_damping
    except ValueError:            # a run of a task this trainer does not know
        return False


def _damping_refusal():
    """Why actor damping was refused, naming the creatures (a task label's first word) whose tasks support it."""
    from learning.tasks import TASKS
    creatures = [t.label.split()[0].lower() for t in TASKS.values() if t.capabilities.actor_damping]
    if not creatures:
        return 'actor damping is not supported for any task'
    return f"actor damping is supported only for {' and '.join(creatures)} runs"


def start(from_=None, changes=None, root=RL_ROOT, processes=None, launch=None, actor_scale=1.):
    """Start a new run carrying on from a checkpoint (default: the newest run's last), with settings changed."""
    if not 0<actor_scale<=1:
        raise TrainerError('actor scale must be greater than zero and at most one')
    actor_options=['--actor-scale',str(actor_scale)] if actor_scale!=1 else []
    root = Path(root)
    with lifecycle.lock(root):
        runs = _runs(root, processes)
        lifecycle.may_start(runs)
        from learning.tasks import UNTAGGED_TASK, task
        requested_task = (changes or {}).get('task', UNTAGGED_TASK)
        try:
            definition = task(requested_task)
        except ValueError as e:
            raise TrainerError(str(e)) from None
        # A new run of a task with fresh weights, or one that starts from its own policy, like the Campaign Hunter.
        if (definition.capabilities.fresh_weights or definition.start) and from_ in (None, 'new', 'random'):
            from tools.trainer.settings import SETTINGS, Value
            base = {s.dest: Value(s.default, 'recorded') for s in SETTINGS}
            for key, value in dict(task=requested_task, **definition.fresh_defaults).items():
                base[key] = Value(value, 'recorded')
            try:
                values = apply_changes(base, changes)
            except ValueError as e:
                raise TrainerError(str(e)) from None
            name = f'run{next_run_number(root)}'
            out = root/name
            out.mkdir()
            start_policy = str(REPO / definition.start) if definition.start else ''
            command = [sys.executable, '-m', 'tools.ppo_neural', '--start', start_policy, '--out', str(out), *to_argv(values), *actor_options]
            (launch or launch_process)(command, root/f'{name}.log')
            return Started(name, definition.start or 'random', changed(base, values), command)
        source, checkpoint = _source(from_, runs)
        try:
            base = read_settings(source.path)
        except NoSettings as exc:
            raise TrainerError(str(exc)) from None
        source_task = (config_of(source.path) or {}).get('task', UNTAGGED_TASK)
        if actor_scale!=1 and not _actor_damping(source_task):
            raise TrainerError(_damping_refusal())
        if 'task' in (changes or {}) and changes['task'] != source_task:
            raise TrainerError('cannot transfer a checkpoint between creature tasks; start a new task')
        try:
            values = apply_changes(base, changes)
        except ValueError as e:
            raise TrainerError(str(e)) from None
        name = f'run{next_run_number(root)}'
        out = root / name
        out.mkdir()                      # claims the number before the trainer starts writing
        command = [sys.executable, '-m', 'tools.ppo_neural', '--start', str(source.path / f'{checkpoint}.npz'),
                   '--out', str(out), *to_argv(values), *actor_options]
        (launch or launch_process)(command, root / f'{name}.log')
    return Started(name, f'{source.name}/{checkpoint}', changed(base, values), command)


def _last_iteration(run):
    rows = log_rows(run.path)
    return int(rows[-1]['iteration']) if rows else None


def stop(root=RL_ROOT, processes=None, terminate=terminate_tree):
    """End the training run (or a paused or interrupted one): it is finished, its checkpoints stay."""
    processes = list_processes() if processes is None else processes
    run = lifecycle.to_stop(list_runs(root, processes))
    if run.pid is not None:
        terminate(run.pid, processes)
    lifecycle.mark_stopped(run.path, _last_iteration(run))
    return run.name


def pause(root=RL_ROOT, processes=None):
    """Ask the training run to stop once its current iteration is saved; it can be resumed later."""
    run = lifecycle.to_pause(_runs(root, processes))
    config = config_of(run.path)
    if config is not None and not isinstance(config.get('args'), dict):
        raise TrainerError(f'{run.name} was started before pausing existed; it can only be stopped')
    lifecycle.request_pause(run.path)
    return run.name


def resume(run=None, root=RL_ROOT, processes=None, launch=None, changes=None):
    """Carry on a paused or interrupted run in its own folder, as if it had never stopped."""
    root = Path(root)
    with lifecycle.lock(root):
        target = lifecycle.to_resume(_runs(root, processes), run)
        args = (config_of(target.path) or {}).get('args')
        if not isinstance(args, dict):
            raise TrainerError(f'{target.name} was started before pausing existed; it cannot resume')
        values = read_settings(target.path)
        try:
            edit = resume_edits.prepare(target.path, values, changes) if changes else resume_edits.Edit(values)
        except resume_edits.EditRefused as exc:
            raise TrainerError(str(exc)) from None
        command = [sys.executable, '-m', 'tools.ppo_neural', '--start', str(args['start']), '--out',
                   str(target.path), *to_argv(edit.values), '--resume']
        with lifecycle.resuming(target.path, edit.files):
            (launch or launch_process)(command, root / f'{target.name}.log')
    return target.name
