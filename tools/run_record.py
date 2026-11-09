"""A run's settings as their only copy: its config.json's recorded arguments, and everything derived from them.

A run records its settings once, as `args`. What it trains with (a hunter-loop run's episode block and shaping, its
rewards through its task's reward table; a creature run's horizon) is derived from them when needed, by the code that
trains it, so the dashboard, the trainer, replays and resumes all read the same values, and changing a setting is
changing `args`.

Beside its arguments a run records only what is not a setting:
  rules     the rules a hunter run keeps for its whole life (tools.ppo_neural.RUN_RULES: its combat ruleset,
            weapon tuning, curved walls and whether it flies random mazes), fixed when it starts (random mazes may
            be taken up on a resume), whatever the code later defaults to;
  history   the iterations at which a hunter run took up something new (random mazes, a level cycle);
  manifest  a creature run's rules snapshot (learning.run), which resume and replay check the code against.

Runs recorded before `rules` and `history` existed kept them in their shaping and episode blocks, and are read from
there. A run without recorded arguments (from before run51) has no settings: it can be viewed, not used.
"""
import json
from pathlib import Path
from types import SimpleNamespace

HISTORY = ('random_maze_from_iteration', 'level_cycle_from_iteration', 'rules_share')


class NoSettings(ValueError):
    """The run records no settings (its config predates recorded arguments, or it has none)."""


def load(run):
    """The run's config.json, or None when it has none or it does not read."""
    try:
        config = json.loads((Path(run) / 'config.json').read_text())
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def task_of(config):
    from learning.tasks import UNTAGGED_TASK
    return (config or {}).get('task', UNTAGGED_TASK)


def has_settings(config):
    return isinstance((config or {}).get('args'), dict)


def arguments(config):
    """The run's settings as the trainer's parser would give them: its recorded arguments, and an option added since
    it started at that option's default. Raises NoSettings for a run that records none."""
    if not has_settings(config):
        raise NoSettings('this run records no settings: it can be viewed but not resumed or used as a settings source')
    from tools import ppo_neural
    args = config['args']
    parsed = ppo_neural.build_parser().parse_args(['--task', args.get('task', task_of(config))])
    return SimpleNamespace(**{**vars(parsed), **args})


def is_ppo(config):
    """Whether the run trains in the hunter's loop (tools.ppo_neural) rather than learning.run."""
    from tools import ppo_neural
    return task_of(config) in ppo_neural.PPO_TASKS


def rules(config):
    """The rules a hunter run keeps for life, tools.ppo_neural.RUN_RULES each: as it recorded them, and for one it
    is older than, the value it flew with before that rule existed."""
    from tools import ppo_neural
    config = config or {}
    recorded = config.get('rules') if isinstance(config.get('rules'), dict) else (config.get('shaping') or {})
    return {key: recorded.get(key, before) for key, before in ppo_neural.RUN_RULES.items()}


def history(config):
    """When a hunter-loop run took up something new: {HISTORY key: iteration} (and a historical rules_share)."""
    config = config or {}
    if isinstance(config.get('history'), dict):
        return dict(config['history'])
    episode = config.get('episode') or {}
    return {key: episode[key] for key in HISTORY if key in episode}


def episode(config):
    """The run's episode block: for a hunter-loop run, tools.ppo_neural.run_config()'s, with its history; for a
    creature run, its horizon."""
    args = arguments(config)
    if is_ppo(config):
        from tools import ppo_neural
        return {**ppo_neural.run_config(args)['episode'], **history(config)}
    return {'horizon_seconds': args.horizon}


def shaping(config, iteration=None):
    """What the run's episodes are flown with: for a hunter-loop run, tools.ppo_neural.shaping_from() its arguments,
    a hunter run's with the rules it keeps for life (as of `iteration`, when given: before it took up random mazes it
    flew without them); for a creature run, its manifest's rewards."""
    if not is_ppo(config):
        return dict((config.get('manifest') or {}).get('rewards') or {})
    from tools import ppo_neural
    out = ppo_neural.shaping_from(arguments(config))
    if task_of(config) != 'hunter':
        return out
    out.update(rules(config))
    if iteration is not None and iteration < history(config).get('random_maze_from_iteration', 0):
        out['random_maze'] = False
    return out


def start(config):
    """The policy the run started from."""
    config = config or {}
    return (config['args'].get('start') if has_settings(config) else None) or config.get('start')


DERIVED = ('shaping', 'rewards', 'episode', 'start')   # blocks older runs recorded beside their arguments


def migrated(config, lifetime_rules=None, run_history=None):
    """The run's config as a run records it now: its arguments and what is not a setting (a hunter run's `rules`,
    a hunter-loop run's `history` when it has any, a creature run's manifest), without the blocks older runs derived
    from their arguments. `lifetime_rules` and `run_history` default to what the config records."""
    out = {key: value for key, value in config.items() if key not in DERIVED}
    out['task'] = task_of(config)
    if is_ppo(config):
        if out['task'] == 'hunter':
            out['rules'] = rules(config) if lifetime_rules is None else dict(lifetime_rules)
        else:
            out.pop('rules', None)
        taken = history(config) if run_history is None else run_history
        if taken:
            out['history'] = dict(taken)
        else:
            out.pop('history', None)
    return out
