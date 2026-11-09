"""A run's settings: the trainer's options, read back from any run, changed, and turned into a command line.

Runs before run51 did not record every setting. A missing one is recovered from what the run did record (its
rewards summary and episode block), else guessed from the newest run that recorded it, never taken from the
trainer's own defaults, which are far from what recent runs use.
"""
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from learning.tasks import UNTAGGED_TASK
from tools import ppo_neural

NOT_SETTINGS = {'start', 'out', 'resume', 'help', 'actor_scale'}
DEFAULT_ITERATIONS = 1000     # what run52 uses: the guess for a run that never recorded its total

# dest: (group, label). A trainer option missing here still appears, under Training.
LABELS = {
    'bootstrap_epochs': ('Training', 'Teacher training passes (0: task default)'),
    'powerup_reward': ('Rewards', 'Powerup collected'),
    'upgrade_bonus': ('Rewards', 'Hunter power boost, per second'),
    'rules_bounty': ('Rewards', 'Rules hunter destroyed'),
    'compass_reward': ('Rewards', 'Following the enemy compass'),
    'hit_reward': ('Rewards', 'Hit on an enemy'),
    'wall_scale': ('Rewards', 'Wall impact scale'),
    'hit_scale': ('Rewards', 'Enemy hit penalty (points)'),
    'death_penalty': ('Rewards', 'Death'),
    'speed_penalty': ('Rewards', 'Flying below top speed'),
    'still_penalty': ('Rewards', 'Crawling under the still speed'),
    'still_speed': ('Rewards', 'Still speed (share of top speed)'),
    'turn_change_penalty': ('Rewards', 'Starting or reversing a turn'),
    'reversal_penalty': ('Rewards', 'Reversing a turn directly'),
    'turning_penalty': ('Rewards', 'Turning, per second'),
    'shot_price': ('Rewards', 'Unaimed shot'),
    'boredom_penalty': ('Rewards', 'Boredom, per second'),
    'boredom_window': ('Rewards', 'Boredom window (s)'),
    'levels': ('Episodes', 'Levels'),
    'episodes': ('Episodes', 'Episodes per iteration'),
    'horizon': ('Episodes', 'Fighting time (s)'),
    'after_clear': ('Episodes', 'Time after the field is clear (s)'),
    'health': ('Episodes', 'Hunter health (0: indestructible)'),
    'fixed_start': ('Episodes', 'Start beside the player'),
    'decoy_player': ('Episodes', 'Leave the player as a decoy'),
    'panic_score': ('Episodes', 'Panic score (none: off)'),
    'panic_temperature': ('Episodes', 'Panic temperature'),
    'iterations': ('Training', 'Iterations'),
    'lr': ('Training', 'Learning rate'),
    'gamma': ('Training', 'Discount (gamma)'),
    'gae_lambda': ('Training', 'Credit assignment (GAE lambda)'),
    'workers': ('Training', 'Workers'),
}

from learning.inchworm_rewards import DEFAULTS as INCHWORM_REWARD_DEFAULTS
LABELS.update({f'inchworm_{name}': ('Rewards', 'Inchworm ' + name.replace('_', ' ')) for name in INCHWORM_REWARD_DEFAULTS})
from learning.reward_table import tables as _reward_tables
for _table in _reward_tables().values():
    LABELS.update(_table.labels())

# The rewards summary of older runs, by row name: (dest, what the value was multiplied by). Rows are only written
# when non-zero, so a run with a summary but no row for one of these trained with it at 0.
def _reward_rows():
    from experiments.score import Weights
    w = Weights()
    return {'Powerup collected': ('powerup_reward', 1.0), 'Gun upgraded': ('upgrade_bonus', 1.0),
            'Wall impact': ('wall_scale', w.quotient), 'Enemy hit': ('hit_scale', w.hit),
            'Flying below top speed': ('speed_penalty', 1.0),
            'Crawling under the still speed': ('still_penalty', 1.0),
            'Starting or reversing a turn': ('turn_change_penalty', 1.0),
            'Reversing a turn directly': ('reversal_penalty', 1.0), 'Death': ('death_penalty', 1.0),
            'Rules hunter destroyed': ('rules_bounty', 1.0), 'Following the enemy compass': ('compass_reward', 1.0),
            'Hit on an enemy': ('hit_reward', 1.0), 'Turning': ('turning_penalty', 1.0),
            'Unaimed shot': ('shot_price', 1.0), 'Boredom': ('boredom_penalty', 1.0)}

# The episode block of older runs: its key -> dest.
EPISODE_KEYS = {'episodes_per_iteration': 'episodes', 'learning_rate': 'lr', 'horizon_seconds': 'horizon',
                'after_clear_seconds': 'after_clear', 'health': 'health',
                'levels': 'levels', 'panic_score': 'panic_score', 'panic_temperature': 'panic_temperature'}


@dataclass(frozen=True)
class Setting:
    dest: str
    flag: str
    kind: str          # 'int' | 'float' | 'bool' | 'levels' | 'optional_float'
    default: object
    help: str
    group: str
    label: str
    display_scale: float = 1.0
    tasks: tuple = ()     # the task keys whose settings form offers it, see applies_to


@dataclass(frozen=True)
class Value:
    value: object
    source: str        # 'recorded' | 'recovered' | 'guessed'


def _kind(action):
    if action.type is str or isinstance(action.default, str):
        return 'string'
    if action.nargs == 0:
        return 'bool'
    if action.type is ppo_neural.parse_levels:
        return 'levels'
    if action.type is int:
        return 'int'
    return 'optional_float' if action.default is None else 'float'


COMMON_SETTINGS = ('task','iterations','episodes','workers','lr','gamma','gae_lambda','horizon')


def applies_to(dest):
    """The registered tasks a setting applies to: every task for a common setting, else each task listing it in its
    options, and each task whose settings scope is 'unclaimed' unless another task lists it."""
    from learning.tasks import TASKS
    def claimed_by_another(t):
        return any(dest in other.options for other in TASKS.values() if other.key != t.key)
    return tuple(t.key for t in TASKS.values() if dest in COMMON_SETTINGS or dest in t.options or
                 (t.capabilities.settings_scope == 'unclaimed' and not claimed_by_another(t)))


def scoped_settings():
    """SETTINGS, each with the tasks it applies to as the registry stands now."""
    return [replace(s, tasks=applies_to(s.dest)) for s in SETTINGS]


def _settings():
    from experiments.score import Weights
    out = []
    for action in ppo_neural.build_parser()._actions:
        if action.dest in NOT_SETTINGS:
            continue
        group, label = LABELS.get(action.dest, ('Training', action.dest.replace('_', ' ').capitalize()))
        default = list(action.default) if isinstance(action.default, tuple) else action.default
        help_text = ('Reward points lost per enemy collision or projectile hit; 0 disables this penalty.'
                     if action.dest == 'hit_scale' else action.help or '')
        out.append(Setting(action.dest, action.option_strings[0], _kind(action), default, help_text,
                           group, label, Weights().hit if action.dest == 'hit_scale' else 1.0,
                           applies_to(action.dest)))
    return out


SETTINGS = _settings()
BY_DEST = {s.dest: s for s in SETTINGS}


def config_of(run_dir):
    """A run's config.json, or None when it has none or it does not read."""
    try:
        config = json.loads((Path(run_dir) / 'config.json').read_text())
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _recorded(config):
    """What a run's config records for certain: its arguments, else its shaping block."""
    if not config:
        return {}
    if isinstance(config.get('args'), dict):
        return {k: v for k, v in config['args'].items() if k in BY_DEST}
    out = {}
    for key, value in (config.get('shaping') or {}).items():
        if key == 'random_start':
            out['fixed_start'] = not value
        elif key == 'pursue_hunter':
            out['decoy_player'] = not value
        elif key in BY_DEST:
            out[key] = value
    return out


def _recovered(config):
    """What can be worked out from an older run's episode block and rewards summary."""
    if not config:
        return {}
    from learning.dashboard import declaration
    try:
        out = declaration(config.get('task', UNTAGGED_TASK)).recovered_settings(config)
    except ValueError:   # a task this trainer no longer knows
        out = {}
    for key, dest in EPISODE_KEYS.items():
        if key in (config.get('episode') or {}):
            out[dest] = config['episode'][key]
    rewards = config.get('rewards')
    if isinstance(rewards, list):
        rows = {row.get('name'): row.get('value') for row in rewards if isinstance(row, dict)}
        for name, (dest, factor) in _reward_rows().items():
            out[dest] = rows.get(name, 0.0) / factor
    return out


def _number(run_dir):
    match = re.fullmatch(r'run(\d+)', Path(run_dir).name)
    return int(match.group(1)) if match else -1


def read_settings(run_dir, all_runs):
    """Every setting for the run in `run_dir`, each with where its value came from."""
    config = config_of(run_dir)
    recorded, recovered = _recorded(config), _recovered(config)
    others = [{**_recovered(c), **_recorded(c)} for c in
              (config_of(r) for r in sorted(all_runs, key=_number, reverse=True) if Path(r) != Path(run_dir))
              if (c or {}).get('task', UNTAGGED_TASK) == (config or {}).get('task', UNTAGGED_TASK)]
    out = {}
    for s in SETTINGS:
        if s.dest in recorded:
            out[s.dest] = Value(_normal(s, recorded[s.dest]), 'recorded')
        elif s.dest in recovered:
            out[s.dest] = Value(_normal(s, recovered[s.dest]), 'recovered')
        elif s.dest in ('gamma', 'gae_lambda') or (s.dest == 'bootstrap_epochs' and config is not None):
            # Older runs used these hardcoded values; do not borrow a newer run's experiment.
            out[s.dest] = Value(s.default, 'recovered')
        else:
            guess = next((o[s.dest] for o in others if s.dest in o), None)
            if guess is not None or any(s.dest in o for o in others):
                out[s.dest] = Value(_normal(s, guess), 'guessed')
            elif s.dest == 'iterations':
                out[s.dest] = Value(DEFAULT_ITERATIONS, 'guessed')
            else:
                out[s.dest] = Value(s.default, 'guessed')
    return out


def _normal(setting, value):
    if setting.kind == 'levels':
        return list(value) if value is not None else None
    if setting.kind == 'int' and value is not None:
        return int(value)
    if setting.kind in ('float', 'optional_float') and value is not None:
        return float(value)
    return value


def find(name):
    """The setting called `name`: its dest or its flag, with or without dashes."""
    dest = name.strip().lstrip('-').replace('-', '_')
    if dest not in BY_DEST:
        raise ValueError(f'unknown setting: {name}')
    return BY_DEST[dest]


def parse_value(setting, value):
    """`value` (text from a person, or already typed) as the setting's type."""
    def bad(why):
        return ValueError(f'{setting.dest}: {value!r} {why}')
    if setting.kind == 'string':
        text = str(value)
        if setting.dest == 'task':
            from learning.tasks import task
            task(text)
        return text
    if setting.kind == 'bool':
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ('true', 'yes', 'on', '1'):
            return True
        if text in ('false', 'no', 'off', '0'):
            return False
        raise bad('is not true or false')
    if setting.kind == 'levels':
        if isinstance(value, (list, tuple)):
            levels = [int(v) for v in value]
        else:
            try:
                levels = list(ppo_neural.parse_levels(str(value)))
            except ValueError:
                raise bad('is not a list of levels like 1-20 or 4,7,13') from None
        if not levels or any(level < 1 for level in levels):
            raise bad('is not a list of levels like 1-20 or 4,7,13')
        return levels
    if setting.kind == 'optional_float' and (value is None or str(value).strip().lower() in ('none', 'off', '')):
        return None
    if setting.kind == 'int':
        if isinstance(value, bool):
            raise bad('is not a whole number')
        if isinstance(value, float) and value.is_integer():
            return int(value)
        try:
            return int(str(value).strip())
        except ValueError:
            raise bad('is not a whole number') from None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise bad('is not a number') from None
    if setting.dest in ('gamma', 'gae_lambda') and not 0 <= number <= 1:
        raise bad('must be between 0 and 1')
    return number


def apply_changes(base, changes):
    """`base` with `changes` ({name: value}) applied; changed values count as recorded."""
    out = dict(base)
    for name, value in (changes or {}).items():
        setting = find(name)
        out[setting.dest] = Value(parse_value(setting, value), 'recorded')
    return out


def changed(base, new):
    """(dest, old, new) for each setting whose value differs, in name order."""
    return [(dest, base[dest].value, new[dest].value) for dest in sorted(new)
            if dest in base and base[dest].value != new[dest].value]


def format_levels(levels):
    """[1, 2, 3, 7] as '1-3,7'."""
    parts, levels = [], sorted(set(levels))
    i = 0
    while i < len(levels):
        j = i
        while j + 1 < len(levels) and levels[j + 1] == levels[j] + 1:
            j += 1
        parts.append(str(levels[i]) if i == j else f'{levels[i]}-{levels[j]}')
        i = j + 1
    return ','.join(parts)


def to_argv(values):
    """Every setting as trainer arguments, explicit even when it matches the trainer's default."""
    argv = []
    for s in SETTINGS:
        if s.dest not in values:
            continue
        value = values[s.dest].value
        if s.kind == 'bool':
            argv += [s.flag] if value else []
        elif value is None:
            continue
        elif s.kind == 'levels':
            argv += [s.flag, format_levels(value)]
        else:
            argv += [s.flag, str(value) if s.kind in ('int', 'string') else repr(value)]
    return argv
