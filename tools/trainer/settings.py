"""A run's settings: the trainer's options, read back from any run, changed, and turned into a command line.

A run's settings are its recorded arguments, their only copy (tools.run_record). An option added since the run
started reads as the trainer's default for it. A run that records no arguments (before run51) has no settings.
"""
from dataclasses import dataclass, replace
from pathlib import Path

from tools import ppo_neural, run_record

NOT_SETTINGS = {'start', 'out', 'resume', 'help', 'actor_scale'}

# dest: (group, label). A trainer option missing here still appears, under Training.
LABELS = {
    'bootstrap_epochs': ('Training', 'Teacher training passes (0: task default)'),
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

from learning.reward_table import tables as _reward_tables
# A reward setting's points per unit (a scale on a fixed weight): the settings form shows the setting in points.
UNITS = {}
for _table in _reward_tables().values():
    LABELS.update(_table.labels())
    UNITS.update({_table.prefix + t.key: t.unit for t in _table.settable})


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
    source: str        # 'recorded' | 'default' (an option added since the run started)


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
    out = []
    for action in ppo_neural.build_parser()._actions:
        if action.dest in NOT_SETTINGS:
            continue
        group, label = LABELS.get(action.dest, ('Training', action.dest.replace('_', ' ').capitalize()))
        default = list(action.default) if isinstance(action.default, tuple) else action.default
        out.append(Setting(action.dest, action.option_strings[0], _kind(action), default, action.help or '',
                           group, label, UNITS.get(action.dest, 1.0), applies_to(action.dest)))
    return out


SETTINGS = _settings()
BY_DEST = {s.dest: s for s in SETTINGS}


def config_of(run_dir):
    """A run's config.json, or None when it has none or it does not read."""
    return run_record.load(run_dir)


class NoSettings(run_record.NoSettings):
    """The run records no settings."""


def read_settings(run_dir):
    """Every setting for the run in `run_dir`, each with where its value came from. Raises NoSettings for a run that
    records none."""
    config = config_of(run_dir)
    try:
        args = vars(run_record.arguments(config))
    except run_record.NoSettings:
        raise NoSettings(f'{Path(run_dir).name} records no settings: it can be viewed, but not resumed or used as a '
                         'settings source') from None
    recorded = config['args']
    return {s.dest: Value(_normal(s, args[s.dest]), 'recorded' if s.dest in recorded else 'default')
            for s in SETTINGS}


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
