"""A task's rewards declared once, for any task: each term's key, default, title, help and the episode event it pays on.

Everything a trainer needs about a task's rewards derives from its RewardTable: the shaping values and their
defaults, the trainer flags, the settings-form labels, the dashboard's reward rules, the checks a paused run's edited
values pass, the rewards summary recorded in a run's config.json, and the per-decision events-times-weights sum, which
matches each weight to its event by name, never by position. Adding a reward is one RewardTerm plus the line in the
task's episode that counts its event.

No pygame and no numpy at import: the trainer's server reads these tables.
"""
import math
from dataclasses import dataclass, make_dataclass, field


@dataclass(frozen=True)
class RewardTerm:
    """key: the setting (and trainer flag, with prefix) holding the points; default: its value for a new run; title:
    its name on the settings form and the dashboard; help: what the points are paid for; event: the name of the
    per-decision episode count it multiplies. A negative default makes it a penalty."""
    key: str
    default: float
    title: str
    help: str
    event: str

    @property
    def kind(self):
        return 'penalty' if self.default < 0 or '(negative' in self.help else 'reward'


class RewardTable:
    """The reward terms of one task. prefix: put before each key to give its setting and flag (--<prefix><key>, with
    dashes); help_prefix: put before each term's help on the trainer's command line."""

    def __init__(self, terms, prefix='', help_prefix=''):
        self.terms = tuple(terms)
        self.prefix, self.help_prefix = prefix, help_prefix
        for what, names in (('key', [t.key for t in self.terms]), ('event', [t.event for t in self.terms])):
            repeated = sorted({n for n in names if names.count(n) > 1})
            if repeated:
                raise ValueError(f'reward table: each term needs its own {what}; repeated: {", ".join(repeated)}')
        self.by_key = {t.key: t for t in self.terms}

    # --- the values ---
    @property
    def defaults(self):
        return {t.key: t.default for t in self.terms}

    @property
    def events(self):
        """The event names, one per term, in table order: an episode's per-decision rows count these."""
        return tuple(t.event for t in self.terms)

    @property
    def settings(self):
        """The trainer settings (argparse dests) holding the terms' points."""
        return tuple(self.prefix + t.key for t in self.terms)

    def shaping_class(self, name, doc=None):
        """A dataclass with one field per term at its default, for code that passes the points around as one value."""
        cls = make_dataclass(name, [(t.key, float, field(default=t.default)) for t in self.terms])
        cls.__doc__ = doc or cls.__doc__
        return cls

    def validate(self, values):
        """`values` ({key: points}, the rest at their defaults) checked: known keys and finite numbers."""
        unknown = sorted(set(values) - set(self.by_key))
        if unknown:
            raise ValueError(f'unknown reward setting: {", ".join(unknown)}')
        result = dict(self.defaults, **values)
        for key, value in result.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{self.prefix}{key}: reward points must be a finite number, not {value!r}')
        return {key: float(value) for key, value in result.items()}

    def from_args(self, args):
        """The points a run's arguments (an argparse namespace, or anything with the settings as attributes) give."""
        return self.validate({t.key: getattr(args, self.prefix + t.key, t.default) for t in self.terms})

    # --- the trainer and its page ---
    def add_arguments(self, parser):
        """One float flag per term: --<prefix><key> with underscores as dashes."""
        for t in self.terms:
            parser.add_argument('--' + (self.prefix + t.key).replace('_', '-'), type=float, default=t.default,
                                help=self.help_prefix + t.help)

    def labels(self, group='Rewards'):
        """The settings form's (group, label) for each setting."""
        return {self.prefix + t.key: (group, t.title) for t in self.terms}

    def reward_rules(self):
        """The dashboard's reward rules. A penalty's value is its size: the page draws it with a minus sign, and its
        help drops the "(negative: a penalty)" note."""
        from learning.dashboard import RewardRule
        return tuple(RewardRule(t.key, t.title, t.help.split(' (negative')[0], t.kind, (), self.prefix + t.key,
                                abs(t.default)) for t in self.terms)

    def summary(self, values):
        """The rewards rows recorded in a run's config.json: name, sign, size and unit of each term that is not 0. A
        row's name is its key in words ("own_kill": "Own kill"), as runs have always recorded it."""
        values = self.validate(values)
        return [{'name': t.key.replace('_', ' ').capitalize(), 'sign': '+' if values[t.key] > 0 else '-',
                 'value': round(abs(values[t.key]), 4), 'unit': t.help} for t in self.terms if values[t.key]]

    # --- the episode ---
    def row(self):
        """One decision's event counts, all 0: the episode adds to them by event name."""
        return dict.fromkeys(self.events, 0)

    def pay(self, rows, values):
        """Per decision, the points its event counts earn: the sum over terms of each term's points times its event's
        count, matched by name. `rows` are dicts of event counts (as row() makes); a row missing a term's event, or
        holding an event no term pays on, is an error, so a count and its points cannot drift apart."""
        import numpy as np
        values = self.validate(values)
        events = set(self.events)
        rewards = np.zeros(len(rows), dtype=np.float64)
        for i, row in enumerate(rows):
            if set(row) != events:
                missing, extra = sorted(events - set(row)), sorted(set(row) - events)
                raise ValueError(f'decision {i}: event counts do not match the reward table '
                                 f'(missing {missing}, unpaid {extra})')
        for t in self.terms:
            rewards += values[t.key] * np.asarray([row[t.event] for row in rows], dtype=np.float64)
        return rewards


def tables():
    """{task key: its RewardTable} for every registered task that declares its rewards in one."""
    from learning.tasks import TASKS
    return {key: t.reward_table() for key, t in TASKS.items() if t.rewards}
