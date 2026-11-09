"""A task's rewards declared once, for any task: each term's key, default, title, help and the episode event it pays on.

Everything a trainer needs about a task's rewards derives from its RewardTable: the shaping values and their
defaults, the trainer flags, the settings-form labels, the dashboard's reward rules, the checks a paused run's edited
values pass, and the per-decision events-times-weights sum, which
matches each weight to its event by name, never by position. Adding a reward is one RewardTerm plus the line in the
task's episode that counts its event.

No pygame and no numpy at import: the trainer's server reads these tables.
"""
import math
from dataclasses import dataclass, make_dataclass, field
from enum import Enum


class RewardKind(Enum):
    """What a term is. A reward or a penalty pays on an event; a shaping parameter (a distance, a share) tunes other
    terms, and a setting is a run setting kept beside the rewards (a lesson, a limit): neither pays."""
    REWARD = 'reward'
    PENALTY = 'penalty'
    SHAPING = 'shaping'
    SETTING = 'setting'

    @property
    def pays(self):
        return self in (RewardKind.REWARD, RewardKind.PENALTY)


@dataclass(frozen=True)
class RewardTerm:
    """key: the setting (and trainer flag, with prefix) holding the points; kind: its RewardKind, always declared;
    default: its value for a new run; title: its name on the settings form and the dashboard; help: what the points
    are paid for; event: the name of the per-decision episode count a reward or penalty multiplies (none for a
    shaping parameter or a setting). stages: the curriculum stages it applies in (empty: every stage). manifest: the
    (block, field) where the task records its value in a run's manifest, if it does. rule: its key on the dashboard,
    when that is not its own key.

    fixed: the term is not a run setting (no flag, no label): it always pays its default. unit: the points one unit of
    the setting is worth, for a setting that scales a fixed weight (1: the setting is the points).
    """
    key: str
    kind: RewardKind
    default: float
    title: str
    help: str
    event: str | None = None
    stages: tuple = ()
    manifest: tuple | None = None
    rule: str | None = None
    fixed: bool = False
    unit: float = 1.

    def __post_init__(self):
        if not isinstance(self.kind, RewardKind):
            raise TypeError(f'reward term {self.key}: kind must be a RewardKind, not {self.kind!r}')
        if (self.event is not None) != self.kind.pays:
            raise ValueError(f'reward term {self.key}: a {self.kind.value} term '
                             + ('pays on no event' if self.event is not None else 'needs the event it pays on'))
        if self.fixed and not self.kind.pays:
            raise ValueError(f'reward term {self.key}: only a reward or a penalty can be fixed')

    @property
    def rule_key(self):
        """Its key on the dashboard."""
        return self.rule or self.key


    @property
    def pays(self):
        return self.kind.pays


class RewardTable:
    """The reward terms of one task. prefix: put before each key to give its setting and flag (--<prefix><key>, with
    dashes); help_prefix: put before each term's help on the trainer's command line.

    signed: a penalty's points are negative and added, as a Campaign Hunter's are (True); or they are its size and
    subtracted, as a Jelly's are (False). rule_defaults: whether the dashboard may show a run that does not record a
    term at the term's default (False when older runs trained with other values than today's defaults)."""

    def __init__(self, terms, prefix='', help_prefix='', signed=True, rule_defaults=True):
        self.terms = tuple(terms)
        self.prefix, self.help_prefix, self.signed = prefix, help_prefix, signed
        self.rule_defaults = rule_defaults
        self.paying = tuple(t for t in self.terms if t.pays)
        self.settable = tuple(t for t in self.terms if not t.fixed)    # the terms that are run settings
        for what, names in (('key', [t.key for t in self.terms]), ('event', [t.event for t in self.paying])):
            repeated = sorted({n for n in names if names.count(n) > 1})
            if repeated:
                raise ValueError(f'reward table: each term needs its own {what}; repeated: {", ".join(repeated)}')
        self.by_key = {t.key: t for t in self.settable}

    # --- the values ---
    @property
    def defaults(self):
        """{key: default} for the terms that are run settings."""
        return {t.key: t.default for t in self.settable}

    @property
    def events(self):
        """The event names, one per paying term, in table order: an episode's per-decision rows count these."""
        return tuple(t.event for t in self.paying)

    @property
    def settings(self):
        """The trainer settings (argparse dests) holding the terms' points."""
        return tuple(self.prefix + t.key for t in self.settable)

    def shaping_class(self, name, doc=None):
        """A dataclass with one field per setting at its default, for code that passes the points around as one value."""
        cls = make_dataclass(name, [(t.key, float, field(default=t.default)) for t in self.settable])
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
        return self.validate({t.key: getattr(args, self.prefix + t.key, t.default) for t in self.settable})

    # --- the trainer and its page ---
    def add_arguments(self, parser):
        """One float flag per setting: --<prefix><key> with underscores as dashes."""
        for t in self.settable:
            parser.add_argument('--' + (self.prefix + t.key).replace('_', '-'), type=float, default=t.default,
                                help=self.help_prefix + self.described(t))

    def described(self, term):
        """The term's help as the trainer's flags give it: a signed table's penalty says that its points are
        negative."""
        return term.help + (' (negative: a penalty)' if self.signed and term.kind is RewardKind.PENALTY else '')

    def labels(self, group='Rewards'):
        """The settings form's (group, label) for each setting."""
        return {self.prefix + t.key: (group, t.title) for t in self.settable}

    def points(self, term, values):
        """The term's points at `values` (validated settings): its default when fixed, else its unit times its
        setting."""
        return term.default if term.fixed else term.unit * values[term.key]

    def weight(self, term, values):
        """What one count of the term's event pays at `values`: its points, or less its size for an unsigned table's
        penalty."""
        points = self.points(term, values)
        return -points if term.kind is RewardKind.PENALTY and not self.signed else points

    def manifest_fields(self):
        """{key: (block, field, sign)} for the terms the task records in its manifest. sign: what the setting is
        multiplied by there (an unsigned table records a penalty as a negative reward)."""
        return {t.key: (*t.manifest, -1 if t.kind is RewardKind.PENALTY and not self.signed else 1)
                for t in self.settable if t.manifest}

    def reward_rules(self):
        """The dashboard's reward rules (a setting is not one). A penalty's value is its size: the page draws it with a
        minus sign."""
        from learning.dashboard import RewardRule
        defaults = self.defaults
        return tuple(RewardRule(t.rule_key, t.title, t.help, t.kind.value, tuple(t.stages),
                                '' if t.fixed else self.prefix + t.key,
                                abs(self.points(t, defaults)) if self.rule_defaults else None)
                     for t in self.terms if t.kind is not RewardKind.SETTING)

    # --- the episode ---
    def row(self):
        """One decision's event counts, all 0: the episode adds to them by event name."""
        return dict.fromkeys(self.events, 0)

    def pay(self, rows, values):
        """Per decision (or per agent), the points its event counts earn: the sum over paying terms of each term's
        weight times its event's count, matched by name. `rows` are dicts of event counts (as row() makes); a row
        missing a term's event, or holding an event no term pays on, is an error, so a count and its points cannot
        drift apart. `values` may hold more than the table's settings (a task's own extras): only the table's are read."""
        import numpy as np
        events = set(self.events)
        for i, row in enumerate(rows):
            if set(row) != events:
                missing, extra = sorted(events - set(row)), sorted(set(row) - events)
                raise ValueError(f'decision {i}: event counts do not match the reward table '
                                 f'(missing {missing}, unpaid {extra})')
        return self.pay_columns({e: np.asarray([row[e] for row in rows], dtype=np.float64) for e in self.events},
                                values, len(rows))

    def pay_columns(self, columns, values, n):
        """pay() for `n` decisions whose event counts come as columns: {event: array of n counts}, one per event."""
        import numpy as np
        if set(columns) != set(self.events):
            missing, extra = sorted(set(self.events) - set(columns)), sorted(set(columns) - set(self.events))
            raise ValueError(f'event counts do not match the reward table (missing {missing}, unpaid {extra})')
        values = self.validate({key: value for key, value in values.items() if key in self.by_key})
        rewards = np.zeros(n, dtype=np.float64)
        for t in self.paying:
            rewards = rewards + self.weight(t, values) * np.asarray(columns[t.event], dtype=np.float64)
        return rewards


def tables():
    """{task key: its RewardTable} for every registered task that declares its rewards in one."""
    from learning.tasks import TASKS
    return {key: t.reward_table() for key, t in TASKS.items() if t.rewards}
