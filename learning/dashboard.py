"""What the trainer dashboard shows for each task: shared metrics plus the task's own declaration module."""
from dataclasses import asdict, dataclass
import importlib
from types import SimpleNamespace
from typing import Iterable

from learning.tasks import TASKS, task


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    sub: str = ''
    unit: str = ''
    section: str = 'Training'
    higher_is_better: bool | None = None
    optional: bool = False
    aliases: tuple[str, ...] = ()
    wide: bool = False
    short: str = ''    # its stat-strip label; empty uses the title
    digits: int = 2    # decimals in the stat strip


@dataclass(frozen=True)
class Layout:
    """Ordering overrides; empty means declaration order. strip: the metric keys whose latest value the stat strip
    shows, between its shared cells."""
    sections: tuple[str, ...] = ()
    strip: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardRule:
    """One term of a task's training signal. setting: the run setting holding its value (a key of config.json's args
    unless the task's recorded(config) says otherwise); default: its value for a run whose config does not record it
    (None when no value can be assumed). stages: the curriculum stages in which it applies; empty means every stage.
    kind 'shaping' is a parameter of other terms (a distance, a share) rather than a reward or penalty itself."""
    key: str
    title: str
    description: str
    kind: str
    stages: tuple[int, ...]
    setting: str
    default: float | None = None

    def __post_init__(self):
        if self.kind not in ('reward', 'penalty', 'shaping'):
            raise ValueError(f'reward rule {self.key}: kind must be reward, penalty or shaping, not {self.kind!r}')


SHARED_METRICS = (
    Metric('mean_return', 'Training return', 'Higher is better. What training maximises. Only comparable within a run.',
           higher_is_better=True, wide=True),
    # Their own section, so they come after every task's charts, where the page has always drawn them.
    Metric('kl', 'Distance from the starting policy (KL)', 'How far the policy has moved from where the run started.',
           section='Policy'),
    Metric('entropy', 'Policy entropy', 'Randomness of the controls. Near zero means it has stopped exploring.',
           section='Policy'),
)

# Log keys every trainer writes that are bookkeeping, timing, gate flags or optimiser internals rather than something to
# chart. A task hides its own such keys with HIDDEN in its dashboard module.
HIDDEN_KEYS = frozenset({'iteration', 'rollout_s', 'update_s', 'elapsed_s', 'policy', 'team_size',
                         'selection_gate_passed', 'policy_loss', 'value_loss', 'clip_fraction'})


def _recorded_args(rules):
    """The default recorded(config): each rule's setting as config.json's args recorded it."""
    def recorded(config):
        args = (config or {}).get('args') or {}
        return {r.key: args[r.setting] for r in rules if isinstance(args.get(r.setting), (int, float))
                and not isinstance(args.get(r.setting), bool)}
    return recorded


def declaration(task_key: str) -> SimpleNamespace:
    """The task's declared METRICS, LAYOUT, HIDDEN, REWARD_RULES, cards(config) (structured cards: title, rows of
    label/value/description, note), recorded(config) (the rule values a run's config.json records, by rule key) and
    stage(row) (the curriculum stage a train.log row reports, or None); empty ones for a task with no module."""
    spec = task(task_key)
    module = importlib.import_module(spec.dashboard) if spec.dashboard else None
    rules = tuple(getattr(module, 'REWARD_RULES', ()))
    return SimpleNamespace(
        METRICS=tuple(getattr(module, 'METRICS', ())),
        LAYOUT=getattr(module, 'LAYOUT', Layout()),
        HIDDEN=tuple(getattr(module, 'HIDDEN', ())),
        REWARD_RULES=rules,
        cards=getattr(module, 'cards', lambda config: []),
        recorded=getattr(module, 'recorded', _recorded_args(rules)),
        stage=getattr(module, 'stage', lambda row: None))


def resolve(task_key: str) -> dict:
    spec, declared = task(task_key), declaration(task_key)
    return dict(key=spec.key, label=spec.label, capabilities=asdict(spec.capabilities),
                metrics=[dict(asdict(m), aliases=list(m.aliases)) for m in SHARED_METRICS + declared.METRICS],
                sections=list(declared.LAYOUT.sections), strip=list(declared.LAYOUT.strip),
                rewards=[dict(asdict(r), stages=list(r.stages)) for r in declared.REWARD_RULES], hidden=sorted(HIDDEN_KEYS | set(declared.HIDDEN)))


def rewards(task_key: str, config: dict, row: dict) -> dict:
    """What the viewed run trained with: its task's rules with the values its own config records, the stage its last
    log row reports, and the task's cards. A value the config lacks is reported unrecorded, with the rule's default. A
    rule is active when it applies at that stage (every stage when the stage is unknown) and its value is set and not
    zero. A task this trainer does not know has no rules. episodes: the run's episodes per iteration, for the strip's
    game time."""
    try:
        declared = declaration(task_key)
    except ValueError:
        return dict(stage=None, rules=[], cards=[], episodes=None)
    config = config if isinstance(config, dict) else {}
    stage = declared.stage(row or {})
    try:
        values = declared.recorded(config)
    except (AttributeError, KeyError, TypeError, ValueError):   # a config.json too old or odd to read: nothing recorded
        values = {}
    rules = []
    for r in declared.REWARD_RULES:
        recorded = r.key in values
        value = values[r.key] if recorded else r.default
        applies = stage is None or not r.stages or stage in r.stages
        rules.append(dict(key=r.key, title=r.title, description=r.description, kind=r.kind, stages=list(r.stages),
                          active=bool(applies and value), value=value, recorded=recorded))
    try:
        cards = declared.cards(config)
    except (AttributeError, KeyError, TypeError, ValueError):
        cards = []
    args = config.get('args')
    episodes = args.get('episodes') if isinstance(args, dict) else None
    return dict(stage=stage, rules=rules, cards=cards, episodes=episodes)


def resolve_all() -> list[dict]:
    return [resolve(key) for key in TASKS]


def charted(task_key: str, keys: Iterable[str]) -> list[dict]:
    """The resolved metrics the page draws for a run whose last log row has these keys, in drawing order: a metric is
    drawn iff its key or an alias is present; metrics group by section, sections in the layout's order and then in
    order of first appearance, declaration order within a section."""
    resolved, present = resolve(task_key), set(keys)
    order = list(resolved['sections'])
    order += [m['section'] for m in resolved['metrics'] if m['section'] not in order]
    drawn = [m for m in resolved['metrics'] if present & {m['key'], *m['aliases']}]
    return sorted(drawn, key=lambda m: order.index(m['section']))   # stable: declaration order within a section


def unclaimed(task_key: str, keys: Iterable[str]) -> list[str]:
    declared = declaration(task_key)
    claimed = set(HIDDEN_KEYS) | set(declared.HIDDEN)
    for metric in SHARED_METRICS + declared.METRICS:
        claimed.add(metric.key)
        claimed.update(metric.aliases)
    return sorted(set(keys) - claimed)


def number(value) -> str:
    """A recorded number as the page has always printed it (JavaScript's way): 60.0 as 60, 0.25 as 0.25."""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
