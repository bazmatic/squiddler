"""A standard hook for training-log numbers: an episode records a named stat where it happens, and the trainer
averages every recorded stat into the iteration's log row, where the dashboard charts it.

    from learning.episode_stats import record
    record('wall strikes')            # one more, in the episode being flown
    record('bullets fired', 3)

No other change is needed: the log row gains a `wall_strikes` key and the dashboard charts it under Other. Declare a
Metric (the task's dashboard module) only to give it a nicer title, a section, a direction or digits.

`record` does nothing outside an episode, so game code may call it freely. A task's rollout wraps each episode in
`collecting()`; the trainer then averages with `recorded_means`.
"""
import contextlib
import contextvars
import re
from typing import Dict, Iterable, Iterator, Mapping, Optional

_active: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar('episode_stats', default=None)


def key_for(name: str) -> str:
    """The log key for a stat name: 'Wall strikes' -> 'wall_strikes'."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def record(name: str, amount: float = 1.0) -> None:
    """Add `amount` to the named stat of the episode being flown (nothing when none is)."""
    totals = _active.get()
    if totals is not None:
        key = key_for(name)
        totals[key] = totals.get(key, 0.0) + amount


@contextlib.contextmanager
def collecting() -> Iterator[Dict[str, float]]:
    """Collect the stats recorded inside the block into the yielded dict (name -> total for the episode)."""
    totals: Dict[str, float] = {}
    token = _active.set(totals)
    try:
        yield totals
    finally:
        _active.reset(token)


def recorded_means(episodes: Iterable[Optional[Mapping[str, float]]], always: Iterable[str] = ()) -> Dict[str, float]:
    """The mean per episode of every stat any episode recorded; an episode that did not record one counts as 0.
    The `always` keys are logged even when no episode recorded them (as 0), so a chart that should always show does."""
    episodes = [e or {} for e in episodes]
    keys = sorted({k for e in episodes for k in e} | set(always))
    return {k: sum(e.get(k, 0.0) for e in episodes) / len(episodes) for k in keys} if episodes else {}
