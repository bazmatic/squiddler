"""The Campaign Hunter's evaluation: per-episode rows and a summary that counts a cleared exit only when the ship left
by the exit with no enemy left."""
import pytest

from hunter.neural.loader import DEFAULT_PATH as HUNTER_POLICY_PATH
from tools.campaign_eval import evaluate, summarize


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    from experiments import harness
    harness.reset()


def row(end, cleared, own=0, start=4):
    return dict(level=1, seed=300, end=end, elapsed=10.0, enemies_at_start=start, own_kills=own, cleared=cleared)


def test_cleared_exit_counts_only_exits_with_no_enemy_left():
    summary = summarize([row('exited', True, 4), row('exited', False, 2), row('scored_out', True, 4),
                         row('horizon', False, 0)])
    assert summary['exit'] == 0.5
    assert summary['cleared_exit'] == 0.25
    assert summary['scored_out'] == 0.25
    assert summary['horizon'] == 0.25
    assert summary['own_kills'] == 2.5
    assert summary['kills_fraction'] == pytest.approx(0.625)


def test_a_tiny_eval_flies_the_shipped_hunter_pilot_on_the_campaign_senses():
    rows, summary = evaluate(str(HUNTER_POLICY_PATH), levels=[1, 2], seeds=1, horizon=3.0, workers=1)
    assert [(r['level'], r['seed']) for r in rows] == [(1, 300), (2, 300)]
    assert {'exit', 'cleared_exit', 'scored_out', 'horizon', 'own_kills', 'kills_fraction', 'elapsed'} <= set(summary)
    assert all(r['end'] in ('exited', 'scored_out', 'horizon') for r in rows)
