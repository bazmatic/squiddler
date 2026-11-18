"""The Campaign Hunter's evaluation: per-episode rows and a summary that counts a cleared exit only when the ship left
by the exit with no enemy left."""
import pytest

from hunter.neural.loader import DEFAULT_PATH as HUNTER_POLICY_PATH
from tools.campaign_eval import evaluate, parse_levels, summarize


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    from experiments import harness
    harness.reset()


def row(end, cleared, own=0, start=4, gap=5.0):
    return dict(level=1, seed=300, end=end, elapsed=10.0, enemies_at_start=start, own_kills=own, cleared=cleared,
                max_no_kill_gap=gap)


def test_cleared_exit_counts_only_exits_with_no_enemy_left():
    summary = summarize([row('exited', True, 4), row('exited', False, 2), row('scored_out', True, 4),
                         row('horizon', False, 0)])
    assert summary['exit'] == 0.5
    assert summary['cleared_exit'] == 0.25
    assert summary['scored_out'] == 0.25
    assert summary['horizon'] == 0.25
    assert summary['own_kills'] == 2.5
    assert summary['kills_fraction'] == pytest.approx(0.625)


def test_a_demo_exit_is_an_exit_demos_no_kill_skip_would_have_let_happen(monkeypatch):
    from config import SETTINGS
    monkeypatch.setattr(SETTINGS.autoplay, 'noKillSeconds', 20)
    summary = summarize([row('exited', True, gap=20.0), row('exited', False, gap=20.5), row('scored_out', True, gap=1.0),
                         row('horizon', False, gap=0.0)])
    assert summary['demo_exit'] == 0.25 and summary['exit'] == 0.5


def test_kills_per_start_counts_an_episode_at_most_once_however_many_spawned():
    summary = summarize([row('horizon', False, own=8, start=4), row('horizon', False, own=1, start=4)])
    assert summary['kills_fraction'] == pytest.approx((1.0 + 0.25) / 2)
    assert summary['own_kills'] == 4.5


def test_missing_an_enemy_cannot_improve_the_reported_search_time():
    seen = dict(row('exited', True), started_unseen=True, first_sighting_seconds=2.0)
    missed = dict(row('scored_out', False), started_unseen=True, first_sighting_seconds=None,
                  search_horizon_seconds=60)
    summary = summarize([seen, missed])
    assert summary['search_found'] == .5
    assert summary['search_seconds'] == 31.0


def test_pursuit_eval_counts_contact_and_rejects_visible_starts():
    seen = dict(row('contact', False), pursuit_lesson=True, started_unseen=True,
                first_sighting_seconds=2, search_horizon_seconds=30)
    missed = dict(row('horizon', False), pursuit_lesson=True, started_unseen=True,
                  first_sighting_seconds=None, search_horizon_seconds=30)
    visible = dict(row('visible_start', False), pursuit_lesson=True, started_unseen=False,
                   first_sighting_seconds=0)
    summary = summarize([seen, missed, visible])
    assert summary['lesson_contact'] == pytest.approx(1/3)
    assert summary['search_seconds'] == 16
    assert summary['visible_starts'] == 1


@pytest.mark.parametrize('text', ['5-3', '0', '', '1-0', '-2', '3,,4'])
def test_levels_must_name_a_real_rising_range(text):
    with pytest.raises(ValueError, match='level'):
        parse_levels(text)


def test_levels_read_ranges_and_lists():
    assert parse_levels('1-3,7,9-10') == [1, 2, 3, 7, 9, 10]


def test_a_single_worker_eval_runs_on_the_dummy_drivers(monkeypatch):
    import tools.campaign_eval as campaign_eval
    monkeypatch.delenv('SDL_VIDEODRIVER', raising=False)
    monkeypatch.delenv('SDL_AUDIODRIVER', raising=False)
    monkeypatch.setattr(campaign_eval, '_fly', lambda job: row('horizon', False))
    campaign_eval.evaluate('hunter', levels=[1], seeds=1, workers=1)
    import os
    assert os.environ['SDL_VIDEODRIVER'] == os.environ['SDL_AUDIODRIVER'] == 'dummy'


def test_a_tiny_eval_flies_the_shipped_hunter_pilot_on_the_campaign_senses():
    rows, summary = evaluate(str(HUNTER_POLICY_PATH), levels=[1, 2], seeds=1, horizon=3.0, workers=1)
    assert [(r['level'], r['seed']) for r in rows] == [(1, 300), (2, 300)]
    assert {'demo_exit', 'exit', 'cleared_exit', 'scored_out', 'horizon', 'own_kills', 'kills_fraction',
            'elapsed'} <= set(summary)
    assert all(r['max_no_kill_gap'] >= 0 for r in rows)
    assert all(r['end'] in ('exited', 'scored_out', 'horizon') for r in rows)
