# tests/test_rules_hunter_training.py
import numpy as np
import pytest

import level_config
from experiments import harness, rollout
from experiments.harness import run_episode
from hunter.senses import SENSE_SIZE


@pytest.fixture(scope='module', autouse=True)
def fresh_lab_game():
    """The lab keeps one Game per process; an earlier test may have shut pygame down under it."""
    harness.reset()
    yield
    harness.reset()


@pytest.fixture(autouse=True)
def one_rules_hunter(monkeypatch):
    monkeypatch.setattr(level_config, 'get_level_rules_hunter_count', lambda level: 1 if level == 13 else 0)


def test_an_episode_on_another_level_has_no_rules_hunter():
    assert run_episode(14, 0, horizon=2.0, pursue_hunter=True).rules_hunters == 0


def test_the_lab_overrides_do_not_outlive_the_episode():
    run_episode(13, 0, horizon=1.0, pursue_hunter=True)
    game = harness._game
    assert '_rules_worker_factory' not in game.__dict__ and 'protagonist' not in game.__dict__


class _Counter:
    kills = wall_impulse = hits_taken = spawned = powerups = 0
    idle_seconds = 0.0
    died = False
    rules_beaten = False


def _bounty_rewards(beaten_at, bounty, steps=4):
    recorder = rollout.RolloutRecorder(6, rules_bounty=bounty)
    recorder._start_enemies = 1
    for step in range(steps):
        recorder.senses.append(np.zeros(SENSE_SIZE, np.float32))
        recorder.controls.append((1, 0, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        counter = _Counter()
        counter.rules_beaten = beaten_at is not None and step >= beaten_at
        recorder.counted(counter, step / 10.0)
    return recorder.arrays()[2]


def test_destroying_it_pays_the_bounty_once_in_the_step_it_happens():
    assert _bounty_rewards(2, 50.0).tolist() == pytest.approx([0.0, 0.0, 50.0, 0.0])


def test_no_bounty_without_a_win_and_none_by_default():
    assert not _bounty_rewards(None, 50.0).any()
    assert not _bounty_rewards(2, 0.0).any()
