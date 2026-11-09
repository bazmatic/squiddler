import pytest

import config
from config import SETTINGS
from entities.tentacle_chain import drag_chain


def chain():
    return [(float(i), 0.0) for i in range(5)]


def drag(points, snap=False):
    drag_chain(points, (10.0, 10.0), 1.0, lambda i, t: 0.0, 0.5, 0.5, snap)


def test_low_graphics_is_a_setting_that_defaults_to_off():
    assert SETTINGS.game.lowGraphics is False


def test_tentacles_are_dragged_normally_when_low_graphics_is_off(monkeypatch):
    monkeypatch.setattr(SETTINGS.game, 'lowGraphics', False)
    points = chain()
    drag(points)
    assert points[0] == (10.0, 10.0) and points != chain()


def test_low_graphics_leaves_the_tentacles_alone_between_jumps(monkeypatch):
    monkeypatch.setattr(SETTINGS.game, 'lowGraphics', True)
    points = chain()
    drag(points)
    assert points == chain()


def test_low_graphics_still_lays_the_rest_pose_on_first_use(monkeypatch):
    monkeypatch.setattr(SETTINGS.game, 'lowGraphics', True)
    points = chain()
    drag(points, snap=True)
    assert points[0] == (10.0, 10.0) and points[4] == pytest.approx((14.0, 10.0))


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    from experiments import harness
    harness.reset()


@pytest.mark.parametrize('level', [20, 26])
def test_low_graphics_does_not_change_what_happens_in_an_episode(level):
    from experiments.harness import run_episode
    full = run_episode(level, 7, horizon=10.0, low_graphics=False)
    low = run_episode(level, 7, horizon=10.0, low_graphics=True)
    assert low == full
    assert config.SETTINGS.game.lowGraphics is False        # the harness puts the setting back
