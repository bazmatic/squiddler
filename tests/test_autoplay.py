import pytest
from game_handlers.autoplay import AutoplayDirector


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def director(clock=None, first=4, last=6, limit=90):
    return AutoplayDirector(first, last, limit, clock=clock or Clock())


def test_starts_at_the_first_level():
    assert director().level == 4


def test_a_level_in_progress_has_no_end_reason():
    d = director()
    d.start_level(enemy_count=5)
    assert d.end_reason(enemies_left=5) is None


def test_time_limit_ends_the_level():
    clock = Clock()
    d = director(clock, limit=90)
    d.start_level(enemy_count=5)
    clock.now += 89.9
    assert d.end_reason(5) is None
    clock.now += 0.2
    assert d.end_reason(5) == 'time'


def test_start_level_restarts_the_clock():
    clock = Clock()
    d = director(clock)
    d.start_level(enemy_count=1)
    clock.now += 200
    d.start_level(enemy_count=1)
    assert d.end_reason(1) is None


def test_advance_steps_through_the_range_and_wraps():
    d = director(first=4, last=6)
    assert [d.advance() for _ in range(5)] == [5, 6, 4, 5, 6]
    assert d.level == 6


def test_a_single_level_range_repeats_that_level():
    d = director(first=7, last=7)
    assert d.advance() == 7


def test_inverted_range_is_rejected():
    with pytest.raises(ValueError):
        director(first=9, last=4)



# --- no-kill rule ------------------------------------------------------------

def kill_director(clock, seconds=20, limit=300):
    d = AutoplayDirector(4, 6, limit, clock=clock, no_kill_seconds=seconds)
    d.start_level(enemy_count=5)
    return d


def test_a_level_with_no_kill_for_too_long_ends():
    clock = Clock()
    d = kill_director(clock)
    clock.now += 19.9
    assert d.end_reason(5) is None
    clock.now += 0.2
    assert d.end_reason(5) == 'no_kill'


def test_each_kill_restarts_the_clock():
    clock = Clock()
    d = kill_director(clock)
    for left in (4, 3, 2, 1):
        clock.now += 19
        assert d.end_reason(left) is None
    clock.now += 21
    assert d.end_reason(1) == 'no_kill'


def test_enemies_spawning_are_not_kills_but_later_kills_still_count():
    clock = Clock()
    d = kill_director(clock)
    clock.now += 15
    assert d.end_reason(8) is None   # three more appeared
    clock.now += 15
    assert d.end_reason(7) is None   # one of them died: a kill at +30
    clock.now += 19
    assert d.end_reason(7) is None
    clock.now += 2
    assert d.end_reason(7) == 'no_kill'


def test_zero_no_kill_seconds_disables_the_rule():
    clock = Clock()
    d = kill_director(clock, seconds=0)
    clock.now += 200
    assert d.end_reason(5) is None


def test_start_level_restarts_the_no_kill_clock():
    clock = Clock()
    d = kill_director(clock)
    clock.now += 19
    d.start_level(enemy_count=5)
    clock.now += 19
    assert d.end_reason(5) is None
