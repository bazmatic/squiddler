from creatures.traits.energy import damage
import copy
import pytest
from experiments import harness
from experiments.harness import EpisodeResult, run_episode
from hunter.rule_params import RuleParams


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    harness.reset()


def test_an_episode_returns_a_result_for_the_request():
    result = run_episode(13, 0, horizon=3.0)
    assert isinstance(result, EpisodeResult)
    assert (result.level, result.seed) == (13, 0)
    assert result.end in ('cleared', 'horizon')
    assert 0 < result.elapsed <= 3.0 + 1 / 60
    assert result.enemies_at_start > 0
    assert min(result.kills, result.wall_hits, result.hits_taken, result.idle_seconds) >= 0


def test_the_same_episode_twice_is_identical():
    first = run_episode(13, 1, horizon=6.0)
    second = run_episode(13, 1, horizon=6.0)
    assert first == second


def test_an_episode_does_not_depend_on_what_ran_before():
    alone = run_episode(14, 0, horizon=6.0)
    run_episode(13, 1, horizon=6.0)
    run_episode(18, 0, horizon=6.0)
    assert run_episode(14, 0, horizon=6.0) == alone


def test_params_reach_the_pilot(monkeypatch):
    seen = {}
    original = harness.SyncWorker.__init__

    def spy(self, params, clock):
        seen['params'] = params
        original(self, params, clock)
    monkeypatch.setattr(harness.SyncWorker, '__init__', spy)
    run_episode(13, 0, params=RuleParams(pulse_frames=9.0), horizon=1.0)
    assert seen['params'].pulse_frames == 9.0


def test_a_short_horizon_ends_an_episode_at_the_horizon():
    result = run_episode(13, 0, horizon=2.0)
    assert result.end == 'horizon'
    assert result.elapsed == pytest.approx(2.0, abs=0.05)


def test_the_hunter_cannot_die_and_the_episode_runs_on_after_hits(monkeypatch):
    # In the real game the hunter is indestructible; the lab keeps it so and counts the hits.
    result = run_episode(26, 0, horizon=15.0)
    assert result.end in ('cleared', 'horizon')
    assert result.hits_taken > 0
    assert not hasattr(result, 'died')


def _clear_enemies_after(monkeypatch, frames):
    from game import Game
    original = Game.update
    state = {'frames': 0}

    def clear(self, dt):
        original(self, dt)
        state['frames'] += 1
        if state['frames'] == frames:
            for enemy in list(self.roster.alive()):
                enemy.active = False
    monkeypatch.setattr(Game, 'update', clear)


def test_the_extra_phase_ends_at_once_when_no_crystal_is_left(monkeypatch):
    _clear_enemies_after(monkeypatch, 10)
    frames = []
    result = run_episode(13, 0, horizon=5.0, after_clear=2.0, trace=lambda t, g, h: frames.append(t))
    assert result.end == 'cleared' and frames[-1] - result.elapsed < 0.1


def test_the_extra_phase_waits_for_a_crystal_up_to_after_clear(monkeypatch):
    from entities.powerup_crystal import PowerupCrystal
    from game import Game
    original = Game.update
    state = {'frames': 0}

    def clear_and_drop(self, dt):
        original(self, dt)
        state['frames'] += 1
        if state['frames'] == 10:
            for enemy in list(self.roster.alive()):
                enemy.active = False
            far = PowerupCrystal((-5000.0, -5000.0))     # never reached
            self.powerup_crystals.append(far)
    monkeypatch.setattr(Game, 'update', clear_and_drop)
    frames = []
    result = run_episode(13, 0, horizon=5.0, after_clear=2.0, trace=lambda t, g, h: frames.append(t))
    assert result.end == 'cleared' and result.elapsed < 1.0
    assert 2.0 <= frames[-1] - result.elapsed + 0.05 < 2.2


def test_with_pursue_hunter_every_enemy_targets_the_hunter_not_the_parked_player():
    from game_handlers.combat_targets import nearest_friendly
    seen = []

    def trace(t, game, hunter):
        for enemy in game.roster.alive():
            seen.append(nearest_friendly(enemy, game.ship, game.hunter) is game.hunter)
    run_episode(13, 0, horizon=3.0, pursue_hunter=True, trace=trace)
    assert seen and all(seen)


def test_a_mortal_hunter_ends_the_episode_when_it_is_destroyed():
    def kill_it(t, game, hunter):
        if t > 0.5:
            damage(hunter, hunter.encounter)
    result = run_episode(13, 0, horizon=10.0, health=1, trace=kill_it)
    assert result.end == 'died' and result.elapsed < 2.0


def test_an_immortal_hunter_is_not_hurt_by_hits_it_takes():
    def hit_it(t, game, hunter):
        hunter.immunity_remaining = 0.0
        damage(hunter, hunter.encounter)
    result = run_episode(13, 0, horizon=2.0, trace=hit_it)
    assert result.end != 'died' and result.hits_taken >= 1


def test_a_mortal_hunter_survives_fewer_hits_than_its_health():
    seen = []
    def hit_twice(t, game, hunter):
        if 0.5 < t < 0.52 or 1.5 < t < 1.52:
            hunter.immunity_remaining = 0.0
            damage(hunter, hunter.encounter)
        seen.append(hunter.health)
    # The opening encounter has no guns: later campaign encounters can add a third,
    # incidental hit and turn this health-budget check into a combat-survival test.
    result = run_episode(1, 0, horizon=3.0, health=3, trace=hit_twice)
    assert result.end != 'died' and 0 < min(seen) <= 1.0      # two hits taken; a wall may have taken a little more


def test_waiting_after_the_last_kill_is_not_idling():
    counter = harness.EventCounter()
    counter.observe(make_enemies(1))
    counter.tick(1.0)
    counter.cleared = True
    counter.tick(100.0)
    assert counter.idle_seconds == 0.0


def test_a_cleared_level_ends_the_episode_as_cleared(monkeypatch):
    from game import Game
    original = Game.update
    state = {'frames': 0}

    def clear_after_ten_frames(self, dt):
        original(self, dt)
        state['frames'] += 1
        if state['frames'] == 10:
            for enemy in list(self.roster.alive()):
                enemy.active = False
    monkeypatch.setattr(Game, 'update', clear_after_ten_frames)
    result = run_episode(13, 0, horizon=5.0)
    assert result.end == 'cleared'
    assert result.kills == result.enemies_at_start
    assert result.elapsed < 1.0


def test_counting_helpers_count_events_not_frames():
    counter = harness.EventCounter()
    for hit in (False, True, True, True, False, True):
        counter.wall(hit)
    assert counter.wall_hits == 2


def test_the_counter_adds_up_the_impulse_of_every_collision_frame():
    from experiments.harness import EventCounter
    counter = EventCounter()
    counter.wall(True, 0.5)
    counter.wall(True, 0.25)          # still in contact: one hit, more impulse
    counter.wall(False)
    counter.wall(True, 1.0)
    assert counter.wall_hits == 2 and counter.wall_impulse == pytest.approx(1.75)


def make_enemies(n):
    return [object() for _ in range(n)]


def test_kills_and_spawns_are_counted_from_the_enemies_themselves():
    pool = make_enemies(20)
    counter = harness.EventCounter()
    counter.observe(pool[:10])
    counter.observe(pool[:12])            # two spawned
    counter.observe(pool[:11])            # one died
    counter.observe(pool[:11])            # nothing
    counter.observe(pool[:8])             # three died at once
    assert counter.kills == 4
    assert counter.spawned == 2


def test_an_enemy_dying_as_others_spawn_in_the_same_frame_is_both_a_kill_and_spawns():
    boss, other, child_a, child_b = make_enemies(4)
    counter = harness.EventCounter()
    counter.observe([boss, other])
    counter.observe([other, child_a, child_b])   # the boss died and split into two: net +1
    assert counter.kills == 1
    assert counter.spawned == 2


def test_a_boss_kill_restarts_the_idle_clock():
    boss, other, child = make_enemies(3)
    counter = harness.EventCounter()
    counter.observe([boss, other])
    for second in range(30):
        counter.tick(float(second))
    idle_before = counter.idle_seconds
    assert idle_before > 0
    counter.observe([other, child, make_enemies(1)[0]])    # boss dies, splits in two
    counter.tick(30.0)
    counter.tick(31.0)
    assert counter.idle_seconds == idle_before


def test_idle_time_starts_after_twenty_seconds_without_a_kill():
    pool = make_enemies(6)
    counter = harness.EventCounter()
    counter.observe(pool[:5])
    for second in range(40):
        counter.tick(now=float(second))
    assert counter.idle_seconds == pytest.approx(19.0, abs=1.0)
    counter.observe(pool[:4])
    counter.tick(now=40.0)
    before = counter.idle_seconds
    counter.tick(now=45.0)
    assert counter.idle_seconds == before


def test_a_farming_level_reports_the_enemies_that_spawned():
    result = run_episode(27, 0, horizon=25.0)
    assert result.enemies_spawned >= 0
    assert result.kills <= result.enemies_at_start + result.enemies_spawned


# --- the hunter is never replaced, and the real clock never matters ---------------

def test_a_beacon_pickup_cannot_replace_the_hunter_mid_episode(monkeypatch):
    from game import Game
    original = Game.update
    state = {'frames': 0, 'replaced': None}

    def summon_again(self, dt):
        original(self, dt)
        state['frames'] += 1
        if state['frames'] == 30:
            before = self.hunter
            self._summon_hunter((self.ship.x, self.ship.y))   # what picking up a beacon does
            state['replaced'] = self.hunter is not before
    monkeypatch.setattr(Game, 'update', summon_again)
    result = run_episode(13, 0, horizon=2.0)
    assert state['replaced'] is False
    assert result.end in ('cleared', 'horizon')


def test_results_do_not_depend_on_the_real_clock(monkeypatch):
    import random as stdlib_random
    import time
    from game import Game

    def drifting_clock(drift_seed):
        rng = stdlib_random.Random(drift_seed)   # its own generator, not the game's
        value = [1.0e6]

        def tick():
            value[0] += rng.uniform(0.0, 0.01)
            return value[0]
        return tick
    runs = []
    for drift_seed in (1, 2):
        monkeypatch.setattr(time, 'monotonic', drifting_clock(drift_seed))
        monkeypatch.setattr(time, 'time', drifting_clock(drift_seed + 100))
        # The game reads wall time through Game.clock / clock_ms, bound when game.py was imported
        monkeypatch.setattr(Game, 'clock', staticmethod(drifting_clock(drift_seed + 200)))
        tick_ms = drifting_clock(drift_seed + 300)
        monkeypatch.setattr(Game, 'clock_ms', staticmethod(lambda: int(tick_ms() * 1000)))
        runs.append(run_episode(13, 5, horizon=60.0))
    assert runs[0] == runs[1]


def test_an_episode_notices_if_the_hunter_is_replaced(monkeypatch):
    from game import Game
    original = Game.update
    state = {'frames': 0}

    def swap(self, dt):
        original(self, dt)
        state['frames'] += 1
        if state['frames'] == 5:
            self.hunter_flight = copy.copy(self.hunter_flight)
    monkeypatch.setattr(Game, 'update', swap)
    with pytest.raises(RuntimeError, match='hunter was replaced'):
        run_episode(13, 0, horizon=2.0)


def test_the_hunter_flies_with_the_params_settings():
    from hunter.rule_params import RuleParams
    params = RuleParams(thrust_multiplier=0.25, sensor_range_fraction=0.2, turn_rate_multiplier=0.5)
    seen = []
    run_episode(13, 0, params=params, horizon=0.1, trace=lambda t, game, hunter: seen.append(game.hunter_flight.settings))
    settings = seen[0]
    assert (settings.thrust_multiplier, settings.sensor_range_fraction, settings.turn_rate_multiplier) == (0.25, 0.2, 0.5)


def test_a_stronger_engine_changes_the_episode():
    from hunter.rule_params import RuleParams
    base = run_episode(14, 0, params=RuleParams(thrust_multiplier=0.25), horizon=20.0)
    strong = run_episode(14, 0, params=RuleParams(thrust_multiplier=1.0), horizon=20.0)
    assert (base.kills, base.wall_hits, base.hits_taken, base.idle_seconds) != \
           (strong.kills, strong.wall_hits, strong.hits_taken, strong.idle_seconds)


def test_a_longer_sight_changes_what_the_hunter_does():
    from hunter.rule_params import RuleParams
    short = run_episode(14, 0, params=RuleParams(sensor_range_fraction=0.15), horizon=20.0)
    long = run_episode(14, 0, params=RuleParams(sensor_range_fraction=1.0), horizon=20.0)
    assert (short.kills, short.wall_hits, short.hits_taken, round(short.idle_seconds, 2)) != \
           (long.kills, long.wall_hits, long.hits_taken, round(long.idle_seconds, 2))


def test_a_faster_turn_changes_the_episode():
    from hunter.rule_params import RuleParams
    slow = run_episode(14, 0, params=RuleParams(turn_rate_multiplier=0.25), horizon=20.0)
    fast = run_episode(14, 0, params=RuleParams(turn_rate_multiplier=3.0), horizon=20.0)
    assert (slow.kills, slow.wall_hits, slow.hits_taken, round(slow.idle_seconds, 2)) != \
           (fast.kills, fast.wall_hits, fast.hits_taken, round(fast.idle_seconds, 2))


def test_the_player_is_hidden_from_the_hunter_by_default_and_can_be_shown(monkeypatch):
    import experiments.harness as h
    seen = []
    from hunter.perception import HunterPerception
    original = HunterPerception.observe

    def spy(self, hunter, maze, player, *args, **kwargs):
        seen.append(player)
        return original(self, hunter, maze, player, *args, **kwargs)
    monkeypatch.setattr(HunterPerception, 'observe', spy)
    run_episode(13, 0, horizon=1.0)
    assert seen and all(player is None for player in seen)
    seen.clear()
    run_episode(13, 0, horizon=1.0, hide_player=False)
    assert seen and all(player is not None for player in seen)


def _start_position(level, seed, **kwargs):
    from experiments.harness import run_episode
    seen = []
    run_episode(level, seed, horizon=0.1, trace=lambda t, game, hunter: seen.append((hunter.x, hunter.y)), **kwargs)
    return seen[0]


def test_a_random_start_puts_the_hunter_somewhere_new_for_each_seed_and_the_same_place_for_the_same_seed():
    default = _start_position(14, 1)
    starts = {_start_position(14, seed, random_start=True) for seed in range(6)}
    assert len(starts) >= 4                                   # six seeds, mostly different places
    assert default not in starts or len(starts) > 1
    assert _start_position(14, 3, random_start=True) == _start_position(14, 3, random_start=True)


def test_a_random_start_is_inside_the_maze_and_leaves_the_default_untouched():
    import config
    for seed in range(4):
        x, y = _start_position(20, seed, random_start=True)
        assert 0 <= x <= config.SCREEN_WIDTH and 0 <= y <= config.SCREEN_HEIGHT
    assert _start_position(20, 0) == _start_position(20, 0, random_start=False)


def _maze_grid(level, seed, **kwargs):
    grids = []
    run_episode(level, seed, horizon=0.1,
                trace=lambda t, game, hunter: grids.append(copy.deepcopy(game.maze.grid)), **kwargs)
    return grids[0]


@pytest.mark.parametrize('level', [14, 25])
def test_random_training_mazes_vary_by_episode_seed_and_can_be_replayed(level):
    first = _maze_grid(level, 300, random_maze=True)
    second = _maze_grid(level, 301, random_maze=True)
    assert first != second
    assert _maze_grid(level, 300, random_maze=True) == first


def test_random_maze_keeps_hand_built_levels_intact():
    assert _maze_grid(29, 300, random_maze=True) == _maze_grid(29, 301, random_maze=True)


def test_historical_episodes_keep_their_fixed_maze():
    assert _maze_grid(25, 300) == _maze_grid(25, 301)
