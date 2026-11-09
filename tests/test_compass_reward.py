import numpy as np
import pytest
from experiments import rollout
from hunter.senses import FEATURE_NAMES, SENSE_SIZE


class _Counter:
    kills = wall_impulse = hits_taken = spawned = powerups = 0
    idle_seconds = 0.0
    died = False
    rules_beaten = False


def sense(dist=None, seen=False, bearing=0.0):
    s = np.zeros(SENSE_SIZE, np.float32)
    s[FEATURE_NAMES.index('enemies_present')] = 1.0
    if dist is not None:
        s[FEATURE_NAMES.index('enemy_compass_present')] = 1.0
        s[FEATURE_NAMES.index('enemy_compass_dist')] = dist
        s[FEATURE_NAMES.index('enemy_compass_sin')] = np.sin(np.radians(bearing))
        s[FEATURE_NAMES.index('enemy_compass_cos')] = np.cos(np.radians(bearing))
    if seen:
        s[FEATURE_NAMES.index('enemy0_present')] = 1.0
    return s


def rewards(senses, reward=10.0):
    recorder = rollout.RolloutRecorder(6, compass_reward=reward)
    recorder._start_enemies = 1
    for step, s in enumerate(senses):
        recorder.senses.append(s)
        recorder.controls.append((1, 0, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder.counted(_Counter(), step / 10.0)
    return recorder.arrays()[2]


def test_closing_on_the_compass_pays_and_backing_off_costs():
    r = rewards([sense(0.50), sense(0.48), sense(0.49)])
    assert r.tolist() == pytest.approx([0.2, -0.1, 0.0], abs=1e-5)


def test_nothing_is_paid_while_an_enemy_is_in_sight_or_with_no_compass():
    assert not rewards([sense(0.50, seen=True), sense(0.48)]).any()
    assert not rewards([sense(None), sense(0.48), sense(None)]).any()


def test_a_jump_when_the_nearest_enemy_changes_is_not_paid():
    assert not rewards([sense(0.10), sense(0.60)]).any()          # the near one died: the next is far off


def test_no_compass_reward_by_default():
    assert not rewards([sense(0.50), sense(0.48)], reward=0.0).any()


def test_following_counts_blind_steps_with_the_compass_near_the_nose():
    senses = [sense(0.5, bearing=10), sense(0.5, bearing=-40), sense(0.5, bearing=60), sense(0.5, seen=True), sense(None)]
    assert rollout.compass_following(np.asarray(senses)) == (3, 2)


def bored_rewards(views, seen=(), penalty=6.0, window=10.0):
    """Rewards for an episode in which the hunter sees the things in `views` at each decision (0.1 s apart), with
    only boredom on."""
    recorder = rollout.RolloutRecorder(6, boredom_penalty=penalty, boredom_window=window)
    recorder._start_enemies = 1
    for step, view in enumerate(views):
        recorder.senses.append(sense(0.5, seen=step in seen))
        recorder.controls.append((1, 0, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder._views.append(frozenset(view))
        recorder.counted(_Counter(), step / 10.0)
    return recorder.arrays()[2]


def test_seeing_the_same_things_gets_boring_after_a_second():
    r = bored_rewards([{'a', 'b'}] * 15)                       # spinning in a box: the same four walls go by
    assert not r[:10].any()                                    # the first look is new, then a second's grace
    assert r[10:].tolist() == pytest.approx([-0.6] * 5)        # 6 points a second, 0.1 s a step


def test_something_new_in_view_every_so_often_is_never_boring():
    assert not bored_rewards([{i // 5} for i in range(40)]).any()   # a new wall block every half second


def test_a_thing_not_seen_for_the_window_is_new_again():
    views = [{'a'}] + [set()] * 20 + [{'a'}]
    assert bored_rewards(views, window=1.5)[21] == 0.0 and bored_rewards(views, window=1.5)[11] < 0


def test_no_boredom_while_an_enemy_is_in_sight_or_by_default():
    assert not bored_rewards([{'a'}] * 15, seen=set(range(15))).any()
    assert not bored_rewards([{'a'}] * 15, penalty=0.0).any()


def test_boredom_counts_blind_steps_seeing_nothing_new():
    recorder = rollout.RolloutRecorder(6, boredom_window=10.0)
    assert recorder.bored_steps(np.asarray([sense(0.5)] * 15), [frozenset({'a'})] * 15) == (15, 5)


def test_the_view_is_the_wall_blocks_the_sight_rays_hit():
    from types import SimpleNamespace
    from hunter.senses import ray_hits
    east = SimpleNamespace(start=(600.0, 400.0), end=(600.0, 600.0), cell=(6, 5))
    far_east = SimpleNamespace(start=(800.0, 400.0), end=(800.0, 600.0), cell=(8, 5))
    hits = ray_hits((500.0, 500.0), 0.0, [east, far_east], 1000.0)
    assert hits[0] is east and hits[8] is None                 # ray 0 along the nose meets the nearer wall first


def test_shots_fired_with_no_enemy_in_sight_cost_the_shot_price():
    recorder = rollout.RolloutRecorder(6, shot_price=2.0)
    recorder._start_enemies = 1
    for step, (seen, shots) in enumerate([(False, 1), (True, 2), (False, 0), (False, 3)]):
        recorder.senses.append(sense(0.5, seen=seen))
        recorder.controls.append((1, 0, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder._shots.append(shots)
        recorder.counted(_Counter(), step / 10.0)
    assert recorder.arrays()[2].tolist() == pytest.approx([-2.0, 0.0, 0.0, -6.0])


def test_a_shot_within_20_degrees_of_an_enemy_in_sight_is_free():
    def aimed_at(bearing):
        s = sense(0.5)
        s[FEATURE_NAMES.index('enemy0_present')] = 1.0
        s[FEATURE_NAMES.index('enemy0_sin')] = np.sin(np.radians(bearing))
        s[FEATURE_NAMES.index('enemy0_cos')] = np.cos(np.radians(bearing))
        return s
    recorder = rollout.RolloutRecorder(6, shot_price=2.0)
    recorder._start_enemies = 1
    for step, s in enumerate([aimed_at(15), aimed_at(-19), aimed_at(40), sense(0.5)]):
        recorder.senses.append(s)
        recorder.controls.append((1, 0, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder._shots.append(1)
        recorder.counted(_Counter(), step / 10.0)
    assert recorder.arrays()[2].tolist() == pytest.approx([0.0, 0.0, -2.0, -2.0])


def test_each_second_spent_turning_costs_the_turning_penalty():
    recorder = rollout.RolloutRecorder(6, turning_penalty=2.0)
    recorder._start_enemies = 1
    for step, turn in enumerate([0, 1, 2, 2]):                     # left, straight, right, right
        recorder.senses.append(sense(0.5))
        recorder.controls.append((turn, 1, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder.counted(_Counter(), step / 10.0)
    assert recorder.arrays()[2].tolist() == pytest.approx([-0.2, 0.0, -0.2, -0.2])   # 0.1 s a decision


def test_each_hit_on_an_enemy_pays_the_hit_reward():
    recorder = rollout.RolloutRecorder(6, hit_reward=3.0)
    recorder._start_enemies = 1
    for step, hits in enumerate([0, 2, 1]):
        recorder.senses.append(sense(0.5))
        recorder.controls.append((1, 0, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder._hits.append(hits)
        recorder.counted(_Counter(), step / 10.0)
    assert recorder.arrays()[2].tolist() == pytest.approx([0.0, 6.0, 3.0])


def test_hits_and_shots_from_earlier_episodes_on_the_same_game_are_not_counted():
    from collections import Counter
    from types import SimpleNamespace
    maze = SimpleNamespace(grid_width=10, grid_height=10, cell_size_x=100, cell_size_y=100, offset_x=0, offset_y=0,
                           spatial_grid=SimpleNamespace(get_nearby_walls=lambda origin, reach: []))
    game = SimpleNamespace(roster=SimpleNamespace(alive=lambda: []), maze=maze,
                           encounter=SimpleNamespace(hits=Counter(hunter=800)),
                           hunter_flight=SimpleNamespace(on_observed=lambda f: None,
                                                         controller=SimpleNamespace(frames=0),
                                                         surroundings=SimpleNamespace(pickups=lambda: [])))
    hunter = SimpleNamespace(x=50.0, y=50.0, vx=0.0, vy=0.0, max_speed=8.0, shots_fired=500, angle=0.0,
                             settings=SimpleNamespace(sensor_range=lambda maze: 800.0))
    recorder = rollout.RolloutRecorder(6)
    recorder.begin(game, hunter, None)
    game.encounter.hits['hunter'] += 2
    hunter.shots_fired += 3
    recorder.before_frame(game, hunter)
    assert recorder._hits == [2] and recorder._shots == [3]


def panicking_recorder(turn_cost, steps, threshold=-10.0):
    """A recorder whose hunter has turned for `steps` decisions, each costing `turn_cost` points."""
    from types import SimpleNamespace
    recorder = rollout.RolloutRecorder(6, turning_penalty=turn_cost * 10, panic_score=threshold, panic_temperature=3.0)
    recorder._start_enemies = 1
    recorder._controller = SimpleNamespace(temperature=1.0)
    for step in range(steps):
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder._shots.append(0)
        recorder._hits.append(0)
        recorder._pending = True
        recorder._observed(sense(0.5))
        recorder.controls.append((0, 1, 0))                       # turning left
        recorder._pending = False
    return recorder


def test_the_hunter_panics_once_the_last_five_seconds_score_below_the_threshold():
    calm = panicking_recorder(0.1, 30)                            # 3 s at -1 a second: -3, no panic
    assert calm._controller.temperature == 1.0 and set(calm._temperatures) == {1.0}
    losing = panicking_recorder(0.5, 30)                          # -5 a second: below -10 after 2 s
    assert losing._controller.temperature == 3.0
    assert losing._temperatures[:21] == [1.0] * 21 and losing._temperatures[22:] == [3.0] * 8


def test_panic_is_off_by_default():
    from types import SimpleNamespace
    recorder = rollout.RolloutRecorder(6, turning_penalty=50.0)
    recorder._controller = SimpleNamespace(temperature=1.0)
    for _ in range(30):
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(0)
        recorder._shots.append(0)
        recorder._hits.append(0)
        recorder._pending = True
        recorder._observed(sense(0.5))
        recorder.controls.append((0, 1, 0))
        recorder._pending = False
    assert recorder._controller.temperature == 1.0


def test_rewards_worked_out_step_by_step_match_the_whole_episode():
    recorder = panicking_recorder(0.5, 30)
    recorder._views = [frozenset({'a'})] * 30
    whole = recorder.arrays()[2]
    S = np.asarray(recorder.senses, np.float32)
    parts = np.concatenate([recorder._step_rewards(lo, min(lo + 7, 30), S[lo:min(lo + 7, 30)],
                                                   np.zeros(min(lo + 7, 30) - lo, bool)) for lo in range(0, 30, 7)])
    assert parts == pytest.approx(whole, abs=1e-5)
