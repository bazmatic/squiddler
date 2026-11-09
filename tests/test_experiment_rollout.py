import numpy as np
import pytest
from experiments import rollout
from experiments.score import score
from experiments.student import StudentSpec
from hunter.neural.policy import random_policy
from hunter.senses import SENSE_SIZE


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    from experiments import harness
    harness.reset()


@pytest.fixture(scope='module')
def spec(tmp_path_factory):
    path = tmp_path_factory.mktemp('rollout') / 'p.npz'
    random_policy(0).save(path)
    return StudentSpec(weights=str(path), decision_frames=6, turn_rate=1.0, greedy=False, sample_seed=1)


@pytest.fixture(scope='module')
def long_flight(spec):
    """One 50 s rollout shared by the tests that only read it (each used to fly its own, ~4 s apiece)."""
    return rollout.fly_rollout(14, 300, spec, horizon=50.0)


def test_a_rollout_has_aligned_senses_controls_and_rewards(spec):
    r = rollout.fly_rollout(14, 300, spec, horizon=10.0)
    assert r['senses'].shape[1] == SENSE_SIZE and r['senses'].dtype == np.float32
    assert len(r['senses']) == len(r['controls']) == len(r['rewards']) >= 90     # 10 s at 10 decisions a second
    assert r['controls'].shape[1] == 3 and set(np.unique(r['controls'][:, 0])) <= {0, 1, 2}


def test_the_rewards_add_up_to_the_lab_score_times_minutes(long_flight):
    r = long_flight
    result = r['result']
    assert result.elapsed >= 45.0          # the lab's floor on short episodes does not apply
    assert float(r['rewards'].sum()) == pytest.approx(score(result) * result.elapsed / 60.0, rel=1e-4, abs=1e-3)


def test_collisions_and_idling_cost_and_kills_pay(long_flight):
    r = long_flight
    result = r['result']
    if result.kills == 0:
        assert not (r['rewards'] > 0).any()              # nothing pays without a kill
    if result.wall_impulse == 0 and result.hits_taken == 0 and result.idle_seconds == 0 and result.kills == 0:
        assert not r['rewards'].any()                    # nothing happened, nothing earned
    assert (r['rewards'] < 0).any() or result.kills > 0 or not (
        result.wall_hits or result.hits_taken or result.idle_seconds)


def test_a_rollout_is_reproducible_and_the_sample_seed_changes_it(spec):
    a = rollout.fly_rollout(14, 301, spec, horizon=8.0)
    b = rollout.fly_rollout(14, 301, spec, horizon=8.0)
    assert np.array_equal(a['controls'], b['controls']) and np.array_equal(a['rewards'], b['rewards'])
    other = StudentSpec(weights=spec.weights, greedy=False, sample_seed=2)
    c = rollout.fly_rollout(14, 301, other, horizon=8.0)
    assert not np.array_equal(a['controls'], c['controls'])


def test_a_greedy_spec_is_unchanged_by_the_sample_seed(tmp_path):
    path = tmp_path / 'g.npz'
    random_policy(3).save(path)
    one = rollout.fly_rollout(14, 301, StudentSpec(weights=str(path), sample_seed=1), horizon=6.0)
    two = rollout.fly_rollout(14, 301, StudentSpec(weights=str(path), sample_seed=2), horizon=6.0)
    assert np.array_equal(one['controls'], two['controls'])


def test_the_weights_file_is_not_modified_by_flying(spec):
    before = open(spec.weights, 'rb').read()
    rollout.fly_rollout(14, 301, spec, horizon=4.0)
    assert open(spec.weights, 'rb').read() == before


def test_a_wall_scale_multiplies_only_the_wall_penalty(spec, long_flight):
    plain = long_flight
    heavy = rollout.fly_rollout(14, 300, spec, horizon=50.0, wall_scale=5.0)
    assert heavy['result'] == plain['result']                          # flying is unchanged
    from experiments.score import Weights
    extra = 4.0 * Weights().quotient * plain['result'].wall_impulse
    assert float(plain['rewards'].sum() - heavy['rewards'].sum()) == pytest.approx(extra, rel=1e-4, abs=1e-3)


def test_a_hit_scale_multiplies_only_the_enemy_hit_penalty(spec, long_flight):
    plain = long_flight
    heavy = rollout.fly_rollout(14, 300, spec, horizon=50.0, hit_scale=5.0)
    assert heavy['result'] == plain['result']
    from experiments.score import Weights
    extra = 4.0 * Weights().hit * plain['result'].hits_taken
    assert float(plain['rewards'].sum() - heavy['rewards'].sum()) == pytest.approx(extra, rel=1e-4, abs=1e-3)


class _Counter:
    kills = wall_impulse = hits_taken = spawned = powerups = 0
    died = False
    idle_seconds = 0.0
    rules_beaten = False


def _fly_frames(recorder, speeds, max_speed=10.0):
    """Feed the recorder one decision step of frames at the given speeds (no events happen)."""
    from types import SimpleNamespace
    recorder._start_enemies = 1
    for frame, speed in enumerate(speeds):
        controller = SimpleNamespace(frames=frame, last_controls=(1, 0, 0))
        game = SimpleNamespace(hunter_flight=SimpleNamespace(controller=controller))
        hunter = SimpleNamespace(vx=speed, vy=0.0, max_speed=max_speed)
        recorder.before_frame(game, hunter)
        recorder.counted(_Counter(), frame / 60.0)
        if frame == 0:
            recorder.senses.append(np.zeros(SENSE_SIZE, np.float32))
        recorder.after_frame()
    return recorder.arrays()[2]


def test_near_motionless_frames_cost_the_still_penalty_per_second():
    recorder = rollout.RolloutRecorder(6, still_penalty=3.0, still_speed=0.15)
    rewards = _fly_frames(recorder, [0.0, 0.5, 1.4, 1.6, 5.0, 9.0])       # 3 frames under 15% of 10
    assert rewards[0] == pytest.approx(-3.0 * 3 / 60.0)


def test_moving_hunters_and_the_default_pay_no_still_penalty():
    assert _fly_frames(rollout.RolloutRecorder(6, still_penalty=3.0), [5.0] * 6)[0] == 0.0
    assert _fly_frames(rollout.RolloutRecorder(6), [0.0] * 6)[0] == 0.0


def test_speed_below_top_speed_costs_the_graded_speed_penalty_per_second():
    recorder = rollout.RolloutRecorder(6, speed_penalty=3.0)
    rewards = _fly_frames(recorder, [10.0, 10.0, 5.0, 5.0, 0.0, 0.0])      # shortfalls 0, 0, .5, .5, 1, 1
    assert rewards[0] == pytest.approx(-3.0 * 3.0 / 60.0)


def test_no_speed_penalty_at_top_speed_or_by_default():
    assert _fly_frames(rollout.RolloutRecorder(6, speed_penalty=3.0), [10.0] * 6)[0] == 0.0
    assert _fly_frames(rollout.RolloutRecorder(6), [0.0] * 6)[0] == 0.0


def test_collected_powerups_pay_the_powerup_reward():
    recorder = rollout.RolloutRecorder(6, powerup_reward=5.0)
    recorder._start_enemies = 1
    recorder.senses.append(np.zeros(SENSE_SIZE, np.float32))
    recorder.controls.append((1, 0, 0))
    recorder._events.append([0, 0, 0, 0.0, 0.0, 0.0, 2, 0.0, 0, 0])
    assert recorder.arrays()[2][0] == pytest.approx(10.0)
    assert rollout.RolloutRecorder(6).arrays()[2].size == 0


def test_each_second_with_upgraded_guns_pays_the_upgrade_bonus():
    from types import SimpleNamespace
    recorder = rollout.RolloutRecorder(6, upgrade_bonus=6.0)
    recorder._start_enemies = 1
    for frame in range(6):
        hunter = SimpleNamespace(vx=5.0, vy=0.0, max_speed=10.0, gun_upgrade_level=1 if frame < 3 else 0)
        controller = SimpleNamespace(frames=frame, last_controls=(1, 0, 0))
        recorder.before_frame(SimpleNamespace(hunter_flight=SimpleNamespace(controller=controller)), hunter)
        recorder.counted(_Counter(), frame / 60.0)
        if frame == 0:
            recorder.senses.append(np.zeros(SENSE_SIZE, np.float32))
        recorder.after_frame()
    assert recorder.arrays()[2][0] == pytest.approx(6.0 * 3 / 60.0)
    assert recorder.upgraded_frames == 3


def _death_rewards(died_at, penalty, steps=4):
    from types import SimpleNamespace
    recorder = rollout.RolloutRecorder(6, death_penalty=penalty)
    recorder._start_enemies = 1
    for step in range(steps):
        recorder.senses.append(np.zeros(SENSE_SIZE, np.float32))
        recorder.controls.append((1, 0, 0))
        recorder._events.append([0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0])
        counter = _Counter()
        counter.died = died_at is not None and step >= died_at
        recorder.counted(counter, step / 10.0)
    return recorder.arrays()[2]


def test_dying_costs_the_death_penalty_once_in_the_step_it_happens():
    assert _death_rewards(2, 200.0).tolist() == pytest.approx([0.0, 0.0, -200.0, 0.0])


def test_no_death_no_penalty_and_none_by_default():
    assert not _death_rewards(None, 200.0).any()
    assert not _death_rewards(2, 0.0).any()


def test_a_rollout_reports_whether_the_hunter_died(spec):
    assert rollout.fly_rollout(14, 300, spec, horizon=4.0)['died'] is False


def test_a_random_start_rollout_begins_from_a_different_view_of_the_maze(spec):
    here = rollout.fly_rollout(14, 300, spec, horizon=3.0)
    there = rollout.fly_rollout(14, 300, spec, horizon=3.0, random_start=True)
    assert len(there['senses']) >= 25 and not np.array_equal(here['senses'][0], there['senses'][0])


def _rewards_for_turns(turns, penalty, reversal=0.0):
    recorder = rollout.RolloutRecorder(6, turn_change_penalty=penalty, reversal_penalty=reversal)
    recorder._start_enemies = 1
    for turn in turns:
        recorder.senses.append(np.zeros(SENSE_SIZE, np.float32))
        recorder.controls.append((turn, 0, 0))
        recorder._events.append([0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0])
    return recorder.arrays()[2]


def test_starting_a_turn_or_reversing_one_costs_the_penalty_but_holding_or_stopping_does_not():
    # none, start left, hold left, reverse to right, hold right, stop, start right again
    rewards = _rewards_for_turns([1, 0, 0, 2, 2, 1, 2], 0.5)
    assert rewards.tolist() == pytest.approx([0.0, -0.5, 0.0, -0.5, 0.0, 0.0, -0.5])


def test_a_turn_on_the_very_first_decision_counts_as_starting_to_turn():
    assert _rewards_for_turns([0, 0], 0.5).tolist() == pytest.approx([-0.5, 0.0])


def test_a_direct_reversal_pays_the_reversal_penalty_on_top_but_a_pause_between_does_not():
    # start left, hold, flip to right (reversal), hold, pause, start left again (no reversal), flip to right
    rewards = _rewards_for_turns([0, 0, 2, 2, 1, 0, 2], 0.5, reversal=2.0)
    assert rewards.tolist() == pytest.approx([-0.5, 0.0, -2.5, 0.0, 0.0, -0.5, -2.5])


def test_no_reversal_penalty_by_default():
    assert _rewards_for_turns([0, 2, 0, 2], 0.5).tolist() == pytest.approx([-0.5, -0.5, -0.5, -0.5])


def test_no_turn_penalty_by_default():
    assert not _rewards_for_turns([0, 2, 0, 2], 0.0).any()
