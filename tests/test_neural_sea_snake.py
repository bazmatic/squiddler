"""Sea snake inference wiring: scale-free senses, motor conversion, policy loading and the in-game driver."""
import math

import numpy as np
import pytest

import creatures
from learning.adapters import sea_snake as adapter
from learning.policy import random_policy
from physics.sea_snake import SeaSnakeCommand, MUSCLES
from tests.creature_parts import encounter, open_maze, world
from tests.test_sea_snake_creature import setting, ship_at


def test_observation_is_finite_and_has_versioned_shape_with_no_target():
    c, enc, _ = setting()
    x, masks, extra = adapter.observe(c, enc)
    assert x.shape == (len(adapter.FEATURES),) and np.isfinite(x).all() and extra == ()
    assert x[adapter.FEATURES.index('target_present')] == 0
    assert set(masks) == {n for n, _ in adapter.HEADS}
    enc.close()


def test_observation_does_not_depend_on_the_snakes_size():
    obs = []
    for radius in (20., 30.):
        c = creatures.build('sea_snake', (400., 286.), angle=0.)
        c.radius = radius
        wall = type('W', (), dict(start=(0., 286.+radius*4), end=(2000., 286.+radius*4), active=True))()
        enc = encounter(c, world_=world(ship=ship_at((400.+radius*6, 286.)), maze=open_maze([wall])))
        enc.ready()
        obs.append(adapter.observe(c, enc)[0])
        enc.close()
    assert obs[0] == pytest.approx(obs[1], abs=1e-3)
    assert obs[0][adapter.FEATURES.index('target_present')] == 1


def test_intention_converts_through_tanh_and_rejects_bad_actions():
    assert adapter.intention((0.,)*MUSCLES) == SeaSnakeCommand()
    assert adapter.intention((100., -100.)+(0.,)*5).muscles[:2] == (1., -1.)
    for bad in ((0.,)*(MUSCLES-1), (float('nan'),)*MUSCLES):
        with pytest.raises(ValueError):
            adapter.intention(bad)


def test_policy_round_trip_rejects_wrong_task(tmp_path):
    policy = random_policy(adapter.SPEC, 2, hidden=8, embed=4)
    path = tmp_path/'weights.npz'
    policy.save(path)
    loaded = adapter.load_policy(path)
    x = np.zeros(len(adapter.FEATURES), np.float32)
    assert loaded.choose(x, loaded.initial_state())[0] == policy.choose(x, policy.initial_state())[0]
    from learning.adapters.flocker import SPEC as FLOCKER_SPEC
    random_policy(FLOCKER_SPEC).save(path)
    with pytest.raises(ValueError, match='schema'):
        adapter.load_policy(path)


def test_missing_selected_policy_logs_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv('SEA_SNAKE_POLICY', '/no/such/sea_snake.npz')
    c, enc, _ = setting()
    assert c.snake_driver is None
    enc.step(1.)
    assert 'using scripted controller' in caplog.text
    enc.close()


def test_unloadable_policy_leaves_the_snake_on_the_scripted_controller_and_swimming(tmp_path, monkeypatch):
    monkeypatch.setenv('SEA_SNAKE_POLICY', str(tmp_path/'missing.npz'))
    c, enc, ship = setting((700., 286.))
    start = math.dist(c.get_pos(), ship.get_pos())
    for _ in range(300):
        enc.step(1.)
    assert c.snake_driver is None
    assert math.dist(c.get_pos(), ship.get_pos()) < start-100.
    enc.close()


def test_driver_decides_every_six_frames_and_the_snake_swims_under_it(tmp_path, monkeypatch):
    path = tmp_path/'policy.npz'
    random_policy(adapter.SPEC, 1, hidden=8, embed=4).save(path)
    monkeypatch.setenv('SEA_SNAKE_POLICY', str(path))
    c, enc, _ = setting((700., 286.))
    driver = c.snake_driver
    assert driver is not None and adapter.DECISION_FRAMES == 6
    seen, start = [], c.get_pos()
    for _ in range(60):
        enc.step(1.)
        seen.append(driver.command)
    changes = sum(a is not b for a, b in zip(seen, seen[1:]))
    assert changes == 9 and all(isinstance(s, SeaSnakeCommand) for s in seen)
    assert math.dist(start, c.get_pos()) > 0 and np.isfinite(c.get_pos()).all()
    enc.close()


def test_external_command_wins_over_the_policy(tmp_path, monkeypatch):
    path = tmp_path/'policy.npz'
    random_policy(adapter.SPEC, 1, hidden=8, embed=4).save(path)
    monkeypatch.setenv('SEA_SNAKE_POLICY', str(path))
    a, enc, _ = setting()
    a.snake_command = SeaSnakeCommand()
    enc.step(1.)
    assert not a.snake_driver.state.any()
    enc.close()


# The training task: preset, CLI dispatch, rewards, curriculum and the scripted baseline it must beat.

def test_preset_and_task_defaults_are_consistent_and_do_not_launch():
    import json
    from pathlib import Path
    from learning.tasks import task
    from tools.ppo_neural import build_parser
    preset = json.loads((Path(__file__).resolve().parents[1]/'config/training/sea_snake.json').read_text())
    assert preset['task'] == 'sea_snake'
    args = build_parser().parse_args(['--task', 'sea_snake'])
    for key, value in task('sea_snake').fresh_defaults.items():
        assert getattr(args, key) == value == preset[key]
    for key, value in task('sea_snake').reward_table().defaults.items():
        assert preset['sea_snake_'+key] == value == getattr(args, 'sea_snake_'+key)
    overridden = build_parser().parse_args(['--task', 'sea_snake', '--episodes', '2', '--bootstrap-episodes=0'])
    assert overridden.episodes == 2 and overridden.bootstrap_episodes == 0
    assert task('sea_snake').adapter().USES_OPPONENT is False
    from tools.trainer.settings import applies_to
    assert applies_to('sea_snake_sting_reward') == ('sea_snake',)


def test_cli_dispatch_keeps_planned_horizon_without_running_training(monkeypatch):
    import learning.run
    from learning.tasks import task
    from tools.ppo_neural import main
    seen = []
    monkeypatch.setattr(learning.run, 'train', lambda args: seen.append(args) or dict(paused=False, last='not-started'))
    main(['--task', 'sea_snake'])
    assert seen[-1].horizon == task('sea_snake').fresh_defaults['horizon'] and seen[-1].start == ''
    main(['--task', 'sea_snake', '--horizon=2'])
    assert seen[-1].horizon == 2.


@pytest.mark.parametrize('stage', (1, 2, 3))
def test_a_scripted_episode_of_every_lesson_returns_finite_rewards(stage):
    from learning.adapters import sea_snake_scenario as scenario
    from learning.training import collect
    env = scenario.environment(horizon=2., scripted=True)
    try:
        trajectories, events, trace = collect(env, random_policy(adapter.SPEC, 2, hidden=8, embed=4), 7,
                                              dict(level=0, stage=stage))
        assert np.isfinite(trajectories[0]['rewards']).all() and np.isfinite(trajectories[0]['x']).all()
        assert events['stage'] == stage and math.isfinite(events['return_total']) and trace
    finally:
        env.close()


def test_the_scripted_snake_stings_a_still_target_in_the_open_and_is_paid_for_it():
    from learning.adapters import sea_snake_scenario as scenario
    from learning.sea_snake_rewards import TABLE
    from learning.training import collect
    env = scenario.environment(horizon=15., scripted=True)
    try:
        _, events, _ = collect(env, random_policy(adapter.SPEC, 2, hidden=8, embed=4), 3, dict(level=0, stage=1))
    finally:
        env.close()
    assert events['success'] == 1. and events['stings'] == 1 and events['resets'] == 0
    assert events['return_total'] > TABLE.defaults['sting_reward']*.5


def test_a_solver_reset_costs_the_reset_penalty(monkeypatch):
    from learning.adapters import sea_snake_scenario as scenario
    from learning.sea_snake_rewards import TABLE
    from physics.sea_snake import SeaSnakePhysics
    rewards = []
    for blow_up in (False, True):
        env = scenario.environment(horizon=1.)
        env.reset(5, level=0, stage=1)
        if blow_up:
            real, calls = SeaSnakePhysics.step, []
            def once(self, *a, **k):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError('sea snake physics exceeded its stable substep budget')
                return real(self, *a, **k)
            monkeypatch.setattr(SeaSnakePhysics, 'step', once)
        transition = env.step({0: (0.,)*MUSCLES})
        rewards.append(transition.reward)
        assert transition.events['resets'] == int(blow_up)
        env.close()
        monkeypatch.undo()
    assert rewards[0]-rewards[1] == pytest.approx(TABLE.defaults['reset_penalty'], abs=.5)


def test_curriculum_advances_open_then_moving_then_maze_and_keeps_practising_earlier_lessons():
    from learning.adapters import sea_snake_scenario as scenario
    assert scenario.CURRICULUM == (1, 2, 3)
    state = scenario.initial_curriculum(None)
    assert scenario.advance_curriculum(state, dict(success=1.)) == state
    passing = scenario.validation_evidence(dict(success=1., survived=1., wriggle=.9), state, None)
    # Reaching the target without swimming like a snake does not pass a lesson.
    assert not scenario.validation_evidence(dict(success=1., survived=1., wriggle=.2), state, None)[
        'sea_snake_validation_passed']
    for _ in range(scenario.PASSES_REQUIRED):
        assert state['stage'] == 1
        state = scenario.advance_curriculum(state, passing)
    assert state['stage'] == 2
    failing = scenario.validation_evidence(dict(success=0., survived=1., wriggle=.9), state, None)
    assert scenario.advance_curriculum(state, failing)['passing_blocks'] == 0
    cases = scenario.training_cases(0, 24, 3, curriculum=dict(stage=3))
    assert {c['stage'] for c in cases} == {1, 2, 3} and sum(c['stage'] == 3 for c in cases) >= 12
    assert {c['level'] for c in cases}.isdisjoint(c['level'] for c in scenario.validation_cases())
    assert {c['seed'] for c in scenario.evaluation_cases()}.isdisjoint(c['seed'] for c in scenario.validation_cases())


def test_maze_lessons_put_walls_between_the_snake_and_its_target():
    from learning.adapters import sea_snake_scenario as scenario
    from creatures.traits.rooted import wall_between
    env = scenario.environment(horizon=1.)
    blocked = []
    for seed in range(6):
        env.reset(seed, level=seed % 4, stage=3)
        blocked.append(wall_between(env.snake.get_pos(), env.target.get_pos(), env.world.maze))
    env.reset(0, level=0, stage=1)
    assert all(blocked) and not wall_between(env.snake.get_pos(), env.target.get_pos(), env.world.maze)
    env.close()


def _deploy_rows(success, wriggle):
    """Held-out rows, 12 per lesson; success and wriggle by lesson number (a float applies to every lesson)."""
    from learning.adapters import sea_snake_scenario as scenario
    rows = []
    for s in scenario.CURRICULUM:
        hits = round(12*(success[s] if isinstance(success, dict) else success))
        rows += [dict(level=4, seed=i, stage=s, success=float(i < hits), seconds=5.,
                      wriggle=wriggle[s] if isinstance(wriggle, dict) else wriggle) for i in range(12)]
    return {'episodes': rows}


def test_deployment_passes_a_clean_wave_that_reaches_and_fails_a_twitcher_that_reaches():
    from learning.adapters import sea_snake_scenario as scenario
    scripted = _deploy_rows(1., .97)
    def deploy(trained):
        return scenario.deployment_report({'trained': trained, 'scripted': scripted}, None)
    assert not deploy({'episodes': []})['passed']
    clean = deploy(_deploy_rows({1: 1., 2: .75, 3: .5}, .9))
    assert clean['passed'] and clean['per_lesson']['maze']['reaches'] and clean['per_lesson']['open']['wriggles']
    # It need not beat the scripted controller, which here reaches every time and wriggles better.
    assert clean['scripted_for_information']['maze']['success'] == 1.
    twitcher = deploy(_deploy_rows(1., .1))
    assert twitcher['reaches_target'] and not twitcher['wriggles_like_a_snake'] and not twitcher['passed']
    # Every lesson must pass both: one lesson that twitches, or one that rarely reaches, fails it.
    assert not deploy(_deploy_rows(1., {1: .9, 2: .9, 3: .5}))['passed']
    assert not deploy(_deploy_rows({1: 1., 2: 1., 3: .25}, .9))['passed']


def _wave(direction=1., frames=600, amplitude=.5, period=32., lag=.8):
    """Joint bends of a travelling wave like the scripted controller's (direction 1: head to tail), and their rates."""
    from physics.sea_snake import MAX_BEND
    t = np.arange(frames)[:, None]
    j = np.arange(MUSCLES)[None, :]
    w = 2*math.pi/period
    phase = -w*t-direction*j*lag
    return amplitude*MAX_BEND*np.sin(phase), -w*amplitude*MAX_BEND*np.cos(phase)


def _rates(angles):
    return np.vstack([np.zeros((1, angles.shape[1])), np.diff(angles, axis=0)])


def test_the_wriggle_score_is_high_for_a_travelling_wave_and_low_for_rigid_twitching_coiled_or_backward_swimming():
    from learning import sea_snake_wriggle as wriggle
    from physics.sea_snake import MAX_BEND
    frames, swim = 600, .05                      # 3 radii per second forward
    def grade(angles, extent=.75, forward=swim):
        return wriggle.score(list(angles), [extent]*len(angles), [forward]*len(angles))
    clean = grade(_wave()[0])
    assert clean['wriggle'] > .9 and min(clean[k] for k in wriggle.COMPONENTS) > .85
    rng = np.random.default_rng(0)
    low = dict(rigid=grade(np.zeros((frames, MUSCLES)), extent=1.),
               twitching=grade(rng.uniform(-.6, .6, (frames, MUSCLES))*MAX_BEND),
               coiled=grade(np.full((frames, MUSCLES), .8*MAX_BEND), extent=.3, forward=.01),
               backward=grade(_wave(direction=-1.)[0]))
    for name, graded in low.items():
        assert graded['wriggle'] < .3, (name, graded)
    assert low['backward']['wave'] == 0. and low['rigid']['curvature'] == 0. and low['coiled']['curvature'] == 0.
    assert low['twitching']['rhythm'] < .5


def _only(angles, joints):
    """These joints (0 is at the tail) keep their bends; the rest of the body is held rigid."""
    mask = np.zeros(MUSCLES)
    mask[list(joints)] = 1.
    return angles*mask[None, :]


def test_a_rigid_body_flapping_only_its_tail_does_not_pass_as_wriggling_but_a_tapered_whole_body_wave_does():
    from learning import sea_snake_wriggle as wriggle
    from learning.adapters.sea_snake_scenario import WRIGGLE_GATE
    def grade(angles):
        return wriggle.score(list(angles), [.75]*len(angles), [.05]*len(angles))
    tail_flap = grade(_only(_wave(amplitude=1.)[0], (0, 1)))
    rear_half = grade(_only(_wave(amplitude=.8)[0], range(MUSCLES//2+1)))
    assert tail_flap['wriggle'] < WRIGGLE_GATE and tail_flap['spread'] < .5
    assert rear_half['wriggle'] < WRIGGLE_GATE
    # Like the scripted snake's: biggest swing at the tail, about a third of it at the head end, still a full wave.
    taper = np.linspace(1., .3, MUSCLES)[None, :]
    tapered = grade(_wave(amplitude=.6)[0]*taper)
    assert tapered['wriggle'] > .9 > WRIGGLE_GATE


def test_shaping_pays_a_travelling_wave_above_twitching_at_equal_progress_and_nothing_for_wriggling_in_place():
    from learning import sea_snake_wriggle as wriggle
    from learning.sea_snake_rewards import TABLE
    from physics.sea_snake import MAX_BEND
    def paid(angles, rates, forward=.05, closing=True):
        row = TABLE.row()
        for i in range(1, len(angles)):
            for key, value in wriggle.frame_terms(angles[i], rates[i], rates[i-1], .75, forward, closing).items():
                row[key] += value
        return float(TABLE.pay([row], TABLE.defaults)[0])
    angles, rates = _wave()
    twitching = np.random.default_rng(1).uniform(-.6, .6, angles.shape)*MAX_BEND
    clean = paid(angles, rates)
    assert clean > 0 and clean > paid(twitching, _rates(twitching))+.3
    # Bounded: ten seconds of the cleanest wave pays well under one sting.
    assert clean < TABLE.defaults['sting_reward']/2
    # Wriggling in place, or swimming but not closing on the target, earns no wriggle reward.
    assert paid(angles, rates, forward=0.) <= 0 and paid(angles, rates, closing=False) <= 0
    straight = np.zeros_like(angles)
    assert paid(straight, straight) < 0                  # swimming rigid-straight costs
    # Flapping only the tail, however hard, earns a fraction of what the whole body's wave does.
    flap, flap_rates = _wave(amplitude=1.)
    assert paid(_only(flap, (0, 1)), _only(flap_rates, (0, 1))) < clean/3


def test_maze_progress_pays_for_heading_to_the_gap_not_for_pressing_the_wall_in_line_with_the_target(monkeypatch):
    from learning.adapters import sea_snake_scenario as scenario
    from physics.sea_snake import SeaSnakePhysics
    env = scenario.environment(horizon=1., rewards=dict(effort_cost=0., wall_cost=0., time_cost=0., wave_reward=0.,
                                                        curvature_reward=0., posture_cost=0., twitch_cost=0.))
    env.reset(1, level=0, stage=3)
    along, middle = env.gaps[0]
    start, goal = env.local(env.snake.get_pos()), env.local(env.target.get_pos())
    crossing = start[1]+(goal[1]-start[1])*(along-start[0])/(goal[0]-start[0])
    def screen(a, o):
        return env.point(a, o)
    at_wall, toward_gap = screen(along-1., crossing), screen(along-1., middle)
    # Against the baffle on the straight line is nearer as the crow flies, but further by way of the gap.
    assert math.dist(at_wall, env.target.get_pos()) < math.dist(toward_gap, env.target.get_pos())
    assert env.path_distance(toward_gap) < env.path_distance(at_wall)
    head = [at_wall]
    monkeypatch.setattr(SeaSnakePhysics, 'head', property(lambda self: head[0]))
    monkeypatch.setattr(env.enc, 'step', lambda dt: head.__setitem__(0, toward_gap))
    transition = env.step({0: (0.,)*MUSCLES})
    assert transition.reward == pytest.approx(
        .3*(env.path_distance(at_wall)-env.path_distance(toward_gap)))
    assert transition.reward > 0
    env.close()


def test_recorded_actions_replay_exactly_and_rule_changes_are_rejected(tmp_path):
    import json
    from types import SimpleNamespace
    from learning import replay
    from learning.adapters import sea_snake_scenario as scenario
    from learning.training import collect
    env = scenario.environment(horizon=.5)
    case = dict(seed=18, level=4, stage=2)
    try:
        _, events, trace = collect(env, random_policy(adapter.SPEC, 1, hidden=8, embed=4), case['seed'],
                                   dict(level=case['level'], stage=case['stage']))
    finally:
        env.close()
    manifest = scenario.metadata(SimpleNamespace(horizon=.5))
    manifest['training'] = {}
    config = dict(task='sea_snake', manifest=manifest)
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'episodes').mkdir()
    (tmp_path/'episodes/iter0000.json').write_text(json.dumps(dict(jobs=[case], episodes=[dict(events, **case)],
                                                                   actions=[trace])))
    result, = replay.fly(tmp_path, 0)
    assert result['return'] == pytest.approx(events['return_total'])
    config['manifest']['scenario_version'] += 1
    (tmp_path/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='rules'):
        replay.open_environment(tmp_path)


def test_in_the_game_a_snake_senses_prey_beyond_its_sight_but_in_training_it_does_not():
    import creatures
    from learning.adapters.sea_snake import observe, FEATURES, SIGHT_RADII
    from tests.creature_parts import encounter, open_maze, world
    from tests.test_sea_snake_creature import ship_at
    c = creatures.build('sea_snake', (400., 286.), angle=0.)
    far = (400.+c.radius*SIGHT_RADII*3, 286.)
    enc = encounter(c, world_=world(ship=ship_at(far), maze=open_maze([])))
    enc.ready()
    present = FEATURES.index('target_present')
    assert observe(c, enc)[0][present] == 0.
    assert observe(c, enc, scent=True)[0][present] == 1.
