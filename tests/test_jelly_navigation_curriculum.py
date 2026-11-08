"""Jelly navigation lessons, idle costs, and reproducible run/replay behaviour."""
import json
import math
import shutil

import pytest

from learning.adapters import jelly_scenario as scenario
from learning import jelly_curriculum
from tools import ppo_neural


def test_navigation_unlocks_barriers_then_caves_only_after_successful_evidence_blocks():
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team'])
    state = scenario.initial_curriculum(args)
    assert {c['lesson'] for c in scenario.training_cases(0, 8, 3, curriculum=state)} == {'open'}
    assert all(c['stage'] == 1 and c['team_size'] == 1 for c in scenario.training_cases(0, 8, 3, curriculum=state))
    failure = dict(episodes=192, seconds=1920., catch_probability=.2, idle_seconds=0.)
    state = scenario.advance_curriculum(state, failure)
    assert scenario.curriculum_key(state) == 'open'
    success = dict(episodes=192, seconds=1920., catch_probability=.8, idle_seconds=0.)
    state = scenario.advance_curriculum(state, success)
    assert scenario.curriculum_key(state) == 'barrier'
    assert scenario.curriculum_promote(1, dict(navigation_lesson=0), state) == 1
    state = scenario.advance_curriculum(state, success)
    assert scenario.curriculum_key(state) == 'cave'
    assert {c['lesson'] for c in scenario.training_cases(0, 8, 1, curriculum=state)} == {'cave'}
    assert {c['seed'] for c in scenario.curriculum_validation_cases(scenario.initial_curriculum(args))}.isdisjoint(
        c['seed'] for c in scenario.curriculum_evaluation_cases(scenario.initial_curriculum(args)))


def idle_environment(**rewards):
    env = scenario.JellyEnvironment(horizon=3.,
        rewards=dict(progress_reward=0., boredom_penalty=.2, boredom_window=.5, **rewards))
    env.reset(5, lesson='open')
    target = env.world.additional_targets[0]
    target.vx = target.vy = 0.
    return env


def test_braking_gets_an_idle_cost_after_grace_and_real_movement_clears_it():
    env = idle_environment()
    try:
        for _ in range(4):
            assert env.step({0: (0., -3.)}).agent_rewards[0] == 0.
        result = env.step({0: (0., -3.)})
        assert result.agent_rewards[0] == pytest.approx(-.2)
        assert result.events['idle_seconds'] > 0
        jelly = env.team.members()[0]
        jelly.x, jelly.y = 600., 400.
        jelly.vx, jelly.vy = 2.5, 0.
        jelly.jelly_facing = 0.
        target = env.world.additional_targets[0]
        target.x, target.y = 350., 400.  # Moving away from quarry is still useful movement, not idle.
        for _ in range(10):
            assert env.step({0: (0., 0.)}).agent_rewards[0] == 0.
    finally:
        env.close()


def test_useful_contact_resets_the_idle_grace_period():
    env = idle_environment()
    try:
        for _ in range(6):
            env.step({0: (0., -3.)})
        jelly = env.team.members()[0]
        target = env.world.additional_targets[0]
        target.x, target.y = jelly.x + 3, jelly.y
        caught = env.step({0: (0., -3.)})
        assert caught.events['catches'] == 1
        assert caught.agent_rewards[0] == pytest.approx(10.)
        # No idle cost immediately after the useful contact, even though no thrust was used.
        for _ in range(3):
            assert env.step({0: (0., -3.)}).agent_rewards[0] == 0.
    finally:
        env.close()


def test_lesson_arenas_progress_from_open_to_one_barrier_and_shorten_early_episodes():
    env = scenario.JellyEnvironment()
    try:
        env.reset(6, lesson='open')
        assert env.world.maze.is_open((env.world.maze.offset_x + 5.5*env.world.maze.cell_size_x,
                                       env.world.maze.offset_y + 4.5*env.world.maze.cell_size_y), 8)
        while True:
            result = env.step({0: (0., -3.)})
            if result.truncated:
                break
        assert result.events['seconds'] == 10.
        env.reset(6, lesson='barrier')
        assert not env.world.maze.is_open((env.world.maze.offset_x + 5.5*env.world.maze.cell_size_x,
                                           env.world.maze.offset_y + 4.5*env.world.maze.cell_size_y), 8)
        from hunter.maze_route import MazeRoutes
        jelly = env.team.members()[0]
        assert MazeRoutes().route(env.world.maze, jelly.get_pos(), [env.enc.target_for(jelly)]) is not None
    finally:
        env.close()


def test_new_lesson_and_idle_penalty_replay_with_the_recorded_rules(tmp_path):
    from learning import replay
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--horizon', '1',
        '--jelly-boredom-penalty', '.2', '--jelly-boredom-window', '.3',
        '--jelly-progress-reward', '0'])
    shutil.copyfile(args.opponent, tmp_path/'opponent.npz')
    args.opponent = str(tmp_path/'opponent.npz')
    manifest = dict(scenario.metadata(args), training={})   # learning.run always records the training settings
    (tmp_path/'config.json').write_text(json.dumps(dict(task=args.task, args=vars(args), manifest=manifest)))
    env = scenario.environment(horizon=1, arenas=manifest['arenas'], **scenario.environment_options(manifest))
    job = dict(level=11, seed=5, stage=1, team_size=1, lesson='open')
    try:
        env.reset(**job)
        while True:
            result = env.step({0: (0., -3.)})
            if result.truncated:
                break
        expected = sum(result.events['agent_returns'].values())
        assert expected < 0
        (tmp_path/'episodes').mkdir()
        (tmp_path/'episodes/iter0000.json').write_text(json.dumps(dict(jobs=[job],
            episodes=[dict(result.events, **job)], actions=[env.action_trace])))
    finally:
        env.close()
    assert replay.fly(tmp_path, 0)[0]['return'] == pytest.approx(expected)


def test_navigation_run_records_its_lesson_evaluates_caves_separately_and_resumes_evidence(tmp_path):
    import torch
    from learning.run import train
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--start', '', '--out', str(tmp_path),
        '--jelly-boredom-penalty', '.02', '--horizon', '.2',
        '--bootstrap-episodes', '2', '--bootstrap-epochs', '1', '--episodes', '2', '--iterations', '1',
        '--workers', '1', '--budget-seconds', '0'])
    train(args)
    bootstrap = json.loads((tmp_path/'episodes/bootstrap.json').read_text())
    assert all(c['lesson'] == 'open' for c in bootstrap['jobs'])
    assert all(c['seconds'] == .2 for c in bootstrap['episodes'])
    report = json.loads((tmp_path/'evaluation.json').read_text())
    assert report['selection_context'] == 'open' and 'cave_benchmark' in report
    assert all(e['navigation_lesson'] == 0 for e in report['trained']['episodes'])
    assert all(e['navigation_lesson'] == 2 for e in report['cave_benchmark']['episodes'])
    args.resume, args.iterations = True, 2
    train(args)
    saved = torch.load(tmp_path/'resume.pt', map_location='cpu', weights_only=False)
    assert saved['curriculum']['episodes'] == 4


def test_a_caught_target_respawns_even_when_the_jelly_hugs_a_curved_cave_wall(tmp_path):
    """Run 154 crashed here: a Jelly in the thin strip of open floor along a curved wall sat in a grid cell whose
    centre was behind the wall, so no spawn looked reachable and the whole training run died."""
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--horizon', '10'])
    manifest = dict(scenario.metadata(args), training={})
    env = scenario.environment(horizon=10, arenas=manifest['arenas'], **scenario.environment_options(manifest))
    try:
        env.reset(level=14, seed=358, stage=1, team_size=1, lesson='open')
        env.team.members()[0].x, env.team.members()[0].y = 722.8, 101.3   # pressed against the top wall
        target = env.targets[0]
        env.respawn(target)
        assert any(math.dist((target.x, target.y), cell) < 1e-6 for cell in env.cells)
    finally:
        env.close()


def test_training_rollouts_end_early_after_a_long_gap_without_a_catch_but_evaluation_does_not():
    def run(training):
        env = scenario.JellyEnvironment(horizon=30., rewards=dict(no_catch_timeout=3.))
        if training:
            env.enable_training_limits()
        try:
            env.reset(5, lesson='cave')
            while True:
                result = env.step({0: (0., -3.)})   # braking never catches anything
                if result.truncated:
                    return result.events['seconds']
        finally:
            env.close()
    assert run(training=True) == pytest.approx(3., abs=.2)
    assert run(training=False) == 30.


def test_a_run_capped_at_stage_one_stays_on_solo_navigation_however_well_it_does(monkeypatch):
    # The cap is what is tested, not the scripted baseline (flying it took ~30 s): stand in a modest one.
    monkeypatch.setattr(scenario, '_baselines', {})
    monkeypatch.setattr(scenario, 'scripted_metrics',
                        lambda *a, **k: dict(catches_per_minute=1., stuck_fraction=.5))
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--jelly-start-lesson', '2', '--jelly-max-stage', '1'])
    state = scenario.initial_curriculum(args)
    great = dict(episodes=96, catches_per_minute=99., stuck_fraction=0., navigation_lesson=2, angular_spread=1.,
                 team_catches_per_minute=99.)
    assert scenario.curriculum_promote(1, great, state) == 1
    uncapped = scenario.initial_curriculum(ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--jelly-start-lesson', '2']))
    assert scenario.curriculum_promote(1, great, uncapped) == 2

