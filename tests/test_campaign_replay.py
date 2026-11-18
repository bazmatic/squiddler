"""Replaying a Campaign Hunter run's episodes from the trainer page: listed as training flew them, and flown again
exactly as training flew them."""
import json
import shutil

import numpy as np
import pytest

from experiments.campaign_rollout import CampaignShaping, fly_campaign
from experiments.student import StudentSpec
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
from hunter.neural.loader import DEFAULT_PATH as HUNTER_POLICY_PATH
from tools.migrate_policy import migrate
from tools.trainer import replay

SHAPING = dict(own_kill=10.0, pickup=3.0, beacon=5.0, exit=100.0, cleared_exit=100.0, score_out=-100.0,
               wall_crash=-1.0, compass_following=0.5)


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    from experiments import harness
    harness.reset()


def campaign_run(root, levels, episodes, horizon, iterations=1):
    """A campaign run folder as tools.ppo_neural writes it, started from the shipped hunter pilot, with `iterations`
    checkpoints (each the start migrated to the campaign senses)."""
    run = root / 'run7'
    run.mkdir()
    from tools import ppo_neural
    argv = ['--task', 'campaign_hunter', '--start', str(HUNTER_POLICY_PATH), '--out', str(run),
            '--episodes', str(episodes), '--horizon', str(horizon), '--levels', ','.join(map(str, levels))]
    for name, value in SHAPING.items():
        argv += ['--' + name.replace('_', '-'), str(value)]
    args = ppo_neural.build_parser().parse_args(argv)
    ppo_neural.write_run_config(args)
    migrate(str(HUNTER_POLICY_PATH), str(run / 'iter0000.npz'), CAMPAIGN_FEATURE_NAMES)
    for i in range(1, iterations):
        shutil.copy(run / 'iter0000.npz', run / f'iter{i:04d}.npz')
    return run


def test_a_campaign_run_lists_its_episodes_as_training_flew_them(tmp_path):
    run = campaign_run(tmp_path, [4, 5, 6], episodes=8, horizon=180.0, iterations=4)
    assert replay.replayable(run)
    rows = replay.episodes('run7', 3, root=tmp_path)
    assert rows == [{'episode': i, 'level': (4, 5, 6)[(24 + i) % 3], 'seed': 300 + 24 + i} for i in range(8)]


def test_a_flown_again_campaign_episode_is_the_one_training_flew(tmp_path):
    campaign_run(tmp_path, [1], episodes=1, horizon=2.0, iterations=2)
    [row] = replay.fly('run7', 1, root=tmp_path, workers=1)
    spec = StudentSpec(weights=str(tmp_path / 'run7' / 'iter0000.npz'), greedy=False, sample_seed=1)
    flown = fly_campaign(1, 301, spec, horizon=2.0, shaping=CampaignShaping(**SHAPING), graded_shooting=True,
                         sticky_targets=True)
    assert row == {'episode': 0, 'level': 1, 'end': flown['result'].end, 'return': float(flown['rewards'].sum()),
                   'kills': flown['result'].own_kills, 'seconds': flown['result'].elapsed}


def test_the_first_iteration_flies_the_shipped_start_migrated_as_training_did(tmp_path):
    """Training deletes its migrated start (iter_start.npz) when a run ends: iteration 0 migrates the start again."""
    from tools.watch_episode import _fly, episode_jobs
    run = campaign_run(tmp_path, [1], episodes=1, horizon=1.0)
    [job], shaping = episode_jobs(run, 0)
    level, end, ret, kills, seconds = _fly((job, shaping))
    spec = StudentSpec(weights=str(run / 'iter0000.npz'), greedy=False, sample_seed=0)   # the start, migrated
    flown = fly_campaign(1, 300, spec, horizon=1.0, shaping=CampaignShaping(**SHAPING), graded_shooting=True,
                         sticky_targets=True)
    assert (end, ret, kills, seconds) == (flown['result'].end, float(flown['rewards'].sum()),
                                          flown['result'].own_kills, flown['result'].elapsed)


def test_watching_flies_the_same_episode_frame_by_frame(tmp_path):
    """The window turns the tentacles' drag physics back on and is shown every frame; the episode must not change."""
    migrate(str(HUNTER_POLICY_PATH), str(tmp_path / 'p.npz'), CAMPAIGN_FEATURE_NAMES)
    spec = StudentSpec(weights=str(tmp_path / 'p.npz'), greedy=False, sample_seed=3)
    seen = []
    watched = fly_campaign(1, 303, spec, horizon=2.0, trace=lambda seconds, game: seen.append(seconds),
                           low_graphics=False)
    trained = fly_campaign(1, 303, spec, horizon=2.0)
    assert len(seen) == 120 and seen[-1] == pytest.approx(2.0)
    for key in ('senses', 'controls', 'rewards', 'temperatures'):
        assert np.array_equal(watched[key], trained[key])
    assert watched['result'] == trained['result']


def test_pursuit_lesson_settings_round_trip_into_replays(tmp_path):
    from tools.watch_episode import episode_jobs, _fly_campaign
    run = campaign_run(tmp_path, [10], episodes=1, horizon=.2)
    config = json.loads((run / 'config.json').read_text())
    config['args'].update(pursuit_lesson=True, search_contact=100)
    (run / 'config.json').write_text(json.dumps(config))
    [job], shaping = episode_jobs(run, 0)
    assert shaping['pursuit_lesson'] is True
    flown = _fly_campaign(job, shaping)
    assert flown['result'].pursuit_lesson
    assert flown['result'].started_unseen and flown['result'].end == 'horizon'


def test_mixed_practice_replays_preserve_combat_and_pursuit_episode_modes(tmp_path):
    from tools.watch_episode import episode_jobs, _fly_campaign
    run = campaign_run(tmp_path, [10, 40], episodes=2, horizon=.2)
    config = json.loads((run / 'config.json').read_text())
    config['args'].update(pursuit_lesson=True, combat_practice_levels=[40], search_contact=100)
    (run / 'config.json').write_text(json.dumps(config))
    jobs, shaping = episode_jobs(run, 0)
    search, combat = [_fly_campaign(job, shaping) for job in jobs]
    assert search['result'].pursuit_lesson
    assert not combat['result'].pursuit_lesson
    assert combat['result'].enemies_at_start > 1
    assert combat['reward_totals']['search_contact'] == 0


def test_old_campaign_replays_keep_the_zero_hit_inputs_but_new_runs_record_feedback(tmp_path):
    from tools import run_record
    from tools.watch_episode import episode_jobs
    run = campaign_run(tmp_path, [40], episodes=1, horizon=.2)
    config = json.loads((run / 'config.json').read_text())
    assert config['rules']['ship_hit_feedback'] is True
    assert config['rules']['graded_shooting'] is True
    assert config['rules']['accuracy_power'] == 2
    assert config['rules']['sticky_targets'] is True
    assert episode_jobs(run, 0)[1]['sticky_targets'] is True
    assert episode_jobs(run, 0)[1]['ship_hit_feedback'] is True
    config['rules'].pop('sticky_targets')  # runs made after hit repair but before persistent targeting
    (run / 'config.json').write_text(json.dumps(config))
    assert episode_jobs(run, 0)[1]['sticky_targets'] is False
    assert episode_jobs(run, 0)[1]['ship_hit_feedback'] is True
    config.pop('rules')  # an actual pre-repair run had no campaign lifetime rules
    (run / 'config.json').write_text(json.dumps(config))
    assert run_record.shaping(config)['ship_hit_feedback'] is False
    assert episode_jobs(run, 0)[1]['ship_hit_feedback'] is False
    assert episode_jobs(run, 0)[1]['graded_shooting'] is False
    assert episode_jobs(run, 0)[1]['accuracy_power'] == 1
    assert episode_jobs(run, 0)[1]['sticky_targets'] is False


def test_target_recovery_is_a_lifetime_rule_with_legacy_false(tmp_path):
    from tools import run_record
    from tools.watch_episode import episode_jobs, _fly_campaign
    run = campaign_run(tmp_path, [2], episodes=1, horizon=.2)
    config = json.loads((run / 'config.json').read_text())
    assert config['rules']['target_recovery'] is True
    config['rules'].pop('target_recovery')
    assert run_record.rules(config)['target_recovery'] is False
    assert run_record.shaping(config)['target_recovery'] is False
    config['rules']['target_recovery'] = True
    (run / 'config.json').write_text(json.dumps(config))
    jobs, shaping = episode_jobs(run, 0)
    assert shaping['target_recovery'] is True
    assert _fly_campaign(jobs[0], shaping)['result'].level == 2


def test_exit_practice_round_trips_through_training_replay(tmp_path):
    from learning.campaign_training_levels import EXIT_WITH_CRAWLER
    from tools.watch_episode import episode_jobs, _fly_campaign
    run = campaign_run(tmp_path, [EXIT_WITH_CRAWLER], episodes=1, horizon=.2)
    [job], shaping = episode_jobs(run, 0)
    assert job[0] == EXIT_WITH_CRAWLER
    episode = _fly_campaign(job, shaping)
    assert episode['result'].level == EXIT_WITH_CRAWLER
    assert episode['result'].enemies_at_start == 1


def test_replay_reconstructs_level_cycle_before_and_after_resume_edits(tmp_path):
    from tools.watch_episode import episode_jobs
    from tools.ppo_neural import make_jobs
    run = campaign_run(tmp_path, [2, 32, 1032], episodes=4, horizon=.2, iterations=4)
    config = json.loads((run / 'config.json').read_text())
    config['resume_settings_changes'] = [{'after_iteration': 1, 'changes': [
        {'setting': 'levels', 'previous': [2, 16, 32, 1032], 'value': [2, 32, 1032]}]}]
    (run / 'config.json').write_text(json.dumps(config))
    before, _ = episode_jobs(run, 1)
    after, _ = episode_jobs(run, 2)
    assert [j[0] for j in before] == [j[0] for j in make_jobs(1, 4, 'w', 6, 1, [2, 16, 32, 1032])]
    assert [j[0] for j in after] == [j[0] for j in make_jobs(2, 4, 'w', 6, 1, [2, 32, 1032])]
