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
    config = {'task': 'campaign_hunter',
              'episode': {'horizon_seconds': horizon, 'episodes_per_iteration': episodes, 'learning_rate': 1e-4,
                          'levels': levels},
              'start': str(HUNTER_POLICY_PATH), 'shaping': dict(horizon=horizon, **SHAPING),
              'decision_frames': 6, 'turn_rate': 1.0}
    (run / 'config.json').write_text(json.dumps(config))
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
    flown = fly_campaign(1, 301, spec, horizon=2.0, shaping=CampaignShaping(**SHAPING))
    assert row == {'episode': 0, 'level': 1, 'end': flown['result'].end, 'return': float(flown['rewards'].sum()),
                   'kills': flown['result'].own_kills, 'seconds': flown['result'].elapsed}


def test_the_first_iteration_flies_the_shipped_start_migrated_as_training_did(tmp_path):
    """Training deletes its migrated start (iter_start.npz) when a run ends: iteration 0 migrates the start again."""
    from tools.watch_episode import _fly, episode_jobs
    run = campaign_run(tmp_path, [1], episodes=1, horizon=1.0)
    [job], shaping = episode_jobs(run, 0)
    level, end, ret, kills, seconds = _fly((job, shaping))
    spec = StudentSpec(weights=str(run / 'iter0000.npz'), greedy=False, sample_seed=0)   # the start, migrated
    flown = fly_campaign(1, 300, spec, horizon=1.0, shaping=CampaignShaping(**SHAPING))
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
