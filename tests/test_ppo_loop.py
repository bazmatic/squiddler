import numpy as np
import pytest

pytest.importorskip('torch')

from experiments.harness import EpisodeResult
from experiments.run import TUNE_LEVELS
from hunter.neural.policy import InvalidPolicy, NeuralPolicy, random_policy
from hunter.senses import SENSE_SIZE
from tools import ppo_neural as ppo


def fake_rollout_factory():
    def rollout(jobs, workers):
        episodes = []
        for level, seed, spec, _ in jobs:
            rng = np.random.default_rng(spec.sample_seed)
            n = 30
            episodes.append({
                'senses': rng.normal(0, 1, (n, SENSE_SIZE)).astype(np.float32),
                'controls': np.stack([rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 2, n)],
                                     axis=1).astype(np.int8),
                'rewards': rng.normal(0, 1, n).astype(np.float32),
                'result': EpisodeResult(level, seed, 5, 30.0, 1, 0, 0, 0.0, 'horizon', 0)})
        return episodes
    return rollout


def test_jobs_depend_only_on_the_iteration_and_count_not_on_the_workers():
    a = ppo.make_jobs(3, 24, 'w.npz', 6, 1.0)
    b = ppo.make_jobs(3, 24, 'w.npz', 6, 1.0)
    assert a == b and len(a) == 24
    assert all(not spec.greedy and 300 <= seed < 400 and level in TUNE_LEVELS for level, seed, spec, _ in a)
    assert len({spec.sample_seed for _, _, spec in a}) == 24
    assert {spec.sample_seed for _, _, spec in a}.isdisjoint(
        {spec.sample_seed for _, _, spec in ppo.make_jobs(4, 24, 'w.npz', 6, 1.0)})


def test_the_kl_coefficient_fades_then_holds():
    assert ppo.kl_coefficient(0, 100) == pytest.approx(0.1)
    assert ppo.kl_coefficient(25, 100) == pytest.approx(0.06)
    assert ppo.kl_coefficient(50, 100) == pytest.approx(0.02)
    assert ppo.kl_coefficient(99, 100) == pytest.approx(0.02)


def test_a_missing_or_invalid_start_policy_stops_training(tmp_path):
    with pytest.raises(InvalidPolicy):
        ppo.train_rl(tmp_path / 'absent.npz', tmp_path / 'out', 1, rollout=fake_rollout_factory(), log=lambda *_: None)
    bad = tmp_path / 'bad.npz'
    bad.write_bytes(b'junk')
    with pytest.raises(InvalidPolicy):
        ppo.train_rl(bad, tmp_path / 'out', 1, rollout=fake_rollout_factory(), log=lambda *_: None)
    assert not (tmp_path / 'out').exists() or not list((tmp_path / 'out').glob('*.npz'))


def test_the_loop_writes_a_loadable_checkpoint_per_iteration_and_returns_the_last(tmp_path):
    start = tmp_path / 'start.npz'
    random_policy(0, hidden=16, embed=8).save(start)
    result = ppo.train_rl(start, tmp_path / 'out', iterations=3, episodes=6, workers=1,
                          rollout=fake_rollout_factory(), log=lambda *_: None)
    names = [p.name for p in sorted((tmp_path / 'out').glob('iter*.npz'))]
    assert names == ['iter0000.npz', 'iter0001.npz', 'iter0002.npz']
    assert len(result['history']) == 3 and result['last'] == tmp_path / 'out' / 'iter0002.npz'
    NeuralPolicy.load(result['last'])


def test_each_iteration_reports_the_wall_quotient_not_the_wall_hit_count(tmp_path):
    start = tmp_path / 'start.npz'
    random_policy(0, hidden=16, embed=8).save(start)
    result = ppo.train_rl(start, tmp_path / 'out', iterations=1, episodes=4, workers=1,
                          rollout=fake_rollout_factory(), log=lambda *_: None)
    stats = result['history'][0]
    assert 'wall_quotient_per_min' in stats and 'wall_hits_per_min' not in stats
    assert 'gate_score' not in stats


def test_the_default_rollout_carries_the_penalty_scales(monkeypatch, tmp_path):
    seen = []

    class FakePool:
        def __init__(self, workers, shaping=None):
            seen.append(shaping)

        def __call__(self, jobs):
            return fake_rollout_factory()(jobs, 1)

        def close(self):
            seen.append('closed')
    monkeypatch.setattr(ppo, 'PoolRollout', FakePool)
    start = tmp_path / 'start.npz'
    random_policy(0, hidden=16, embed=8).save(start)
    ppo.train_rl(start, tmp_path / 'out', iterations=1, episodes=4, workers=1,
                 shaping={'wall_scale': 5.0, 'still_penalty': 2.0}, log=lambda *_: None)
    assert seen == [{'wall_scale': 5.0, 'still_penalty': 2.0}, 'closed']


def test_each_iteration_logs_where_its_time_went(tmp_path):
    start = tmp_path / 'start.npz'
    random_policy(0, hidden=16, embed=8).save(start)
    result = ppo.train_rl(start, tmp_path / 'out', iterations=1, episodes=4, workers=1,
                          rollout=fake_rollout_factory(), log=lambda *_: None)
    assert result['history'][0]['rollout_s'] >= 0 and result['history'][0]['update_s'] > 0


def _timed_job(job):
    import os
    import time
    level, seed, spec, _ = job
    time.sleep(0.02 * (level % 3))
    return {'pid': os.getpid(), 'level': level, 'seed': seed, 'wall': 0.02 * (level % 3)}


def test_longest_first_orders_by_the_level_estimates_and_keeps_unknown_levels_first():
    jobs = [(13, 0, None), (14, 1, None), (15, 2, None), (16, 3, None)]
    order = ppo.longest_first(jobs, {13: 5.0, 14: 9.0, 15: 1.0})
    assert order == [3, 1, 0, 2]                       # level 16 is unknown: it goes first; then 9, 5, 1
    assert ppo.longest_first(jobs, {}) == [0, 1, 2, 3]  # nothing known: submission order


def test_the_pool_returns_results_in_job_order_and_reuses_its_workers():
    jobs = [(level, level * 10, None) for level in (14, 13, 15, 14, 13, 15)]
    pooled = ppo.PoolRollout(2, job=_timed_job)
    try:
        first = pooled(jobs)
        second = pooled(jobs)
    finally:
        pooled.close()
    assert [(e['level'], e['seed']) for e in first] == [(l, s) for l, s, _ in jobs]
    assert [(e['level'], e['seed']) for e in second] == [(l, s) for l, s, _ in jobs]
    assert {e['pid'] for e in first + second} <= {e['pid'] for e in first} | {e['pid'] for e in second}
    assert len({e['pid'] for e in first + second}) <= 2          # two workers, the same two both times


def test_progress_lines_are_flushed_so_a_redirected_log_shows_them(monkeypatch):
    import io
    import sys

    class Spy(io.StringIO):
        flushes = 0

        def flush(self):
            Spy.flushes += 1
    monkeypatch.setattr(sys, 'stdout', Spy())
    ppo.log_line('iteration=0')
    assert Spy.flushes >= 1


def test_the_training_horizon_comes_from_the_shaping_and_defaults_to_ninety_seconds(tmp_path):
    path = tmp_path / 'p.npz'
    random_policy(0).save(path)
    from experiments.student import StudentSpec
    job = (14, 300, StudentSpec(weights=str(path), greedy=False))
    short = ppo._rollout_job(job, shaping={'horizon': 4.0})
    assert short['result'].elapsed == pytest.approx(4.0, abs=0.05) and short['result'].end == 'horizon'
    assert ppo.DEFAULT_HORIZON == 90.0


def test_credit_settings_reach_the_actual_optimizer_update(monkeypatch, tmp_path):
    start = tmp_path / 'start.npz'
    random_policy(0, hidden=16, embed=8).save(start)
    seen = []
    original = ppo.improve

    def capture(*args, **kwargs):
        seen.append((kwargs['gamma'], kwargs['lam']))
        return original(*args, **kwargs)

    monkeypatch.setattr(ppo, 'improve', capture)
    ppo.train_rl(start, tmp_path / 'out', iterations=1, episodes=2, workers=1,
                 gamma=0.995, gae_lambda=0.98, rollout=fake_rollout_factory(), log=lambda *_: None)
    assert seen == [(0.995, 0.98)]
