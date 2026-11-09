import json

from learning.adapters import hunter
from tools import ppo_neural as ppo
from tools.watch_episode import episode_jobs


def run_args(argv=('--episodes', '24', '--levels', '1-3', '--horizon', '60', '--panic-score', '-50')):
    args = vars(ppo.build_parser().parse_args(['--start', 'start.npz', *argv]))
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in args.items()}


def run_dir(tmp_path):
    """A hunter run as the trainer records one, whose history holds the historical 5% Rules-level share."""
    config = {'task': 'hunter', 'args': run_args(), 'rules': dict(hunter.RUN_RULES, random_maze=True),
              'history': {'rules_share': 0.05}, 'decision_frames': 6, 'turn_rate': 1.0}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    return tmp_path


def test_an_iterations_episodes_are_the_ones_training_flew(tmp_path):
    run = run_dir(tmp_path)
    jobs, shaping = episode_jobs(run, 7)
    expected = ppo.make_jobs(7, 24, run / 'iter0006.npz', 6, 1.0, (1, 2, 3))
    assert [(seed, spec) for _, seed, spec, _ in jobs] == [(seed, spec) for _, seed, spec, _ in expected]
    # Historical 5% Rules scheduling put the twentieth global episode on 25.
    start = 7 * 24
    assert [level for level, *_ in jobs] == [25 if (n + 1) // 20 > n // 20 else (1, 2, 3)[n % 3]
                                               for n in range(start, start + 24)]
    assert shaping['horizon'] == 60.0 and shaping['panic_score'] == -50.0 and shaping['combat_ruleset'] == 'legacy'


def test_the_first_iteration_flies_the_runs_starting_model(tmp_path):
    jobs, _ = episode_jobs(run_dir(tmp_path), 0)
    assert all(spec.weights == "start.npz" for _, _, spec, _ in jobs)


def test_new_runs_replay_level_25_as_an_ordinary_selected_level(tmp_path):
    run = run_dir(tmp_path)
    config = json.loads((run / 'config.json').read_text())
    del config['history']
    config['args']['levels'] = [25, 26]
    (run / 'config.json').write_text(json.dumps(config))
    jobs, _ = episode_jobs(run, 7)
    assert [level for level, *_ in jobs] == [25, 26] * 12


def test_resumed_historical_runs_replay_each_iterations_original_schedule(tmp_path):
    run = run_dir(tmp_path)
    config = json.loads((run / 'config.json').read_text())
    config['history']['level_cycle_from_iteration'] = 8
    (run / 'config.json').write_text(json.dumps(config))
    old, _ = episode_jobs(run, 7)
    new, _ = episode_jobs(run, 8)
    assert 25 in [level for level, *_ in old]
    assert [level for level, *_ in new] == [1, 2, 3] * 8


def test_old_and_new_maze_seeding_replay_correctly_across_resume(tmp_path):
    run = run_dir(tmp_path)
    config = json.loads((run / 'config.json').read_text())
    config['history']['random_maze_from_iteration'] = 8
    (run / 'config.json').write_text(json.dumps(config))
    assert episode_jobs(run, 7)[1]['random_maze'] is False
    assert episode_jobs(run, 8)[1]['random_maze'] is True


def test_a_run_recorded_before_the_single_copy_lists_and_flies_as_it_did(tmp_path):
    """A run that also kept its shaping and episode blocks reads its rules and history from them."""
    args = run_args()
    shaping = dict(ppo.shaping_from(type('A', (), args)), random_maze=True, combat_ruleset='legacy',
                   curved_walls=False, weapon_tuning=None)
    config = {'args': args, 'start': 'start.npz', 'rewards': [], 'shaping': shaping,
              'episode': {'episodes_per_iteration': 24, 'levels': [1, 2, 3], 'rules_share': 0.05,
                          'random_maze_from_iteration': 8},
              'decision_frames': 6, 'turn_rate': 1.0}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    jobs, flown = episode_jobs(tmp_path, 7)
    assert 25 in [level for level, *_ in jobs]
    assert flown == {**shaping, 'random_maze': False}
    assert episode_jobs(tmp_path, 8)[1] == shaping
