"""Editing a paused run's settings as it resumes works for every task (tools.trainer.resume_edits): the launch command
and the run's recorded settings agree, a setting another task owns is refused, and a failed launch puts it all back."""
import json
import shutil
from argparse import Namespace

import pytest

from tools import ppo_neural as ppo, run_record
from tools.trainer.control import TrainerError, resume
from tools.trainer.processes import REPO

# For each task, a reward (or, for the Wallcrawler team, which has no reward settings, an episode setting) and a new
# value for it.
EDITS = {'hunter': ('compass_reward', 30.0), 'campaign_hunter': ('beacon', 50.0),
         'jelly_team': ('jelly_catch_reward', 12.0), 'flocker_team': ('flocker_hit_reward', 2.0),
         'inchworm': ('inchworm_bite_reward', 3.0), 'wallcrawler_team': ('horizon', 45.0)}


class FakeLaunch:
    def __init__(self, fail=False):
        self.commands, self.fail = [], fail

    def __call__(self, command, log_path):
        self.commands.append(command)
        if self.fail:
            raise OSError('could not launch')
        return 4242


def _scenario_args(namespace, path):
    from learning.tasks import task
    namespace = Namespace(**vars(namespace))
    if getattr(task(namespace.task).adapter(), 'USES_OPPONENT', True):
        namespace.opponent = str((path / 'opponent.npz').resolve())
    return namespace


def recorded_for(namespace, path):
    """What the trainer records for a run with these arguments, by the code that records it."""
    from learning.run import expected_manifest
    if run_record.is_ppo({'task': namespace.task}):
        return ppo.run_record(namespace)
    return dict(task=namespace.task, args=vars(namespace), manifest=expected_manifest(_scenario_args(namespace, path)))


def paused_run(root, task):
    """run1, paused at iteration 2, recorded as the trainer records a run of `task`."""
    path = root / 'run1'
    path.mkdir(parents=True)
    argv = ['--task', task, '--out', str(path), '--start', str(path / 'iter0000.npz'), '--iterations', '50']
    if task == 'campaign_hunter':
        argv += ['--levels', '1-3']
    namespace = ppo.build_parser().parse_args(argv)
    namespace.levels = list(namespace.levels) if namespace.levels else namespace.levels
    shutil.copyfile(REPO / 'assets' / 'neural' / 'hunter.npz', path / 'opponent.npz')
    for i in range(3):
        (path / f'iter{i:04d}.npz').write_bytes(b'')
    (path / 'config.json').write_text(json.dumps(recorded_for(namespace, path), indent=2))
    (path / 'resume.pt').write_text('unchanged optimizer')
    (path / 'state.json').write_text('{"state":"paused","iteration":2}')
    return path


def launched_args(command):
    """The arguments the launched trainer parses from its command line."""
    return ppo.build_parser().parse_args(command[3:])


@pytest.mark.parametrize('task', sorted(EDITS))
def test_a_paused_run_resumes_with_a_changed_setting_and_records_what_it_launches_with(tmp_path, task):
    path = paused_run(tmp_path, task)
    setting, value = EDITS[task]
    launch = FakeLaunch()
    resume('run1', root=tmp_path, processes=[], launch=launch, changes={setting: value})
    args = launched_args(launch.commands[0])
    assert getattr(args, setting) == value and args.resume
    updated = json.loads((path / 'config.json').read_text())
    assert updated['args'][setting] == value
    # The run's settings are its arguments, their only copy; what it trains with, derived from them, is what the
    # launched trainer works out from its command line.
    from tools import run_record
    args.levels = list(args.levels) if args.levels else args.levels
    expected = recorded_for(args, path)
    assert not {'shaping', 'rewards', 'episode', 'start'} & set(updated)
    if run_record.is_ppo({'task': task}):
        assert run_record.shaping(updated) == run_record.shaping(dict(expected, rules=updated.get('rules')))
        assert run_record.episode(updated) == run_record.episode(expected)
    else:
        assert updated['manifest'] == expected['manifest']
    assert updated['resume_settings_changes'][-1]['changes'][0]['setting'] == setting
    assert (path / 'resume.pt').read_text() == 'unchanged optimizer'


@pytest.mark.parametrize('task', sorted(EDITS))
def test_a_setting_another_task_owns_is_refused_and_the_run_left_alone(tmp_path, task):
    path = paused_run(tmp_path, task)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    foreign = 'beacon' if task != 'campaign_hunter' else 'flocker_hit_reward'
    with pytest.raises(TrainerError, match='require a new run'):
        resume('run1', root=tmp_path, processes=[], launch=FakeLaunch(), changes={foreign: 3.0})
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


@pytest.mark.parametrize('task', sorted(EDITS))
def test_a_failed_launch_puts_the_run_back_as_it_was(tmp_path, task):
    path = paused_run(tmp_path, task)
    (path / ppo.PAUSE_FILE).touch()
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    setting, value = EDITS[task]
    with pytest.raises(OSError):
        resume('run1', root=tmp_path, processes=[], launch=FakeLaunch(fail=True), changes={setting: value})
    after = {p.name: p.read_bytes() for p in path.iterdir() if not p.name.startswith('config-before-resume-')}
    assert after == before


@pytest.mark.parametrize('task,setting,value,why', [
    ('campaign_hunter', 'beacon', 'nan', 'finite'),
    ('flocker_team', 'flocker_hit_reward', -1.0, 'nonnegative'),
    ('flocker_team', 'flocker_volley_reward', 5.0, 'below one accepted hit'),
    ('inchworm', 'inchworm_bite_reward', -1.0, 'nonnegative'),
])
def test_a_reward_the_task_refuses_is_refused_and_the_run_left_alone(tmp_path, task, setting, value, why):
    path = paused_run(tmp_path, task)
    before = (path / 'config.json').read_bytes()
    with pytest.raises(TrainerError, match=why):
        resume('run1', root=tmp_path, processes=[], launch=FakeLaunch(), changes={setting: value})
    assert (path / 'config.json').read_bytes() == before


def test_a_run_whose_rules_changed_since_it_started_cannot_take_edited_rewards(tmp_path):
    path = paused_run(tmp_path, 'jelly_team')
    config = json.loads((path / 'config.json').read_text())
    config['manifest']['scenario_version'] = -1          # as if the Jelly's rules moved on since the run started
    (path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(TrainerError, match='rules this run trains with have changed'):
        resume('run1', root=tmp_path, processes=[], launch=FakeLaunch(), changes={'jelly_catch_reward': 12.0})


def test_a_hunter_run_keeps_the_rules_it_has_for_life_when_its_rewards_change(tmp_path):
    path = paused_run(tmp_path, 'hunter')
    from tools import run_record
    config = json.loads((path / 'config.json').read_text())
    config['rules'].update(combat_ruleset='legacy', weapon_tuning=None, curved_walls=False)
    (path / 'config.json').write_text(json.dumps(config))
    resume('run1', root=tmp_path, processes=[], launch=FakeLaunch(), changes={'compass_reward': 30.0})
    shaping = run_record.shaping(json.loads((path / 'config.json').read_text()))
    assert shaping['compass_reward'] == 30.0
    assert (shaping['combat_ruleset'], shaping['weapon_tuning'], shaping['curved_walls']) == ('legacy', None, False)


def test_a_run_recorded_before_the_single_copy_records_only_its_settings_once_edited(tmp_path):
    """Its shaping, rewards and episode blocks go; the rules and history they held are kept apart from them."""
    from tools import run_record
    path = paused_run(tmp_path, 'hunter')
    config = json.loads((path / 'config.json').read_text())
    rules = config.pop('rules')
    config.update(shaping=dict(ppo.shaping_from(run_record.arguments(config)), **rules), rewards=[{'name': 'Kills'}],
                  episode={'episodes_per_iteration': 96, 'random_maze_from_iteration': 1}, start=config['args']['start'])
    (path / 'config.json').write_text(json.dumps(config))
    resume('run1', root=tmp_path, processes=[], launch=FakeLaunch(), changes={'compass_reward': 30.0})
    updated = json.loads((path / 'config.json').read_text())
    assert not {'shaping', 'rewards', 'episode', 'start'} & set(updated)
    assert updated['rules'] == rules and updated['history'] == {'random_maze_from_iteration': 1}


def test_the_dashboard_resume_form_can_edit_a_creature_runs_reward(tmp_path):
    """The page posts the paused run and its edits to /api/resume."""
    import threading
    from tests.test_trainer_server import post
    from tools.trainer import server as server_module
    path = paused_run(tmp_path, 'jelly_team')
    launch = FakeLaunch()
    httpd = server_module.make_server(0, root=tmp_path, processes=lambda: [], launch=launch)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, answer = post({'url': f'http://127.0.0.1:{httpd.server_address[1]}'}, '/api/resume',
                              {'run': 'run1', 'changes': {'jelly_catch_reward': 12}})
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert status == 200 and answer['run'] == 'run1'
    assert json.loads((path / 'config.json').read_text())['manifest']['rewards']['catch'] == 12.0
    assert launched_args(launch.commands[0]).jelly_catch_reward == 12.0
