import json

import pytest

from tools import ppo_neural as ppo
from tools.trainer.settings import SETTINGS, NoSettings, Value, apply_changes, changed, read_settings, to_argv


def args_for(argv):
    args = vars(ppo.build_parser().parse_args(argv))
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in args.items()}


NEW_ARGV = ['--iterations', '1000', '--levels', '1-30', '--speed-penalty', '3', '--compass-reward', '100',
            '--health', '10', '--panic-score', '-50', '--boredom-window', '12']


def make_run(root, name, config):
    path = root / name
    path.mkdir(parents=True)
    if config is not None:
        (path / 'config.json').write_text(json.dumps(config))
    return path


def new_run(root, name='run52', argv=NEW_ARGV):
    return make_run(root, name, {'rewards': [], 'args': args_for(argv)})


def test_a_run_with_args_is_read_exactly(tmp_path):
    run = new_run(tmp_path)
    s = read_settings(run)
    assert all(v.source == 'recorded' for v in s.values())
    assert s['speed_penalty'] == Value(3.0, 'recorded') and s['levels'].value == list(range(1, 31))
    assert s['iterations'] == Value(1000, 'recorded') and s['panic_score'].value == -50.0


def test_changes_accept_dest_or_flag_names_and_report_what_changed(tmp_path):
    run = new_run(tmp_path)
    base = read_settings(run)
    new = apply_changes(base, {'speed-penalty': '2', '--levels': '1-5'})
    assert changed(base, new) == [('levels', list(range(1, 31)), [1, 2, 3, 4, 5]), ('speed_penalty', 3.0, 2.0)]
    assert new['speed_penalty'] == Value(2.0, 'recorded')
    assert apply_changes(base, {'health': 12, 'fixed_start': True})['fixed_start'].value is True


@pytest.mark.parametrize('bad', [{'speed_penalty': 'fast'}, {'levels': '1-x'}, {'no_such': '1'},
                                 {'health': '2.5'}, {'fixed_start': 'maybe'}, {'levels': ''}])
def test_bad_names_and_values_are_refused(tmp_path, bad):
    run = new_run(tmp_path)
    with pytest.raises(ValueError):
        apply_changes(read_settings(run), bad)


def test_the_command_line_round_trips_through_the_trainer_parser(tmp_path):
    run = new_run(tmp_path)
    values = apply_changes(read_settings(run), {'fixed_start': True, 'panic_score': 'none'})
    parsed = vars(ppo.build_parser().parse_args(to_argv(values)))
    for dest, v in values.items():
        expected = tuple(v.value) if dest == 'levels' and v.value else v.value
        assert parsed[dest] == expected, dest


def test_every_parser_flag_is_a_setting():
    assert {s.dest for s in SETTINGS} == {a.dest for a in ppo.build_parser()._actions} - {'start', 'out', 'resume', 'help', 'actor_scale'}
    assert {s.group for s in SETTINGS} <= {'Rewards', 'Episodes', 'Training'}


def test_old_rules_share_is_not_a_setting_or_a_new_training_argument(tmp_path):
    args = args_for(['--levels', '25'])
    args['rules_share'] = 0.05
    run = make_run(tmp_path, 'run68', {'args': args})
    values = read_settings(run)
    assert 'rules_share' not in values
    assert '--rules-share' not in to_argv(values)
    assert values['levels'].value == [25]


@pytest.mark.parametrize('name', ['gamma', 'gae_lambda'])
@pytest.mark.parametrize('value', ['-0.1', '1.1', 'nan', 'inf'])
def test_credit_settings_reject_values_outside_zero_to_one(tmp_path, name, value):
    run = new_run(tmp_path)
    with pytest.raises(ValueError, match='between 0 and 1'):
        apply_changes(read_settings(run), {name: value})


def test_an_option_a_run_predates_reads_as_the_trainers_default(tmp_path):
    args = args_for(NEW_ARGV)
    del args['gamma'], args['jelly_catch_reward']
    run = make_run(tmp_path, 'run60', {'args': args})
    s = read_settings(run)
    assert s['gamma'] == Value(0.99, 'default') and s['jelly_catch_reward'].source == 'default'
    assert s['speed_penalty'] == Value(3.0, 'recorded')
    assert changed(s, apply_changes(s, {})) == []


@pytest.mark.parametrize('config', [None, {'rewards': [], 'episode': {'episodes_per_iteration': 96}},
                                    {'shaping': {'speed_penalty': 3.0}}])
def test_a_run_that_records_no_arguments_has_no_settings(tmp_path, config):
    run = make_run(tmp_path, 'run40', config)
    with pytest.raises(NoSettings, match='records no settings'):
        read_settings(run)


def test_a_run_that_records_no_arguments_is_not_a_settings_source(tmp_path):
    from tests.test_trainer_control import make_run as checkpoint_run
    from tools.trainer.control import TrainerError, start
    old = checkpoint_run(tmp_path, 'run1', 1)
    (old / 'config.json').write_text(json.dumps({'rewards': [], 'shaping': {'speed_penalty': 3.0}}))
    with pytest.raises(TrainerError, match='records no settings'):
        start(from_='run1', root=tmp_path, processes=[], launch=lambda *a: None)
