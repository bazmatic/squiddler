import io
import json
from pathlib import Path

from tools.trainer import __main__ as cli
from tools.trainer.control import Started, TrainerError
from tools.trainer.runs import Run


def run(argv):
    out = io.StringIO()
    code = cli.main(argv, out=out)
    return code, out.getvalue()


def test_status_json_is_the_engine_status(monkeypatch):
    monkeypatch.setattr(cli.runs, 'status', lambda: {'state': 'training', 'run': 'run52', 'iteration': 3})
    code, text = run(['status', '--json'])
    assert code == 0 and json.loads(text) == {'state': 'training', 'run': 'run52', 'iteration': 3}


def test_status_text_names_the_run_and_progress(monkeypatch):
    monkeypatch.setattr(cli.runs, 'status', lambda: {
        'state': 'training', 'run': 'run52', 'iteration': 121, 'iterations': 1000, 'seconds_per_iteration': 20.7,
        'started': '2026-10-22T18:54:40', 'latest': {'score': 74.0, 'kills_per_min': 16.6}})
    code, text = run(['status'])
    assert code == 0 and 'run52' in text and 'training' in text and '121' in text and 'score' in text


def test_start_passes_from_and_set_and_prints_the_changes(monkeypatch):
    seen = {}

    def start(from_=None, changes=None):
        seen['args'] = (from_, changes)
        return Started('run53', 'run51/iter0278', [('speed_penalty', 3.0, 2.0)], [])

    monkeypatch.setattr(cli.control, 'start', start)
    code, text = run(['start', '--from', 'run51', '--set', 'speed_penalty=2', '--set', 'levels=1-5'])
    assert code == 0 and seen['args'] == ('run51', {'speed_penalty': '2', 'levels': '1-5'})
    assert 'run53' in text and 'run51/iter0278' in text and 'speed_penalty: 3.0 → 2.0' in text


def test_level_changes_read_as_ranges(monkeypatch):
    monkeypatch.setattr(cli.control, 'start', lambda from_=None, changes=None: Started(
        'run53', 'run52/iter0001', [('levels', [1, 2, 3, 7], [1, 2, 3, 4, 5])]))
    assert 'levels: 1-3,7 → 1-5' in run(['start'])[1]


def test_a_start_with_no_changes_says_so(monkeypatch):
    monkeypatch.setattr(cli.control, 'start', lambda from_=None, changes=None: Started('run53', 'run52/iter0001'))
    assert 'no settings changed' in run(['start'])[1]


def test_a_set_without_equals_is_an_error(monkeypatch):
    monkeypatch.setattr(cli.control, 'start', lambda **_: (_ for _ in ()).throw(AssertionError('started')))
    code, text = run(['start', '--set', 'speed_penalty'])
    assert code == 1 and 'NAME=VALUE' in text


def test_trainer_errors_print_and_return_one(monkeypatch):
    def pause():
        raise TrainerError('nothing is training')
    monkeypatch.setattr(cli.control, 'pause', pause)
    assert run(['pause']) == (1, 'error: nothing is training\n')


def test_stop_and_resume_name_the_run(monkeypatch):
    monkeypatch.setattr(cli.control, 'stop', lambda: 'run52')
    monkeypatch.setattr(cli.control, 'resume', lambda run=None: run or 'run52')
    assert 'run52' in run(['stop'])[1]
    assert 'run51' in run(['resume', 'run51'])[1]


def test_runs_lists_name_state_progress_and_source(monkeypatch, tmp_path):
    listed = [Run('run51', 51, tmp_path / 'run51', 'stopped', 279, None, 'run50/iter0117', ['iter0278']),
              Run('run52', 52, tmp_path / 'run52', 'training', 122, 1000, 'run51/iter0278', ['iter0121'], 9)]
    monkeypatch.setattr(cli.runs, 'list_runs', lambda: listed)
    monkeypatch.setattr(cli, 'run_changes', lambda run, all_runs: [('speed_penalty', 3.0, 2.0)] if run.name == 'run52' else [])
    code, text = run(['runs'])
    lines = text.splitlines()
    assert code == 0 and 'run51' in lines[-2] and 'stopped' in lines[-2] and '279' in lines[-2]
    assert 'run52' in lines[-1] and '122/1000' in lines[-1] and 'run51/iter0278' in lines[-1]
    assert 'speed_penalty 3.0 → 2.0' in lines[-1]
    data = json.loads(run(['runs', '--json'])[1])
    assert data[1]['name'] == 'run52' and data[1]['changes'] == [['speed_penalty', 3.0, 2.0]]


def test_run_changes_compare_with_the_run_it_started_from(tmp_path):
    from tests.test_trainer_control import make_run
    make_run(tmp_path, 'run1', 1)
    new = make_run(tmp_path, 'run2', 1)
    config = json.loads((new / 'config.json').read_text())
    config['args']['speed_penalty'] = 2.0
    config['args']['start'] = str(tmp_path / 'run1' / 'iter0000.npz')
    (new / 'config.json').write_text(json.dumps(config))
    from tools.trainer.runs import list_runs
    listed = list_runs(tmp_path, [])
    assert cli.run_changes(listed[1], [r.path for r in listed]) == [('speed_penalty', 3.0, 2.0)]
    assert cli.run_changes(listed[0], [r.path for r in listed]) == []
