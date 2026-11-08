import json
import signal

from tools.trainer.processes import Process, descendants, terminate_tree, training
from tools.trainer.runs import ROOT, list_runs, log_rows, next_run_number, status

LINE = 'policy_loss=0.1 iteration={i} score=70.5 kills_per_min=16.0 rollout_s=18.0 update_s=3.0 rules_beaten=0.000'


def make_run(root, name, iterations=0, files=(), config=None):
    path = root / name
    path.mkdir(parents=True)
    for i in range(iterations):
        (path / f'iter{i:04d}.npz').write_bytes(b'')
    if iterations:
        (path / 'train.log').write_text(''.join(LINE.format(i=i) + '\n' for i in range(iterations)))
    for name_, text in dict(files).items():
        (path / name_).write_text(text)
    (path / 'config.json').write_text(json.dumps(config or {'args': {'iterations': 10}, 'start': 'x.npz'}))
    return path


def trainer(pid, out):
    return Process(pid, 1, f'/v/bin/python -m tools.ppo_neural --start x.npz --out {out} --iterations 5')


def by_name(runs):
    return {run.name: run for run in runs}


def test_a_process_with_this_out_folder_is_training(tmp_path):
    procs = [trainer(10, tmp_path / 'run3'), Process(11, 10, 'python -c from multiprocessing.spawn import spawn_main'),
             Process(12, 1, 'python serve.py 8765')]
    assert training(procs, tmp_path) == {'run3': 10}
    assert descendants(10, procs) == [11]


def test_a_relative_out_folder_is_resolved_from_the_repo_root():
    root = ROOT / '.astrovoid-local' / 'neural' / 'rl'
    assert training([trainer(10, '.astrovoid-local/neural/rl/run9')], root) == {'run9': 10}
    assert training([trainer(10, 'elsewhere/run9')], root) == {}


def test_states_come_from_process_then_state_file_then_resume_state(tmp_path):
    make_run(tmp_path, 'run1', 1)
    make_run(tmp_path, 'run2', 1, {'resume.pt': ''})
    make_run(tmp_path, 'run3', 1, {'resume.pt': '', 'state.json': '{"state": "paused", "iteration": 0}'})
    make_run(tmp_path, 'run4', 1, {'resume.pt': '', 'state.json': '{"state": "finished", "iteration": 0}'})
    make_run(tmp_path, 'run5', 1, {'resume.pt': ''})
    make_run(tmp_path, 'run6', 1, {'resume.pt': '', 'PAUSE': ''})
    runs = by_name(list_runs(tmp_path, [trainer(50, tmp_path / 'run5'), trainer(60, tmp_path / 'run6')]))
    assert {n: r.state for n, r in runs.items()} == {
        'run1': 'stopped', 'run2': 'interrupted', 'run3': 'paused', 'run4': 'finished',
        'run5': 'training', 'run6': 'pausing'}
    assert runs['run5'].pid == 50 and runs['run1'].pid is None
    assert [r.name for r in list_runs(tmp_path, [])] == [f'run{i}' for i in range(1, 7)]


def test_runs_are_ordered_by_number_not_name(tmp_path):
    for n in (2, 10, 1):
        make_run(tmp_path, f'run{n}', 1)
    assert [r.name for r in list_runs(tmp_path, [])] == ['run1', 'run2', 'run10']


def test_a_stale_pause_request_on_a_dead_run_is_not_pausing(tmp_path):
    make_run(tmp_path, 'run1', 1, {'resume.pt': '', 'PAUSE': ''})
    assert list_runs(tmp_path, [])[0].state == 'interrupted'


def test_a_half_written_or_missing_log_is_tolerated(tmp_path):
    run = make_run(tmp_path, 'run1', 2)
    with open(run / 'train.log', 'a') as log:
        log.write('policy_loss=0.1 iterat')
    assert [row['iteration'] for row in log_rows(run)] == [0.0, 1.0]
    assert log_rows(make_run(tmp_path, 'run2')) == []
    (run / 'train.log').write_text('pygame 2.6.1\n' + LINE.format(i=0) + '\nweird=line=here iteration=1\n')
    assert len(log_rows(run)) == 1


def test_a_run_with_no_checkpoints_is_listed_and_numbers_continue(tmp_path):
    make_run(tmp_path, 'run6', 2)
    (tmp_path / 'run7').mkdir()
    (tmp_path / 'run7.log').write_text('')
    runs = by_name(list_runs(tmp_path, []))
    assert runs['run7'].checkpoints == [] and runs['run7'].iterations_done == 0
    assert runs['run6'].checkpoints == ['iter0000', 'iter0001'] and runs['run6'].total == 10
    assert next_run_number(tmp_path) == 8
    assert next_run_number(tmp_path / 'empty') == 1


def test_status_reports_the_training_run(tmp_path):
    make_run(tmp_path, 'run1', 3)
    make_run(tmp_path, 'run2', 2)
    s = status(tmp_path, [trainer(10, tmp_path / 'run2')])
    assert s['state'] == 'training' and s['run'] == 'run2' and s['iteration'] == 1 and s['iterations'] == 10
    assert s['seconds_per_iteration'] == 21.0 and s['latest']['score'] == 70.5 and s['started']


def test_status_is_idle_with_the_newest_run_when_nothing_trains(tmp_path):
    make_run(tmp_path, 'run1', 3)
    make_run(tmp_path, 'run2', 2)
    s = status(tmp_path, [])
    assert s['state'] == 'idle' and s['run'] == 'run2' and s['iteration'] == 1
    make_run(tmp_path, 'run3', 1, {'resume.pt': '', 'state.json': '{"state": "paused", "iteration": 0}'})
    assert status(tmp_path, [])['state'] == 'paused'
    assert status(tmp_path / 'none', [])['state'] == 'idle'


def test_terminate_tree_terms_then_kills_what_is_left():
    procs = [Process(10, 1, 'trainer'), Process(11, 10, 'worker')]
    sent, alive = [], {10, 11}

    def kill(pid, sig):
        if sig == 0:
            if pid not in alive:
                raise ProcessLookupError
            return
        sent.append((pid, sig))
        if pid == 11 or sig == signal.SIGKILL:
            alive.discard(pid)

    terminate_tree(10, procs, kill=kill, wait=0.05)
    assert (10, signal.SIGTERM) in sent and (11, signal.SIGTERM) in sent and (10, signal.SIGKILL) in sent
    assert (11, signal.SIGKILL) not in sent


def test_a_last_line_without_its_newline_is_not_counted(tmp_path):
    run = make_run(tmp_path, 'run1', 1)
    with open(run / 'train.log', 'a') as log:
        log.write('policy_loss=0.1 iteration=1 kl=0.2 update_s=1.0 compass_follow=')
    assert [row['iteration'] for row in log_rows(run)] == [0.0]


def test_only_the_trainer_itself_counts_as_training_not_commands_that_mention_it(tmp_path):
    out = tmp_path / 'run3'
    procs = [Process(20, 1, f'zsh -c nohup ./venv/bin/python -m tools.ppo_neural --out {out} &'),
             Process(21, 1, f'claude -p run python -m tools.ppo_neural --out {out}'),
             Process(22, 1, f'grep tools.ppo_neural --out {out}')]
    assert training(procs, tmp_path) == {}
    assert training([Process(23, 1, f'/v/bin/python3.10 tools/ppo_neural.py --out {out}')], tmp_path) == {'run3': 23}


def test_status_reports_evaluation_and_worker_cleanup_phases(tmp_path):
    run=make_run(tmp_path,'run1',1,{'state.json':'{"state":"evaluating","iteration":0}'})
    processes=[trainer(10,run)]
    assert status(tmp_path,processes)['phase']=='evaluating'
    (run/'state.json').write_text('{"state":"finished","iteration":0}')
    assert status(tmp_path,processes)['phase']=='finished'
    assert status(tmp_path,[])['state']=='idle'


def test_listing_runs_again_does_not_reparse_unchanged_logs_but_picks_up_a_log_that_grew(tmp_path, monkeypatch):
    from tools.trainer import runs
    for n in (1, 2, 3):
        make_run(tmp_path, f'run{n}', iterations=n)
    parsed = []
    real = runs._parse_log
    monkeypatch.setattr(runs, '_parse_log', lambda path: parsed.append(path.parent.name) or real(path))
    first = list_runs(tmp_path, processes=[])
    assert sorted(parsed) == ['run1', 'run2', 'run3']
    parsed.clear()
    assert list_runs(tmp_path, processes=[]) == first and parsed == []
    with open(tmp_path / 'run2' / 'train.log', 'a') as log:
        log.write(LINE.format(i=2) + '\n')
    again = by_name(list_runs(tmp_path, processes=[]))
    assert parsed == ['run2'] and again['run2'].iterations_done == 3 and again['run1'].iterations_done == 1


def test_cached_listing_equals_a_fresh_parse_and_callers_cannot_corrupt_it(tmp_path):
    from tools.trainer import runs
    make_run(tmp_path, 'run1', iterations=3)
    runs._LOG_CACHE.clear()
    cold = log_rows(tmp_path / 'run1')
    cold.pop()
    cold[0]['score'] = -1
    warm = log_rows(tmp_path / 'run1')
    runs._LOG_CACHE.clear()
    assert len(warm) == 3 and warm[1:] == log_rows(tmp_path / 'run1')[1:]
    assert list_runs(tmp_path, processes=[]) == list_runs(tmp_path, processes=[])
