import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.test_trainer_control import FakeLaunch, make_run, trainer
from tools.trainer import server as server_module

LINE = 'policy_loss=0.1 iteration={i} score=70.5 rollout_s=18.0 update_s=3.0'


@pytest.fixture
def setup(tmp_path):
    make_run(tmp_path, 'run51', 2)
    run52 = make_run(tmp_path, 'run52', 3)
    (run52 / 'train.log').write_text(''.join(LINE.format(i=i) + '\n' for i in range(3)) + 'policy_loss=0.2 itera')
    state = {'processes': [], 'launch': FakeLaunch()}
    httpd = server_module.make_server(0, root=tmp_path, processes=lambda: state['processes'],
                                      launch=lambda *a: state['launch'](*a))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state['url'] = f'http://127.0.0.1:{httpd.server_address[1]}'
    state['root'] = tmp_path
    yield state
    httpd.shutdown()
    httpd.server_close()


def get(setup, path):
    with urllib.request.urlopen(setup['url'] + path) as response:
        return response.status, response.headers, response.read()


def get_json(setup, path):
    return json.loads(get(setup, path)[2])


def post(setup, path, body=None):
    request = urllib.request.Request(setup['url'] + path, data=json.dumps(body or {}).encode(), method='POST',
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_status_and_runs_are_served(setup):
    assert get_json(setup, '/api/status')['state'] == 'idle'
    setup['processes'] = [trainer(9, setup['root'] / 'run52')]
    status = get_json(setup, '/api/status')
    assert status['state'] == 'training' and status['run'] == 'run52' and status['iteration'] == 2
    runs = get_json(setup, '/api/runs')
    assert [r['name'] for r in runs] == ['run51', 'run52'] and runs[1]['checkpoints'] == ['iter0000', 'iter0001', 'iter0002']
    assert runs[1]['state'] == 'training'
    _, headers, _ = get(setup, '/api/status')
    assert headers['Cache-Control'] == 'no-store'


def test_run_data_skips_bad_lines_and_unknown_runs_are_404(setup):
    assert [row['iteration'] for row in get_json(setup, '/api/runs/run52/data')] == [0, 1, 2]
    assert get_json(setup, '/api/runs/run52/config')['args']['iterations'] == 50
    for path in ('/api/runs/run99/data', '/api/runs/../x/data', '/nope'):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(setup, path)
        assert e.value.code == 404


def test_settings_carry_their_sources_and_the_levels_to_choose_from(setup):
    values = get_json(setup, '/api/runs/run52/settings')
    assert values['speed_penalty'] == {'value': 3.0, 'source': 'recorded'}
    schema = get_json(setup, '/api/settings')
    assert {'dest': 'levels', 'kind': 'levels'}.items() <= next(s for s in schema['settings'] if s['dest'] == 'levels').items()
    assert 1 in schema['levels'] and 25 in schema['levels'] and schema['levels'] == sorted(schema['levels'])


def test_start_posts_from_and_changes_and_returns_the_new_run(setup):
    code, body = post(setup, '/api/start', {'from': 'run51/iter0000', 'changes': {'speed_penalty': 2, 'levels': [1, 2]}})
    assert code == 200 and body['run'] == 'run53' and body['started_from'] == 'run51/iter0000'
    assert body['changes'] == [{'setting': 'levels', 'old': list(range(1, 31)), 'new': [1, 2]},
                               {'setting': 'speed_penalty', 'old': 3.0, 'new': 2.0}]
    assert len(setup['launch'].commands) == 1


def test_a_busy_trainer_is_a_409_with_the_message(setup):
    setup['processes'] = [trainer(9, setup['root'] / 'run52')]
    assert post(setup, '/api/start') == (409, {'error': 'run52 is training; stop or pause it first'})
    assert post(setup, '/api/resume', {'run': None})[0] == 409
    setup['processes'] = []
    assert post(setup, '/api/stop') == (409, {'error': 'nothing is training'})


def test_pause_and_stop_act_on_the_training_run(setup):
    setup['processes'] = [trainer(9, setup['root'] / 'run52')]
    assert post(setup, '/api/pause') == (200, {'run': 'run52'})
    assert (setup['root'] / 'run52' / 'PAUSE').exists()


def test_bad_requests_are_a_400(setup):
    assert post(setup, '/api/start', {'changes': {'speed_penalty': 'fast'}})[0] == 409
    request = urllib.request.Request(setup['url'] + '/api/start', data=b'{nope', method='POST',
                                     headers={'Content-Type': 'application/json'})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(request)
    assert e.value.code == 400


def test_the_page_is_served(setup):
    code, headers, body = get(setup, '/')
    assert code == 200 and headers['Content-Type'].startswith('text/html') and b'id="status"' in body


def test_a_taken_port_moves_to_the_next_one(tmp_path):
    taken = socket.socket()
    taken.bind(('127.0.0.1', 0))
    taken.listen()
    port = taken.getsockname()[1]
    try:
        httpd, used = server_module.open_server(port, root=tmp_path)
        assert used != port and port < used <= port + 10
        httpd.server_close()
    finally:
        taken.close()


def raw_post(setup, path, headers):
    request = urllib.request.Request(setup['url'] + path, data=b'{}', method='POST', headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code


def test_other_web_pages_cannot_drive_the_trainer(setup):
    setup['processes'] = [trainer(9, setup['root'] / 'run52')]
    assert raw_post(setup, '/api/pause', {'Content-Type': 'text/plain'}) == 415          # no-cors simple request
    assert raw_post(setup, '/api/pause', {'Content-Type': 'application/json', 'Origin': 'https://evil.example'}) == 403
    assert raw_post(setup, '/api/pause', {'Content-Type': 'application/json', 'Host': 'evil.example:8765'}) == 403
    assert not (setup['root'] / 'run52' / 'PAUSE').exists()
    port = setup['url'].rsplit(':', 1)[1]
    assert raw_post(setup, '/api/pause', {'Content-Type': 'application/json',
                                         'Origin': f'http://localhost:{port}', 'Host': f'localhost:{port}'}) == 200


def replayable(setup):
    run = setup['root'] / 'run52'
    config = json.loads((run / 'config.json').read_text())
    config['args'].update(episodes=6, levels=[3, 16])
    config.update({'decision_frames': 6, 'turn_rate': 1.0})
    (run / 'config.json').write_text(json.dumps(config))
    return run


def test_an_iterations_episodes_are_listed_with_their_levels(setup):
    replayable(setup)
    code, body = post(setup, '/api/episodes', {'run': 'run52', 'iteration': 2})
    assert code == 200 and [e['episode'] for e in body] == list(range(6))
    assert {e['level'] for e in body} <= {3, 16}
    assert post(setup, '/api/episodes', {'run': 'run52', 'iteration': 3}) == \
        (409, {'error': 'run52 has not finished iteration 3'})
    assert post(setup, '/api/episodes', {'run': '../x', 'iteration': 0})[0] == 409


def test_a_run_from_before_replays_says_so(setup):
    (setup['root'] / 'run52' / 'config.json').write_text(json.dumps({'start': 'x.npz', 'rewards': []}))
    code, body = post(setup, '/api/episodes', {'run': 'run52', 'iteration': 1})
    assert code == 409 and body['error'].startswith('run52 cannot be replayed')


def test_an_unexpected_replay_failure_answers_with_a_json_error(setup):
    """A server whose task list predates the run's task (a dashboard left running across a new task) raised
    ValueError, which dropped the connection: the page could only say "Failed to fetch"."""
    run = replayable(setup)
    config = json.loads((run / 'config.json').read_text())
    (run / 'config.json').write_text(json.dumps(dict(config, task='no_such_task')))
    code, body = post(setup, '/api/episodes', {'run': 'run52', 'iteration': 2})
    assert code == 500 and 'unknown training task: no_such_task' in body['error']


def test_flying_reports_how_each_episode_went(setup, monkeypatch):
    replayable(setup)
    import tools.watch_episode as watch_episode
    monkeypatch.setattr(watch_episode, 'fly_all', lambda run, iteration, workers:
                        [(16, 0.5, -100.0, 1, 'died', 20.0), (3, 0.0, 40.0, 10, 'cleared', 30.0)])
    code, body = post(setup, '/api/episodes', {'run': 'run52', 'iteration': 2, 'fly': True})
    assert code == 200 and body[0] == {'episode': 0, 'level': 16, 'panic': 0.5, 'return': -100.0, 'kills': 1,
                                       'end': 'died', 'seconds': 20.0}


def test_watching_opens_the_episode_at_the_chosen_speed(setup):
    run = replayable(setup)
    assert post(setup, '/api/replay', {'run': 'run52', 'iteration': 2, 'episode': 5, 'speed': 0.5}) == \
        (200, {'watching': 5})
    assert setup['launch'].commands[-1][1:] == ['-m', 'tools.watch_episode', str(run), '2', '5', '--speed', '0.5']
    assert post(setup, '/api/replay', {'run': 'run52', 'iteration': 2, 'episode': 6})[0] == 409
    assert post(setup, '/api/replay', {'run': 'run52', 'iteration': 2, 'episode': 0, 'speed': 4})[0] == 409
    assert len(setup['launch'].commands) == 1


def test_runs_say_whether_they_can_be_replayed(setup):
    replayable(setup)
    (setup['root'] / 'run51' / 'config.json').write_text(json.dumps({'start': 'x.npz', 'rewards': []}))
    assert [(r['name'], r['replayable']) for r in get_json(setup, '/api/runs')] == [('run51', False), ('run52', True)]


def test_explicit_network_bind_accepts_its_own_host_and_origin_only(tmp_path):
    from types import SimpleNamespace
    handler = object.__new__(server_module.make_handler(tmp_path, lambda: [], None))
    handler.server = SimpleNamespace(server_address=('100.76.248.12', 8766))
    handler.headers = {'Host': '100.76.248.12:8766', 'Origin': 'http://100.76.248.12:8766'}
    assert handler.from_this_page()
    handler.headers['Origin'] = 'http://unrelated.example'
    assert not handler.from_this_page()
    handler.headers = {'Host': 'unrelated.example:8766'}
    assert not handler.from_this_page()


def test_tailscale_aliases_accept_replays_and_reject_untrusted_origins(tmp_path):
    from types import SimpleNamespace
    aliases = ['quiescent', 'quiescent.tailf8052b.ts.net']
    handler = object.__new__(server_module.make_handler(tmp_path, lambda: [], None, allowed_hosts=aliases))
    handler.server = SimpleNamespace(server_address=('100.76.248.12', 8766))
    for name in aliases + ['Quiescent']:
        handler.headers = {'Host': f'{name}:8766', 'Origin': f'http://{name}:8766'}
        assert handler.from_this_page()
        handler.headers['Origin'] = 'http://unrelated.example:8766'
        assert not handler.from_this_page()
    handler.headers = {'Host': 'unrelated.example:8766', 'Origin': 'http://unrelated.example:8766'}
    assert not handler.from_this_page()


def test_flying_a_stale_crawler_run_names_the_changed_rules(setup):
    """Old Wallcrawler team runs need their original code revision; the page must say which rules changed."""
    from types import SimpleNamespace
    from learning.adapters import crawler_scenario
    run = setup['root'] / 'run60'
    run.mkdir()
    (run / 'iter0000.npz').write_bytes(b'')
    (run / 'opponent.npz').write_bytes(b'the frozen opponent learning.run copied in')
    manifest = json.loads(json.dumps(crawler_scenario.metadata(SimpleNamespace(opponent=run / 'opponent.npz', horizon=1.0))))
    manifest.setdefault('arenas', [])
    manifest['training'] = {'seed': 0}
    manifest['scenario_version'] = 'an older one'
    (run / 'config.json').write_text(json.dumps(dict(task='wallcrawler_team', manifest=manifest, shaping={},
                                                     episode={'horizon_seconds': 1.0})))
    (run / 'episodes').mkdir()
    (run / 'episodes' / 'iter0000.json').write_text(json.dumps(dict(
        jobs=[dict(level=11, seed=1, team_size=1)], episodes=[{}], actions=[[]])))
    code, body = post(setup, '/api/episodes', {'run': 'run60', 'iteration': 0, 'fly': True})
    assert code == 409
    assert body['error'] == ('run60 iteration 0 cannot be flown again: '
                             'replay rules or schemas differ from the recorded run: scenario_version')


def test_the_server_flies_a_jelly_run_without_a_window_and_still_stops_on_sigterm(tmp_path):
    """Flying ran pygame inside the server on a request thread: macOS refuses a window off the main thread, and
    pygame's signal handlers then left the server deaf to SIGTERM and ctrl-c."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from learning import replay as learning_replay
    from tests.test_replay_general import ROOT, record_jelly_run
    run = tmp_path / 'run7'
    record_jelly_run(run, [dict(level=11, seed=301, team_size=1, stage=1)], horizon=.5)
    (run / 'iter0000.npz').write_bytes(b'')
    expected = json.loads(json.dumps(learning_replay.fly(run, 0)))
    env = {k: v for k, v in os.environ.items() if not k.startswith('SDL_')}     # the server's own environment
    code = ('import sys\nfrom tools.trainer import server\n'
            'httpd = server.make_server(0, root=sys.argv[1], processes=lambda: [])\n'
            'print(httpd.server_address[1], flush=True)\nhttpd.serve_forever()\n')
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT, env=env,
                             stdout=subprocess.PIPE, text=True)
    try:
        port = int(child.stdout.readline())
        request = urllib.request.Request(f'http://127.0.0.1:{port}/api/episodes', method='POST',
                                         data=json.dumps({'run': 'run7', 'iteration': 0, 'fly': True}).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                answer = response.status, json.loads(response.read())
        except urllib.error.HTTPError as e:
            answer = e.code, json.loads(e.read())
        assert answer == (200, expected)
        child.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 10
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        assert child.poll() is not None, 'the server ignored SIGTERM'
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def raw_get(setup, path):
    """A GET that sends `path` exactly as written (urllib would tidy some of it)."""
    import http.client
    connection = http.client.HTTPConnection(setup['url'].split('//', 1)[1])
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        return response.status, response.getheader('Content-Type'), response.read()
    finally:
        connection.close()


def test_static_serves_every_script_and_stylesheet_the_page_loads_with_content_types(setup):
    import re
    status, _, page = raw_get(setup, '/')
    assert status == 200
    wanted = re.findall(r'(?:href|src)="(static/[^"]+)"', page.decode())
    assert 'static/main.js' in wanted and 'static/style.css' in wanted
    kinds = {'.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8'}
    seen = set()
    while wanted:                                       # the page's files and every module they import
        url = wanted.pop()
        if url in seen:
            continue
        seen.add(url)
        status, kind, body = raw_get(setup, '/' + url)
        assert status == 200, url
        assert kind == kinds[url[url.rindex('.'):]], url
        wanted += ['static/' + name for name in re.findall(r"""from\s+['"]\./([^'"]+)['"]""", body.decode())]
    assert len(seen) > 3


@pytest.mark.parametrize('path', ['/static/../server.py', '/static/%2e%2e/server.py', '/static/..%2fserver.py',
                                  '/static/%2E%2E%2Fserver.py', '/static//etc/passwd', '/static/%2fetc%2fpasswd',
                                  '/static/x.html', '/static/x.py', '/static/main.js/', '/static/', '/static',
                                  '/static/missing.js', '/static/main.JS', '/static/.css'])
def test_static_refuses_traversal_encoded_separators_absolute_paths_and_other_files(setup, path):
    status, _, body = raw_get(setup, path)
    assert status == 404
    assert b'make_handler' not in body and b'root:' not in body


def settings_by_task(setup, keys):
    settings = get_json(setup, '/api/settings')['settings']
    return {key: sorted(s['dest'] for s in settings if key in s['tasks']) for key in keys}


def test_hunter_and_wallcrawler_settings_forms_offer_the_same_settings_as_before_capabilities(setup):
    """Recorded from the task-name rules (commit a7ccfee) before the settings scope became a task capability."""
    golden = json.loads((Path(__file__).parent / 'fixtures' / 'trainer_settings_scope.json').read_text())
    golden = {key: golden[key] for key in ('hunter', 'wallcrawler_team')}
    assert settings_by_task(setup, golden) == golden


def test_a_new_task_plugs_in_without_changing_the_dashboard_or_trainer(setup, monkeypatch):
    """Registering a task is all it takes: it is listed, its settings are its own, it starts from fresh weights,
    it refuses actor damping it does not support, and its episodes are listed through its replay hooks."""
    import sys
    import types
    from learning import tasks
    from learning.dashboard import Metric
    from tools.trainer import control
    replay_hooks = types.ModuleType('tests._demo_replay')
    replay_hooks.episodes = lambda path, iteration: [{'episode': 0, 'level': 7, 'seed': 3}]
    dashboard = types.ModuleType('tests._demo_dashboard')
    dashboard.METRICS = (Metric('demo_score', 'Demo score'),)
    for module in (replay_hooks, dashboard):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(tasks.TASKS, 'demo', tasks.Task(
        'demo', 'tests._demo_adapter', 'Demo', fresh_defaults=dict(horizon=30.0), options=('budget_seconds', 'seed'),
        replay_module=replay_hooks.__name__, dashboard=dashboard.__name__,
        capabilities=tasks.Capabilities(fresh_weights=True, actor_damping=False, replay_mode='hooks',
                                        settings_scope='listed')))

    demo = next(t for t in get_json(setup, '/api/tasks') if t['key'] == 'demo')
    assert demo['label'] == 'Demo' and demo['capabilities']['fresh_weights'] is True
    assert 'demo_score' in [m['key'] for m in demo['metrics']]
    assert settings_by_task(setup, ['demo'])['demo'] == sorted(
        ['task', 'iterations', 'episodes', 'workers', 'lr', 'gamma', 'gae_lambda', 'horizon', 'budget_seconds', 'seed'])

    code, body = post(setup, '/api/start', {'from': 'new', 'changes': {'task': 'demo'}})
    assert code == 200 and body['started_from'] == 'random'
    command = setup['launch'].commands[-1]
    assert command[command.index('--task') + 1] == 'demo' and command[command.index('--horizon') + 1] == '30.0'

    run = setup['root'] / 'run60'
    run.mkdir()
    (run / 'iter0000.npz').write_bytes(b'')
    (run / 'config.json').write_text(json.dumps({'task': 'demo', 'args': {'task': 'demo'}}))
    with pytest.raises(control.TrainerError, match='^actor damping is supported only for '):
        control.start('run60', root=setup['root'], processes=[], launch=FakeLaunch(), actor_scale=.5)
    assert post(setup, '/api/episodes', {'run': 'run60', 'iteration': 0}) == \
        (200, [{'episode': 0, 'level': 7, 'seed': 3}])


def test_restart_answers_then_stops_the_server_so_serve_can_start_it_again(setup):
    assert post(setup, '/api/restart') == (200, {'restarting': True})
    for _ in range(50):
        try:
            urllib.request.urlopen(setup['url'] + '/api/status', timeout=.2)
        except (urllib.error.URLError, OSError):
            return
        threading.Event().wait(.1)
    raise AssertionError('the server kept answering after a restart was requested')
