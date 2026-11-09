"""A run's life in one place (tools.trainer.lifecycle): which command each state allows, the run a command acts on,
and one refusal for each illegal command, the same from the command line, the page and an agent's MCP tools."""
import json

import pytest

from tests.test_trainer_control import FakeLaunch, make_run, trainer
from tools.trainer import control, lifecycle, runs
from tools.trainer.lifecycle import Command, State


def test_each_state_allows_its_commands():
    assert lifecycle.ALLOWED == {State.TRAINING: (Command.PAUSE, Command.STOP), State.PAUSING: (Command.STOP,),
                                 State.PAUSED: (Command.RESUME, Command.STOP),
                                 State.INTERRUPTED: (Command.RESUME, Command.STOP),
                                 State.FINISHED: (), State.STOPPED: ()}


@pytest.fixture
def root(tmp_path):
    make_run(tmp_path, 'run51', 2)
    make_run(tmp_path, 'run52', 3)
    return tmp_path


def paused(root):
    (root / 'run52' / 'resume.pt').write_text('')
    (root / 'run52' / 'state.json').write_text('{"state": "paused", "iteration": 2}')


@pytest.mark.parametrize('arrange,processes,expected', [
    (lambda root: None, lambda root: [trainer(9, root / 'run52')], ['pause', 'stop']),
    (lambda root: (root / 'run52' / 'PAUSE').touch(), lambda root: [trainer(9, root / 'run52')], ['stop']),
    (paused, lambda root: [], ['resume', 'stop']),
    (lambda root: (root / 'run52' / 'resume.pt').write_text(''), lambda root: [], ['resume', 'stop']),
    (lambda root: None, lambda root: [], []),
])
def test_the_status_offers_the_commands_its_state_allows(root, arrange, processes, expected):
    arrange(root)
    assert runs.status(root, processes(root))['commands'] == expected


def test_a_run_in_its_final_evaluation_can_only_be_stopped(root):
    (root / 'run52' / 'state.json').write_text('{"state": "evaluating"}')
    assert runs.status(root, [trainer(9, root / 'run52')])['commands'] == ['stop']


def test_a_run_already_pausing_is_not_asked_again(root):
    (root / 'run52' / 'PAUSE').touch()
    with pytest.raises(control.TrainerError, match='run52 is already pausing'):
        control.pause(root=root, processes=[trainer(9, root / 'run52')])


@pytest.mark.parametrize('command,arrange,message', [
    ('pause', lambda root: None, 'nothing is training'),
    ('resume', lambda root: None, 'nothing to resume'),
    ('start', paused, 'run52 is paused; resume or stop it first'),
])
def test_an_illegal_command_is_refused_alike_on_the_command_line_the_page_and_mcp(root, monkeypatch, command,
                                                                                 arrange, message):
    import io
    from tools.trainer import __main__ as cli, mcp_server, server as server_module
    from tests.test_trainer_server import post
    import threading
    arrange(root)
    monkeypatch.setattr(control, 'RL_ROOT', root)
    for name in ('start', 'pause', 'resume', 'stop'):
        original = getattr(control, name)
        monkeypatch.setattr(control, name, lambda *a, _f=original, **k: _f(*a, **{**k, 'root': root, 'processes': [],
                                                                               **({'launch': FakeLaunch()} if _f in
                                                                                  (control.start, control.resume)
                                                                                  else {})}))
    # The agent's tool.
    tool = {'start': mcp_server.start_run, 'pause': mcp_server.pause_run, 'resume': mcp_server.resume_run}[command]
    assert tool() == {'error': message}
    # The command line.
    with pytest.raises(control.TrainerError, match=f'^{message}$'):
        getattr(control, command)()
    out = io.StringIO()
    assert cli.main([command], out=out) == 1
    assert out.getvalue().strip() == f'error: {message}'
    # The page.
    httpd = server_module.make_server(0, root=root, processes=lambda: [], launch=FakeLaunch())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, body = post({'url': f'http://127.0.0.1:{httpd.server_address[1]}'}, f'/api/{command}', {})
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert (status, body) == (409, {'error': message})
