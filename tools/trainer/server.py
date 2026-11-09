"""The training manager's web app. Loopback by default; an explicit host can expose it on Tailscale."""
import json
import os
import re
import sys
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from learning.tasks import UNTAGGED_TASK
from tools.trainer import control, replay, runs
from tools.trainer.control import TrainerError
from tools.trainer.processes import REPO, list_processes
from tools.trainer.settings import NoSettings, config_of, read_settings, scoped_settings

PAGE = Path(__file__).with_name('page.html')
STATIC = Path(__file__).with_name('static').resolve()
STATIC_PATH = re.compile(r'/static/([A-Za-z0-9_-]+\.(js|css))')
STATIC_KINDS = {'js': 'text/javascript; charset=utf-8', 'css': 'text/css; charset=utf-8'}
RESTARTED = 'TRAINER_RESTARTED'    # set across a restart so the new process does not open another browser tab
RUN_PATH = re.compile(r'/api/runs/(run\d+)/(data|config|settings|rewards)')


def available_levels():
    return sorted(int(p.stem) for p in (REPO / 'levels').glob('*.json') if p.stem.isdigit())


def run_json(run):
    return {'name': run.name, 'state': run.state, 'iterations_done': run.iterations_done, 'total': run.total,
            'started_from': run.started_from, 'checkpoints': run.checkpoints,
            'replayable': replay.replayable(run.path)}


def make_handler(root, processes, launch, allowed_hosts=()):
    root = Path(root)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def static_file(self, match):
            """One of the page's own scripts or stylesheets: a plain name, never a path, from the static folder only."""
            file = (STATIC / match.group(1)).resolve()
            if file.parent != STATIC or not file.is_file():
                return self.send(404, {'error': 'not found'})
            return self.send(200, file.read_bytes(), STATIC_KINDS[match.group(2)])

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                return self.send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            static = STATIC_PATH.fullmatch(path)
            if static:
                return self.static_file(static)
            if path == '/api/status':
                return self.send(200, runs.status(root, processes()))
            if path == '/api/runs':
                return self.send(200, [dict(run_json(r), task=(config_of(r.path) or {}).get('task', UNTAGGED_TASK)) for r in runs.list_runs(root, processes())])
            if path == '/api/tasks':
                from learning import dashboard
                return self.send(200, dashboard.resolve_all())
            if path == '/api/settings':
                from learning.tasks import TASKS
                return self.send(200, {'settings': [asdict(s) for s in scoped_settings()], 'levels': available_levels(),
                                       'tasks':[dict(key=t.key,label=t.label,defaults=t.fresh_defaults) for t in TASKS.values()]})
            match = RUN_PATH.fullmatch(path)
            if match and (root / match.group(1)).is_dir():
                run_dir, what = root / match.group(1), match.group(2)
                if what == 'data':
                    return self.send(200, runs.log_rows(run_dir))
                if what == 'config':
                    return self.send(200, config_of(run_dir) or {})
                if what == 'rewards':
                    from learning import dashboard
                    config, rows = config_of(run_dir) or {}, runs.log_rows(run_dir)
                    return self.send(200, dashboard.rewards(config.get('task', UNTAGGED_TASK), config,
                                                            rows[-1] if rows else {}))
                try:
                    values = read_settings(run_dir)
                except NoSettings as exc:
                    return self.send(409, {'error': str(exc)})
                return self.send(200, {dest: asdict(v) for dest, v in values.items()})
            self.send(404, {'error': 'not found'})

        def from_this_page(self):
            """Only this app's own page may act: other sites in the browser can send a plain POST to localhost
            without asking, and a site renamed to point here (DNS rebinding) arrives with its own Host."""
            port = self.server.server_address[1]
            ours = {f'127.0.0.1:{port}', f'localhost:{port}'}
            host = self.server.server_address[0]
            if host not in ('0.0.0.0', '::'):
                ours.add(f'{host}:{port}')
            ours.update(f'{name.lower()}:{port}' for name in allowed_hosts)
            origin = self.headers.get('Origin')
            return self.headers.get('Host', '').lower() in ours and (
                origin is None or origin.split('//', 1)[-1].lower() in ours)

        def do_POST(self):
            if not (self.headers.get('Content-Type') or '').startswith('application/json'):
                return self.send(415, {'error': 'send JSON'})       # forces a browser preflight we never answer
            if not self.from_this_page():
                return self.send(403, {'error': 'only the training manager page can do that'})
            try:
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self.send(400, {'error': 'the request is not a JSON object'})
            path = self.path.split('?')[0]
            try:
                if path == '/api/start':
                    started = control.start(body.get('from'), body.get('changes'), root=root,
                                            processes=processes(), launch=launch)
                    return self.send(200, {'run': started.run, 'started_from': started.started_from,
                                           'changes': [{'setting': d, 'old': o, 'new': n}
                                                       for d, o, n in started.changes]})
                if path == '/api/stop':
                    return self.send(200, {'run': control.stop(root=root, processes=processes())})
                if path == '/api/pause':
                    return self.send(200, {'run': control.pause(root=root, processes=processes())})
                if path == '/api/resume':
                    return self.send(200, {'run': control.resume(body.get('run'), root=root, processes=processes(),
                                                                 launch=launch,changes=body.get('changes'))})
                if path == '/api/restart':
                    # Training is not a child of this server, so it carries on while the dashboard reloads its code.
                    self.server.restart_requested = True
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return self.send(200, {'restarting': True})
                if path == '/api/episodes':
                    if body.get('fly'):
                        return self.send(200, replay.fly(body.get('run'), body.get('iteration'), root=root))
                    return self.send(200, replay.episodes(body.get('run'), body.get('iteration'), root=root))
                if path == '/api/replay':
                    replay.watch(body.get('run'), body.get('iteration'), body.get('episode'), body.get('speed', 1.0),
                                 root=root, launch=launch)
                    return self.send(200, {'watching': body.get('episode')})
            except TrainerError as e:
                return self.send(409, {'error': str(e)})
            except Exception as e:      # an uncaught error drops the connection and the page only sees "Failed to fetch"
                return self.send(500, {'error': f'{path} failed: {type(e).__name__}: {e}'})
            self.send(404, {'error': 'not found'})

        def log_message(self, *args):
            pass

    return Handler


def make_server(port, root=runs.RL_ROOT, host='127.0.0.1', processes=list_processes, launch=None, allowed_hosts=()):
    return ThreadingHTTPServer((host, port), make_handler(root, processes, launch, allowed_hosts))


def open_server(port, root=runs.RL_ROOT, **kwargs):
    """A server on `port`, or the next free port above it (the old dashboard may still hold 8765)."""
    for candidate in range(port, port + 11):
        try:
            return make_server(candidate, root, **kwargs), candidate
        except OSError:
            continue
    raise OSError(f'ports {port}-{port + 10} are all in use')


def serve(port=8765, open_browser=True, root=runs.RL_ROOT, host='127.0.0.1', allowed_hosts=()):
    httpd, used = open_server(port, root, host=host, allowed_hosts=allowed_hosts)
    if used != port:
        print(f'port {port} is in use (the old dashboard?), so using {used}')
    url = f'http://{host}:{used}/'
    print(f'training manager: {url}  (ctrl-c to close it; training carries on)')
    if open_browser and not os.environ.get(RESTARTED):
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    if getattr(httpd, 'restart_requested', False):
        # The dashboard's Restart button: start again as the same command, so edited code is picked up.
        os.environ[RESTARTED] = '1'
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
