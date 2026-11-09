"""./run-trainer.sh [serve | status | runs | start | stop | pause | resume]: manage creature training.

With no command it opens the training manager in the browser."""
import argparse
import json
import sys

from tools.trainer import control, runs
from tools.trainer.control import TrainerError
from tools.trainer.settings import NoSettings, changed, format_levels, read_settings


def run_changes(run, all_runs):
    """The settings `run` changed from the run it carried on from (those it recorded)."""
    if not run.started_from or '/' not in run.started_from:
        return []
    source = run.path.parent / run.started_from.split('/')[0]
    if source not in all_runs:
        return []
    try:
        mine, theirs = read_settings(run.path), read_settings(source)
    except NoSettings:
        return []
    return changed(theirs, {dest: value for dest, value in mine.items() if value.source == 'recorded'})


def _parser():
    parser = argparse.ArgumentParser(prog='run-trainer.sh', description=__doc__)
    commands = parser.add_subparsers(dest='command')
    serve = commands.add_parser('serve', help='open the training manager (the default)')
    serve.add_argument('--port', type=int, default=8765)
    serve.add_argument('--host', default='127.0.0.1', help='address to listen on (use this Mac\'s Tailscale IP for remote access)')
    serve.add_argument('--allowed-host', action='append', default=[],
                       help='additional trusted hostname, without a port (repeat for Tailscale names)')
    serve.add_argument('--no-browser', action='store_true')
    commands.add_parser('status', help='what is training').add_argument('--json', action='store_true')
    commands.add_parser('runs', help='every run').add_argument('--json', action='store_true')
    start = commands.add_parser('start', help='start a new run (refuses while one is training or paused)')
    start.add_argument('--from', dest='from_', help='RUN or RUN/iterNNNN to continue, or new for fresh creature weights (default: newest run)')
    from learning.tasks import TASKS
    damped = ', '.join(t.label for t in TASKS.values() if t.capabilities.actor_damping) or 'no task'
    start.add_argument('--actor-scale', type=float, default=1.,
                       help=f'one-time output-head damping, for {damped} runs only (0 < scale <= 1)')
    start.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', help='change a setting')
    commands.add_parser('stop', help='end the training run')
    commands.add_parser('pause', help='pause the training run after its current iteration')
    commands.add_parser('resume', help='carry on a paused run').add_argument('run', nargs='?')
    return parser


def _status_text(s):
    if not s.get('run'):
        return 'idle: no runs yet'
    total = f" of {s['iterations']}" if s.get('iterations') else ''
    lines = [f"{s['state']}: {s['run']}, iteration {s['iteration'] if s.get('iteration') is not None else '-'}{total}"]
    if s.get('seconds_per_iteration'):
        lines.append(f"{s['seconds_per_iteration']:.1f} s per iteration, started {s.get('started') or '-'}")
    latest = s.get('latest') or {}
    shown = [k for k in ('score', 'mean_return', 'kills_per_min', 'deaths_per_episode', 'speed_fraction', 'hits_per_minute', 'hit_probability', 'accuracy', 'survival') if k in latest]
    if shown:
        lines.append('latest: ' + ', '.join(f'{k} {latest[k]:g}' for k in shown))
    return '\n'.join(lines)


def _shown(value):
    return format_levels(value) if isinstance(value, list) else value


def _change_text(changes):
    return ', '.join(f'{dest} {_shown(old)} → {_shown(new)}' for dest, old, new in changes)


def main(argv=None, out=sys.stdout):
    args = _parser().parse_args(argv)
    say = lambda text: print(text, file=out)
    try:
        if args.command in (None, 'serve'):
            from tools.trainer.server import serve
            serve(getattr(args, 'port', 8765), open_browser=not getattr(args, 'no_browser', False),
                  host=getattr(args, 'host', '127.0.0.1'), allowed_hosts=getattr(args, 'allowed_host', []))
        elif args.command == 'status':
            s = runs.status()
            say(json.dumps(s, indent=1, default=str) if args.json else _status_text(s))
        elif args.command == 'runs':
            listed = runs.list_runs()
            paths = [r.path for r in listed]
            rows = [{'name': r.name, 'state': r.state, 'iterations_done': r.iterations_done, 'total': r.total,
                     'started_from': r.started_from, 'changes': run_changes(r, paths)} for r in listed]
            if args.json:
                say(json.dumps(rows, indent=1, default=str))
            for row in ([] if args.json else rows):
                progress = f"{row['iterations_done']}/{row['total']}" if row['total'] else str(row['iterations_done'])
                say(f"{row['name']:>6}  {row['state']:<11} {progress:>9}  from {row['started_from'] or '-'}"
                    + (f"  ({_change_text(row['changes'])})" if row['changes'] else ''))
        elif args.command == 'start':
            changes = {}
            for item in args.set:
                name, sep, value = item.partition('=')
                if not sep or not name:
                    raise TrainerError(f'--set needs NAME=VALUE, not {item!r}')
                changes[name] = value
            started = control.start(from_=args.from_, changes=changes, **({'actor_scale':args.actor_scale} if args.actor_scale!=1 else {}))
            say(f'started {started.run}, carrying on from {started.started_from}')
            say('\n'.join(f'  {d}: {_shown(o)} → {_shown(n)}' for d, o, n in started.changes) or '  no settings changed')
            if args.actor_scale!=1:
                say(f'  one-time actor output scale: {args.actor_scale:g}')
        elif args.command == 'stop':
            say(f'stopped {control.stop()}')
        elif args.command == 'pause':
            say(f'{control.pause()} will pause once its current iteration is saved')
        elif args.command == 'resume':
            say(f'resumed {control.resume(run=args.run)}')
    except TrainerError as e:
        say(f'error: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
