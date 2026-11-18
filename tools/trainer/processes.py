"""The machine's processes: which ones are training, and ending a training process with its workers."""
import os
import shlex
import signal
import subprocess
import time
from collections import namedtuple
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

Process = namedtuple('Process', 'pid ppid command')


def list_processes():
    out = subprocess.run(['ps', '-axww', '-o', 'pid=,ppid=,command='], capture_output=True, text=True, check=True).stdout
    processes = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            processes.append(Process(int(parts[0]), int(parts[1]), parts[2]))
    return processes


def _is_trainer(words):
    """Python running the trainer itself: `python -m tools.ppo_neural` or `python .../tools/ppo_neural.py`. Not a
    shell or an agent whose command line merely mentions it."""
    if not words or not Path(words[0]).name.lower().startswith('python'):
        return False
    if len(words) > 2 and words[1] == '-m' and words[2] == 'tools.ppo_neural':
        return True
    return len(words) > 1 and words[1].replace('\\', '/').endswith('tools/ppo_neural.py')


def _out_folder(command):
    """The --out of a trainer's command line, or None for any other command."""
    if 'ppo_neural' not in command:
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    if not _is_trainer(words):
        return None
    for i, word in enumerate(words):
        if word == '--out' and i + 1 < len(words):
            return words[i + 1]
        if word.startswith('--out='):
            return word[len('--out='):]
    return None


def training(processes, root):
    """Run name -> pid of each process training a run in `root`."""
    root = Path(root).resolve()
    found = {}
    for process in processes:
        out = _out_folder(process.command)
        if out is None:
            continue
        path = Path(out)
        path = (path if path.is_absolute() else REPO / path).resolve()
        if path.parent == root:
            found[path.name] = process.pid
    return found


def descendants(pid, processes):
    children = [p.pid for p in processes if p.ppid == pid]
    return children + [d for child in children for d in descendants(child, processes)]


def terminate_tree(pid, processes, kill=os.kill, wait=10.0):
    """Ask the process and everything it started to end; whatever is still there after `wait` seconds is killed."""
    tree = [pid] + descendants(pid, processes)
    for member in tree:
        try:
            kill(member, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def alive(member):
        try:
            kill(member, 0)
        except ProcessLookupError:
            return False
        return True

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and any(alive(member) for member in tree):
        time.sleep(min(0.1, wait))
    for member in tree:
        if alive(member):
            try:
                kill(member, signal.SIGKILL)
            except ProcessLookupError:
                pass
