"""The trainer's entry point chooses by task, through the registry: no conditional on a task's name in the shared
entry point or the trainer control code, and each task's defaults, options and training loop live with the task."""
import ast
from pathlib import Path

import pytest

from learning.tasks import TASKS, Trainer
from tools import ppo_neural

REPO = Path(__file__).resolve().parents[1]
SHARED = [REPO / 'tools' / 'ppo_neural.py', *sorted((REPO / 'tools' / 'trainer').glob('*.py'))]


def _task_name_conditions(path):
    """(line, source) of each comparison in `path` against a registered task's name."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            constants = [n for operand in operands for n in ast.walk(operand) if isinstance(n, ast.Constant)]
            if any(c.value in TASKS for c in constants):
                found.append((node.lineno, ast.get_source_segment(path.read_text(), node)))
    return found


@pytest.mark.parametrize('path', SHARED, ids=lambda p: p.name)
def test_no_conditional_on_a_task_name_remains_in_the_shared_entry_point_or_trainer_control(path):
    assert _task_name_conditions(path) == []


@pytest.mark.parametrize('key,expected', [
    ('hunter', dict(horizon=90.0, bootstrap_episodes=8)),
    ('campaign_hunter', dict(horizon=180.0)),
    ('flocker_team', dict(horizon=60.0, bootstrap_episodes=0)),
    ('jelly_team', dict(horizon=60.0, bootstrap_episodes=8)),
    ('wallcrawler_team', dict(horizon=60.0)),
    ('inchworm', dict(horizon=120.0, workers=4, iterations=5000, episodes=24, bootstrap_episodes=80, gamma=.997)),
])
def test_each_task_keeps_its_command_line_defaults(key, expected):
    args = ppo_neural.build_parser().parse_args(['--task', key])
    assert {k: getattr(args, k) for k in expected} == expected
    # What the command line names wins over the task's default.
    assert ppo_neural.build_parser().parse_args(['--task', key, '--horizon', '7']).horizon == 7.0


def test_a_tasks_own_options_live_with_it():
    from learning import run as scenario_trainer
    from learning.adapters import hunter
    from argparse import ArgumentParser
    for module, dests in ((hunter, {'fixed_start', 'health', 'decoy_player', 'after_clear', 'panic_score',
                                    'panic_temperature'}),
                          (scenario_trainer, {'opponent', 'budget_seconds', 'bootstrap_episodes', 'bootstrap_epochs',
                                              'seed'})):
        parser = ArgumentParser()
        module.add_arguments(parser)
        assert {a.dest for a in parser._actions} - {'help'} == dests
    assert TASKS['hunter'].cli == hunter.__name__


@pytest.mark.parametrize('key', sorted(TASKS))
def test_each_task_trains_with_the_loop_the_registry_names(key, monkeypatch):
    spec = TASKS[key]
    seen = []
    monkeypatch.setattr(spec.trainer.module(), 'train_task', lambda args: seen.append(args.task))
    ppo_neural.main(['--task', key])
    assert seen == [key]
    assert {Trainer.HUNTER_LOOP: ppo_neural, Trainer.SCENARIO: __import__('learning.run').run}[spec.trainer] \
        is spec.trainer.module()


def test_output_head_damping_is_for_a_task_that_takes_it():
    with pytest.raises(ValueError, match='damping'):
        ppo_neural.main(['--task', 'jelly_team', '--actor-scale', '.5'])
