"""A requested imitation duration trains the saved bootstrap and survives run bookkeeping."""
import json

import numpy as np
import pytest

from tools import ppo_neural


def test_more_bootstrap_passes_improve_teacher_fit_and_are_recorded(tmp_path):
    import torch
    from learning import run
    from learning.adapters import jelly_scenario as scenario
    from learning.policy import RecurrentPolicy
    from learning.training import build_network

    errors = []
    for epochs in (1, 20):
        out = tmp_path / f'passes{epochs}'
        args = ppo_neural.build_parser().parse_args([
            '--task', 'jelly_team', '--start', '', '--out', str(out), '--horizon', '.2',
            '--iterations', '1', '--episodes', '2', '--workers', '1', '--bootstrap-episodes', '2',
            '--bootstrap-epochs', str(epochs), '--budget-seconds', '0'])
        run.train(args)
        config = json.loads((out / 'config.json').read_text())
        assert config['args']['bootstrap_epochs'] == epochs
        assert config['manifest']['training']['bootstrap_epochs'] == epochs
        evaluation = json.loads((out / 'bootstrap-evaluation.json').read_text())
        assert 'catches_per_minute' in evaluation['summary']
        policy = RecurrentPolicy.load(out / 'bootstrap.npz', scenario.SPEC)
        net = build_network(policy, scenario.CRITIC_SIZE)
        episodes = torch.load(out / 'demonstrations.pt', weights_only=False)
        losses = []
        with torch.no_grad():
            for e in episodes:
                predictions, _ = net(torch.as_tensor(np.asarray(e['x'])).unsqueeze(0),
                                     torch.as_tensor(np.asarray(e['critic'])).unsqueeze(0))
                wanted = torch.as_tensor(np.asarray(e['actions'])).tanh()
                for i, name in enumerate(scenario.IMITATION_HEADS):
                    losses.append((predictions[name][0, :, 0].tanh() - wanted[:, i]).square().mean().item())
        errors.append(np.mean(losses))
    assert errors[1] < errors[0] / 2


def test_resume_refuses_a_changed_imitation_duration(tmp_path):
    from learning.run import train
    from learning.adapters import jelly_scenario as scenario
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--out', str(tmp_path), '--resume'])
    args.opponent = str(tmp_path / 'opponent.npz')
    manifest = scenario.metadata(args)
    manifest['training'] = {k: getattr(args, k) for k in
                            ('seed', 'lr', 'gamma', 'gae_lambda', 'episodes', 'bootstrap_episodes')}
    (tmp_path / 'config.json').write_text(json.dumps(dict(task=args.task, manifest=manifest)))
    args.bootstrap_epochs = 20
    with pytest.raises(ValueError, match='rules.*changed'):
        train(args)


def test_old_jelly_run_does_not_borrow_a_newer_runs_imitation_experiment(tmp_path):
    from tools.trainer.settings import read_settings
    old, new = tmp_path / 'run1', tmp_path / 'run2'
    for path, args in ((old, {}), (new, {'bootstrap_epochs': 20})):
        path.mkdir()
        (path / 'config.json').write_text(json.dumps(dict(task='jelly_team', args=args)))
    assert read_settings(old)['bootstrap_epochs'].value == 0          # it predates the option: its default
    assert read_settings(new)['bootstrap_epochs'].value == 20
