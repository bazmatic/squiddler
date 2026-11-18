"""The practice teacher provides safe labels without changing campaign gameplay."""
import numpy as np
import pytest

from experiments.campaign_exit_dagger import collect
from experiments.campaign_exit_teacher import ExitTeacher
from experiments.campaign_rollout import fly_campaign
from experiments.student import StudentSpec
from hunter.campaign_pilot import ShipCommand
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES


@pytest.fixture(scope='module', autouse=True)
def restore_harness():
    yield
    from experiments import harness
    harness.reset()


@pytest.fixture(scope='module')
def policy():
    from tools.migrate_policy import migrate
    from hunter.neural.loader import DEFAULT_PATH
    import tempfile
    from pathlib import Path
    path = Path(tempfile.mkdtemp()) / 'campaign.npz'
    migrate(str(DEFAULT_PATH), str(path), CAMPAIGN_FEATURE_NAMES)
    return str(path)


def test_teacher_declines_when_exit_closed():
    from types import SimpleNamespace
    game = SimpleNamespace(exit_opened=False, ship=object(), maze=object())
    assert ExitTeacher().decide(game) is None


def test_student_trajectories_receive_teacher_labels_without_intervention(policy):
    episode, labels, interventions = collect(policy, 300, beta=0.0)
    assert interventions == 0
    assert len(labels) == len(episode['senses']) > 0
    assert all(label is None or (label[0] in (0, 1, 2) and label[1] in (0, 1) and label[2] == 0)
               for label in labels)
    assert any(label is not None for label in labels)


def test_teacher_interventions_use_labeled_action_on_real_game(policy):
    episode, labels, interventions = collect(policy, 301, beta=1.0)
    assert interventions == len(labels) > 0
    assert episode['result'].end == 'exited'
    np.testing.assert_array_equal(episode['controls'], np.asarray(labels, dtype=np.int8))


def test_imitation_fit_exports_campaign_policy_without_overwriting_source(policy, tmp_path):
    from experiments.campaign_exit_dagger import fit
    from hunter.neural.policy import NeuralPolicy
    from pathlib import Path
    source_bytes = Path(policy).read_bytes()
    episode, labels, _ = collect(policy, 302, beta=1.0)
    # Short real trajectory: both supervised and rehearsal paths are exercised.
    short = dict(episode, senses=episode['senses'][:24])
    destination = tmp_path / 'candidate.npz'
    fit(policy, [(short, labels[:24])], [short], destination, epochs=1)
    candidate = NeuralPolicy.load(destination, CAMPAIGN_FEATURE_NAMES)
    action, _ = candidate.act(short['senses'][0], candidate.initial_state())
    assert action[0] in (0, 1, 2)
    assert Path(policy).read_bytes() == source_bytes
