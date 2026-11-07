"""Campaign episodes: the real ship flies one level headless, and is paid roughly as a good player would be."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.campaign_rollout import CampaignShaping, fly_campaign
from experiments.student import StudentSpec
from hunter.campaign_pilot import ShipCommand
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
from hunter.neural.loader import DEFAULT_PATH as HUNTER_POLICY_PATH
from tools.migrate_policy import migrate

SHAPING = CampaignShaping()


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    from experiments import harness
    harness.reset()


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / 'campaign.npz'
    migrate(str(HUNTER_POLICY_PATH), str(path), CAMPAIGN_FEATURE_NAMES)
    return StudentSpec(weights=str(path), greedy=False, sample_seed=1)


class ScriptedPilot:
    """Holds still and runs `script(game, decision)` at each decision, reporting blank senses as a pilot would."""

    def __init__(self, script=lambda game, decision: None):
        self.script, self.decisions, self._listeners = script, 0, []

    def on_observed(self, listener):
        self._listeners.append(listener)

    def decide(self, game):
        for listener in self._listeners:
            listener(np.zeros(len(CAMPAIGN_FEATURE_NAMES), dtype=np.float32), (1, 0, 0))
        self.script(game, self.decisions)
        self.decisions += 1
        return ShipCommand(0, False, False)


def into_the_open_exit(game):
    game.exit_opened = True
    game.maze.exit.set_activated(True, game.sound_manager)
    game.ship.x, game.ship.y = game.maze.exit.get_pos()
    game.ship.vx = game.ship.vy = 0.0


def test_a_campaign_episode_is_deterministic_and_well_formed(spec):
    a = fly_campaign(1, 300, spec, horizon=3.0)
    b = fly_campaign(1, 300, spec, horizon=3.0)
    n = len(a['senses'])
    assert n >= 25                                       # 3 s at 10 decisions a second
    assert a['senses'].shape == (n, len(CAMPAIGN_FEATURE_NAMES)) and a['senses'].dtype == np.float32
    assert a['controls'].shape == (n, 3) and a['controls'].dtype == np.int8
    assert a['rewards'].shape == a['temperatures'].shape == (n,)
    assert np.isfinite(a['rewards']).all()
    for key in ('senses', 'controls', 'rewards', 'temperatures'):
        assert np.array_equal(a[key], b[key])
    assert a['result'] == b['result'] and a['result'].end == 'horizon'


def test_exiting_pays_and_clearing_first_pays_more(spec):
    def leave(game, decision):
        if decision == 1:
            into_the_open_exit(game)

    def clear_then_leave(game, decision):
        if decision == 1:
            for enemy in game.roster.alive():
                game.encounter.kill(enemy)
            into_the_open_exit(game)

    left = fly_campaign(1, 300, spec, horizon=5.0, pilot=ScriptedPilot(leave))
    cleared = fly_campaign(1, 300, spec, horizon=5.0, pilot=ScriptedPilot(clear_then_leave))
    assert left['result'].end == cleared['result'].end == 'exited'
    assert left['result'].enemies_at_start > 0 and not left['result'].cleared and cleared['result'].cleared
    assert left['rewards'][-1] == pytest.approx(SHAPING.exit)
    assert cleared['rewards'][-1] == pytest.approx(SHAPING.exit + SHAPING.cleared_exit)


def test_only_the_ships_own_kills_pay(spec):
    def killed_by(source):
        def script(game, decision):
            if decision == 1:
                game.encounter.kill(game.roster.alive()[0], SimpleNamespace(source=source, weapon_id=None))
        return script

    ally = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(killed_by('hunter')))
    own = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(killed_by('player')))
    assert (ally['result'].own_kills, ally['result'].ally_kills) == (0, 1)
    assert (own['result'].own_kills, own['result'].ally_kills) == (1, 0)
    assert own['rewards'].sum() - ally['rewards'].sum() == pytest.approx(SHAPING.own_kill)


def test_scoring_out_ends_the_episode_with_its_penalty(spec, monkeypatch):
    from scoring.system import ScoringSystem
    monkeypatch.setattr(ScoringSystem, 'calculate_current_potential_score',
                        lambda self, *a, **k: {'potential_score': 0, 'score_percentage': 0.0})
    r = fly_campaign(1, 300, spec, horizon=5.0, pilot=ScriptedPilot())
    assert r['result'].end == 'scored_out' and r['result'].elapsed < 1.0
    assert r['rewards'][-1] == pytest.approx(SHAPING.score_out)


def test_a_fetched_beacon_summons_the_neural_ally_whatever_the_setting(spec, monkeypatch):
    # Jev's pilot reads wall time and calls out: an episode must not depend on it.
    from config import SETTINGS
    from entities.jev_beacon import JevBeacon
    monkeypatch.setattr(SETTINGS.game, 'hunterPilot', 'jev')
    allies = []

    def fetch_beacon(game, decision):
        if decision == 1:
            game.powerup_crystals.append(JevBeacon((game.ship.x, game.ship.y)))
        if decision == 120 and game.hunter_flight is not None:      # a beacon forms for 72 frames first
            allies.append(getattr(game.hunter_flight.controller, 'kind', 'worker'))

    r = fly_campaign(13, 300, spec, horizon=2.5, pilot=ScriptedPilot(fetch_beacon))
    assert r['result'].beacon and allies == ['neural']
    assert SETTINGS.game.hunterPilot == 'jev'
