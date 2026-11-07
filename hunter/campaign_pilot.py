"""The Campaign Hunter: a neural pilot that flies the player ship."""
import logging
import os
from dataclasses import dataclass

from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES, CampaignSenses, CampaignSurroundings, ShipBody
from hunter.neural.controller import NeuralController
from hunter.neural.loader import DEFAULT_PATH as HUNTER_POLICY_PATH
from hunter.neural.policy import InvalidPolicy, NeuralPolicy
from hunter.settings_for import hunter_settings_for
from tools.migrate_policy import migrated
from utils.resource_path import resource_path

CAMPAIGN_POLICY_PATH = resource_path('assets/neural/campaign_hunter.npz')
_warned = set()


@dataclass(frozen=True)
class ShipCommand:
    """What the pilot does this frame: turn -1 left, 0 straight, +1 right."""
    turn: int
    thrust: bool
    fire: bool


class CampaignPilot:
    """Flies the game's ship with a policy on the campaign senses, deciding every `decision_frames` frames and
    holding the controls between. A pilot with no policy holds still."""

    def __init__(self, policy, decision_frames, greedy=True, rng=None):
        self.settings = hunter_settings_for('neural')
        self.controller = NeuralController(policy, decision_frames, greedy=greedy, rng=rng)
        self._senses = CampaignSenses(self.settings)
        self._listeners = []

    def on_observed(self, listener):
        """Call `listener(senses, last_controls)` at each decision, before the policy chooses from those senses
        (`controller.last_controls` holds the choice afterwards)."""
        self._listeners.append(listener)

    def reset(self):
        """A new level: forget the memory and what was seen."""
        self.controller.invalidate()
        self._senses.reset()

    def decide(self, game):
        body = ShipBody(game.ship, game, self.settings)
        surroundings = CampaignSurroundings(game)

        def sense(last_controls, now):
            senses = self._senses.perceive(body, surroundings, last_controls, now)
            for listener in tuple(self._listeners):
                listener(senses, last_controls)
            return senses

        action = self.controller.tick(sense)
        return ShipCommand(action.turn, action.thrust, action.fire)


def load_campaign_policy(path=None):
    """The Campaign Hunter's policy. Until one is shipped, the hunter pilot carried over to the campaign senses
    (blind to the exit and the beacon, so it flies as the allied hunter does). None, with a warning, when neither
    loads."""
    path = str(path or CAMPAIGN_POLICY_PATH)
    log = logging.getLogger(__name__)
    if os.path.exists(path):
        try:
            return NeuralPolicy.load(path, CAMPAIGN_FEATURE_NAMES)
        except InvalidPolicy as exc:
            _warn_once(log, path, 'Campaign pilot unusable (%s): %s; flying the hunter pilot', exc)
    try:
        return migrated(HUNTER_POLICY_PATH, CAMPAIGN_FEATURE_NAMES)
    except (InvalidPolicy, ValueError, OSError, KeyError) as exc:
        _warn_once(log, HUNTER_POLICY_PATH, 'Campaign pilot unavailable (%s): %s', exc)
        return None


def _warn_once(log, path, message, exc):
    if path not in _warned:
        _warned.add(path)
        log.warning(message, path, exc)
