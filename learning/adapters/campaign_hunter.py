"""Campaign Hunter adapter: the neural pilot flies the real player ship through whole Demo levels.

Its episodes come from experiments.campaign_rollout (imported only when an episode is flown) and its policies use the
campaign senses layout; the hunter's PPO loop in tools.ppo_neural trains it.
"""
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES

FEATURE_NAMES = CAMPAIGN_FEATURE_NAMES
DEFAULT_HORIZON = 180.0       # seconds: long enough to clear a level and reach its exit


def default_levels():
    """The Demo's levels."""
    import config
    return tuple(range(config.SETTINGS.autoplay.firstLevel, config.SETTINGS.autoplay.lastLevel + 1))


def summary(flown):
    """An iteration's numbers for the training log: the share of episodes that left by the exit, that left with no
    enemy left, and that scored out, the ship's own kills per episode, the wall strikes and bullets fired per episode, and the share of decisions with the thruster on."""
    import numpy as np
    results = [e['result'] for e in flown]
    return dict(exited=float(np.mean([r.end == 'exited' for r in results])),
                cleared_exit=float(np.mean([r.end == 'exited' and r.cleared for r in results])),
                scored_out=float(np.mean([r.end == 'scored_out' for r in results])),
                own_kills=float(np.mean([r.own_kills for r in results])),
                wall_crashes=float(np.mean([r.wall_crashes for r in results])),
                shots=float(np.mean([r.shots for r in results])),
                thrust=float(np.mean([np.mean(e['controls'][:, 1]) for e in flown if len(e['controls'])] or [0.0])))


def environment(**settings):
    return CampaignEnvironment(**settings)


class CampaignEnvironment:
    def __init__(self, **settings):
        self.settings = settings

    def run(self, level, seed, student, horizon=DEFAULT_HORIZON, **rewards):
        """One episode flown by `student`, paid with `rewards` (CampaignShaping fields; the rest at their defaults)."""
        from experiments.campaign_rollout import fly_campaign
        from learning.campaign_rewards import CampaignShaping
        return fly_campaign(level, seed, student, horizon=horizon, shaping=CampaignShaping(**rewards))
