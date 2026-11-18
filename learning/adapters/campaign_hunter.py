"""Campaign Hunter adapter: the neural pilot flies the real player ship through whole Demo levels.

Its episodes come from experiments.campaign_rollout (imported only when an episode is flown) and its policies use the
campaign senses layout; the hunter's PPO loop in tools.ppo_neural trains it.
"""
DEFAULT_HORIZON = 180.0       # seconds: long enough to clear a level and reach its exit
# Lifetime defaults preserve historical observations and rewards; new runs opt in below.
RUN_RULES = {'ship_hit_feedback': False, 'graded_shooting': False, 'accuracy_power': 1, 'sticky_targets': False, 'target_recovery': False}


def add_arguments(parser):
    parser.add_argument('--pursuit-lesson', action='store_true',
                        help='find a stationary hidden target on an open route in the real maze; ends at first sight')
    from tools.ppo_neural import parse_levels
    parser.add_argument('--combat-practice-levels', type=parse_levels, default=None,
                        help='in a pursuit-lesson run, fly these levels as full combat episodes instead')


def __getattr__(name):
    """FEATURE_NAMES, the campaign senses layout, loaded when first asked for: the game's modules (and pygame) stay out
    of the trainer's server, which reads this adapter's shaping and episode block."""
    if name == 'FEATURE_NAMES':
        from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
        return CAMPAIGN_FEATURE_NAMES
    raise AttributeError(name)


def default_levels():
    """The Demo's levels."""
    import config
    return tuple(range(config.SETTINGS.autoplay.firstLevel, config.SETTINGS.autoplay.lastLevel + 1))


def shaping_from(args):
    """The episode shaping a run's arguments ask for: its horizon and its rewards (learning.campaign_rewards)."""
    from learning.campaign_rewards import TABLE
    return {'horizon': args.horizon, 'pursuit_lesson': getattr(args, 'pursuit_lesson', False),
            'combat_practice_levels': getattr(args, 'combat_practice_levels', None),
            'ship_hit_feedback': True,
            'graded_shooting': True,
            'accuracy_power': 2,
            'sticky_targets': True,
            'target_recovery': True,
            **TABLE.from_args(args)}


def episode_block(args):
    """A run's episode block, for its dashboard card and its replays."""
    return {'horizon_seconds': args.horizon, 'episodes_per_iteration': args.episodes, 'learning_rate': args.lr,
            'levels': list(args.levels)}


def summary(flown):
    """An iteration's numbers for the training log: the share of episodes that left by the exit, that left with no
    enemy left, and that scored out, the ship's own kills per episode, the wall strikes and bullets fired per episode, and the share of decisions with the thruster on."""
    import numpy as np
    results = [e['result'] for e in flown]
    combat = [r for r in results if not getattr(r, 'pursuit_lesson', False)] or results
    metrics = dict(exited=float(np.mean([r.end == 'exited' for r in combat])),
                cleared_exit=float(np.mean([r.end == 'exited' and r.cleared for r in combat])),
                scored_out=float(np.mean([r.end == 'scored_out' for r in combat])),
                own_kills=float(np.mean([r.own_kills for r in combat])),
                wall_crashes=float(np.mean([r.wall_crashes for r in results])),
                shots=float(np.mean([r.shots for r in results])),
                thrust=float(np.mean([np.mean(e['controls'][:, 1]) for e in flown if len(e['controls'])] or [0.0])))
    searches = [r for r in results if getattr(r, 'started_unseen', False)]
    if searches:
        found = [r for r in searches if r.first_sighting_seconds is not None]
        metrics['search_found'] = len(found) / len(searches)
        # Misses count the full time limit, including early score-outs, so dying sooner cannot look like faster search.
        metrics['search_seconds'] = float(np.mean([r.first_sighting_seconds if r.first_sighting_seconds is not None
                                                   else getattr(r, 'search_horizon_seconds', r.elapsed)
                                                   for r in searches]))
    if results and hasattr(results[0], 'unseen_seconds'):
        metrics['unseen_seconds'] = float(np.mean([r.unseen_seconds for r in results]))
    if results and hasattr(results[0], 'damaging_hits'):
        metrics['damaging_hits'] = float(np.mean([r.damaging_hits for r in results]))
        shots = sum(r.shots for r in results)
        metrics['unaimed_shot_share'] = sum(r.unaimed_shots for r in results) / shots if shots else 0.0
    if results and hasattr(results[0], 'breach_blocks'):
        metrics['breach_blocks'] = float(np.mean([r.breach_blocks for r in results]))
    if results and hasattr(results[0], 'wall_impulse'):
        metrics['wall_impulse'] = float(np.mean([r.wall_impulse for r in results]))
    if flown and all('reward_totals' in e for e in flown):
        for term in ('hunt_progress', 'unseen_progress', 'exit_progress', 'missed_opportunity', 'shot_accuracy'):
            if term == 'shot_accuracy' and not any(getattr(r, 'graded_shooting', False) for r in results):
                continue
            if all(term in e['reward_totals'] for e in flown):
                metrics[term + '_reward'] = float(np.mean([e['reward_totals'][term] for e in flown]))
    if results and all(getattr(r, 'missed_opportunities', None) is not None for r in results):
        metrics['missed_opportunities'] = float(np.mean([r.missed_opportunities for r in results]))
    lessons = [r for r in results if getattr(r, 'pursuit_lesson', False)]
    if lessons:
        metrics['lesson_contact'] = float(np.mean([r.end == 'contact' for r in lessons]))
    return metrics


def environment(**settings):
    return CampaignEnvironment(**settings)


class CampaignEnvironment:
    def __init__(self, **settings):
        self.settings = settings

    def run(self, level, seed, student, horizon=DEFAULT_HORIZON, pursuit_lesson=False,
            combat_practice_levels=None, ship_hit_feedback=True, graded_shooting=True, accuracy_power=2,
            sticky_targets=True, target_recovery=False, **rewards):
        """One episode flown by `student`, paid with `rewards` (CampaignShaping fields; the rest at their defaults)."""
        from experiments.campaign_rollout import fly_campaign
        from learning.campaign_rewards import CampaignShaping
        lesson = pursuit_lesson and level not in (combat_practice_levels or ())
        return fly_campaign(level, seed, student, horizon=horizon, pursuit_lesson=lesson,
                            ship_hit_feedback=ship_hit_feedback,
                            graded_shooting=graded_shooting,
                            accuracy_power=accuracy_power,
                            sticky_targets=sticky_targets, target_recovery=target_recovery,
                            shaping=CampaignShaping(**rewards))
