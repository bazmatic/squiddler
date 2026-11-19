"""The Sea Snake's reward settings, declared once (see learning.reward_table).

Each paying term pays on one event that learning.adapters.sea_snake_scenario counts per decision. Distances are in
the snake's own radii, so the weights hold whatever size the snake is drawn at. A Sea Snake's penalties are sizes,
subtracted from its reward. The table's order is the order a run's manifest records them in.
"""
import math

from learning.reward_table import RewardKind, RewardTable, RewardTerm

REWARD, PENALTY = RewardKind.REWARD, RewardKind.PENALTY

TABLE = RewardTable((
    RewardTerm('progress_reward', REWARD, .3, 'Progress reward',
               'Per snake radius the head closes on the target; moving away pays it back in full.', 'progress_radii'),
    RewardTerm('sting_reward', REWARD, 5., 'Sting reward',
               'Touching the target with any part of the body (a sting); the episode ends on it.', 'stings'),
    # Per second, in proportion to the mean bend the muscles hold (0: limp, 1: every joint fully bent).
    RewardTerm('effort_cost', PENALTY, .02, 'Effort cost',
               'Per second of full muscle effort; a gentle wave costs a fraction of it.', 'effort_seconds'),
    RewardTerm('wall_cost', PENALTY, .2, 'Wall cost',
               'Per second with any part of the body pressed against a wall.', 'wall_seconds'),
    RewardTerm('reset_penalty', PENALTY, 8., 'Reset penalty',
               'Each time the physics solver blows up and the snake has to be laid out again.', 'resets'),
    RewardTerm('time_cost', PENALTY, .01, 'Time cost',
               'Per second the target is still unstung, so stinging sooner pays.', 'seconds'),
    # Swimming like a snake (learning.sea_snake_wriggle.frame_terms). The two rewards pay only while the snake swims
    # forward and closes on its target, at most once per second of the episode, so they stay well below a sting and
    # cannot be farmed by wriggling in place.
    RewardTerm('wave_reward', REWARD, .05, 'Wave reward',
               'Per second of bends travelling head to tail (in proportion to the wave, capped), while swimming '
               'forward and closing on the target.', 'wave_seconds'),
    RewardTerm('curvature_reward', REWARD, .03, 'Curvature reward',
               'Per second with bends of a healthy size, neither rigid nor folded, while swimming forward and closing '
               'on the target.', 'curved_seconds'),
    RewardTerm('posture_cost', PENALTY, .1, 'Posture cost',
               'Per second swimming rigid-straight, or coiled or folded up.', 'posture_seconds'),
    RewardTerm('twitch_cost', PENALTY, .05, 'Twitch cost',
               'Per frame, on the joint acceleration beyond what a smooth wave needs: jerky, twitching bends.',
               'twitch'),
), prefix='sea_snake_', help_prefix='Sea snake: ', signed=False)

DEFAULTS = TABLE.defaults
SETTINGS = TABLE.settings


def validate(values):
    if set(values)-set(DEFAULTS):
        raise ValueError('unknown sea snake reward setting')
    result = dict(DEFAULTS, **values)
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('sea snake rewards must be finite and nonnegative')
    return result


def from_args(args):
    return validate({n: getattr(args, 'sea_snake_'+n, v) for n, v in DEFAULTS.items()})


def recovered_settings(config):
    return {'sea_snake_'+n: v for n, v in validate(config.get('manifest', {}).get('rewards', {})).items()}
